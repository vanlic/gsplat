import math
from dataclasses import dataclass
from typing import Any, Dict, Union, Literal

import torch
from torch import Tensor
import torch.nn.functional as F

from .base import Strategy
from .ops import inject_noise_to_position_brush, _multinomial_sample, remove
from .ops import normalized_quat_to_rotmat, _update_param_with_optimizer

@dataclass
class BrushStrategy(Strategy):
    """"
    Brush strategy for densitifing gaussian points.
    This strategy can get better results but just like MCMC, it will get many floaters and be more transparent.
    This strategy allows for operations that:
      1. injecting noise
      2. removing low-opacity points
      3. adding new points
      4. using max absgrad instead of mean
    """
    # nosie operations
    mean_noise_weight: float = 1e4

    # density and remove operations
    prune_opa: float = 0.9 / 255.0  
    grow_grad2d: float = 0.0006 # 0.00085
    max_splats: int  =  10_000_000
    max_steps: int = 30_000

    refine_start_iter: int = 0
    growth_stop_iter: int = 12_500  # 仅停止高斯增长
    refine_stop_iter: int = 25_000  # 精炼终止迭代(删除点和替换死高斯)
    refine_every: int = 150
    refine_grow_fraction: float = 0.1

    # other params
    absgrad: bool = True
    key_for_gradient: Literal["means2d", "gradient_2dgs"] = "means2d"
    verbose: bool = False

    def initialize_state(self, scene_scale: float = 1.0) -> Dict[str, Any]:
        """Initialize and return the running state for this strategy.

        The returned state should be passed to the `step_pre_backward()` and
        `step_post_backward()` functions.
        """
        # Postpone the initialization of the state to the first step so that we can
        # put them on the correct device.
        # - grad2d: running accum of the norm of the image plane gradients for each GS.
        # - count: running accum of how many time each GS is visible.
        # - radii: the radii of the GSs (normalized by the image resolution).
        state = {"grad2d": None, "count": None, "scene_scale": scene_scale, "curr_splats": None}

        state["radii"] = None
        return state

    def check_sanity(
            self,
            params: Union[Dict[str, torch.nn.Parameter], torch.nn.ParameterDict],
            optimizers: Dict[str, torch.optim.Optimizer],
    ):
        """Sanity check for the parameters and optimizers.

        Check if:
            * `params` and `optimizers` have the same keys.
            * Each optimizer has exactly one param_group, corresponding to each parameter.
            * The following keys are present: {"means", "scales", "quats", "opacities"}.

        Raises:
            AssertionError: If any of the above conditions is not met.

        .. note::
            It is not required but highly recommended for the user to call this function
            after initializing the strategy to ensure the convention of the parameters
            and optimizers is as expected.
        """

        super().check_sanity(params, optimizers)
        # The following keys are required for this strategy.
        for key in ["means", "scales", "quats", "opacities"]:
            assert key in params, f"{key} is required in params but missing."

    def step_pre_backward(
            self,
            params: Union[Dict[str, torch.nn.Parameter], torch.nn.ParameterDict],
            optimizers: Dict[str, torch.optim.Optimizer],
            state: Dict[str, Any],
            step: int,
            info: Dict[str, Any],
    ):
        """Callback function to be executed before the `loss.backward()` call."""
        assert (
                self.key_for_gradient in info
        ), "The 2D means of the Gaussians is required but missing."
        info[self.key_for_gradient].retain_grad()


    def step_post_backward(
        self,
        params: Union[Dict[str, torch.nn.Parameter], torch.nn.ParameterDict],
        optimizers: Dict[str, torch.optim.Optimizer],
        state: Dict[str, Any],
        step: int,
        lr: float,
        info: Dict[str, Any],
        packed: bool = False,
    ):   
        self._update_state(params, state, info, packed=packed)
        
        if step < self.refine_stop_iter:
            inject_noise_to_position_brush(
                params=params,
                optimizers=optimizers,
                state=state,
                scaler=(1.0 - step / self.max_steps) * self.mean_noise_weight * lr,
            )
        
        if (step > self.refine_start_iter) and (step % self.refine_every == 0):
            n_prune = self._prune_gs(params, optimizers, state)

            replace_ids = self._replace_pruned_gs(
                params=params,
                optimizers=optimizers,
                state=state,
                n_prune=n_prune,
            )

            add_ids = replace_ids

            if step < self.growth_stop_iter:
                add_new_ids = self._sample_high_grad_gs(
                    params=params,
                    optimizers=optimizers,
                    state=state,
                    n_prune=n_prune,
                )

                add_ids = torch.cat([add_ids, add_new_ids], dim=0)
            
            self._add_gs(
                params=params,
                optimizers=optimizers,
                state=state,
                chosen_inds=add_ids,
            )
            if self.verbose:
                print(
                    f"[Refine@step={step}] pruned={n_prune} added={add_ids.numel()} "
                    f"-> effective add={add_ids.numel() - n_prune} #splats={len(params['means'])} "
                )

            # reset stats
            state["grad2d"].zero_()
            state["count"].zero_()
            if state["radii"] is not None:
                state["radii"].zero_()
            
            torch.cuda.empty_cache()


    def _update_state(
            self,
            params: Union[Dict[str, torch.nn.Parameter], torch.nn.ParameterDict],
            state: Dict[str, Any],
            info: Dict[str, Any],
            packed: bool = False,
    ):
        for key in [
            "width",
            "height",
            "n_cameras",
            "radii",
            "gaussian_ids",
            self.key_for_gradient,
        ]:
            assert key in info, f"{key} is required but missing."

        # normalize grads to [-1, 1] screen space
        if self.absgrad:
            grads = info[self.key_for_gradient].absgrad.clone()
        else:
            grads = info[self.key_for_gradient].grad.clone()
        grads[..., 0] *= info["width"] / 2.0 * info["n_cameras"]
        grads[..., 1] *= info["height"] / 2.0 * info["n_cameras"]

        # initialize state on the first run
        n_gaussian = len(list(params.values())[0])

        if state["grad2d"] is None:
            state["grad2d"] = torch.zeros(n_gaussian, device=grads.device)
        if state["count"] is None:
            state["count"] = torch.zeros(n_gaussian, device=grads.device)
        if state["radii"] is None or state["radii"].shape[0] != n_gaussian:
            assert "radii" in info, "radii is required but missing."
            state["radii"] = torch.zeros(n_gaussian, device=grads.device)

        # update the running state
        if packed:
            # grads is [nnz, 2]
            gs_ids = info["gaussian_ids"]  # [nnz]
            radii = info["radii"]  # [nnz]
        else:
            # grads is [C, N, 2]
            sel = (info["radii"] > 0.0).all(dim=-1)  # [C, N]
            gs_ids = torch.where(sel)[1]  # [nnz]
            grads = grads[sel]  # [nnz, 2]
            radii = info["radii"][sel].max(dim=-1).values  # [nnz]
            state["curr_splats"] = sel.float().flatten()

        # state["grad2d"].index_add_(0, gs_ids, grads.norm(dim=-1))
        state["grad2d"][gs_ids] = torch.maximum(
            state["grad2d"][gs_ids],
            grads.norm(dim=-1)
        )
        state["count"].index_add_(
            0, gs_ids, torch.ones_like(gs_ids, dtype=torch.float32)
        )

        # Should be ideally using scatter max
        state["radii"][gs_ids] = torch.maximum(
            state["radii"][gs_ids],
            # normalize radii to [0, 1] screen space
            radii / float(max(info["width"], info["height"])),
        )
    
    @torch.no_grad()
    def _prune_gs(
            self, 
            params: Dict[str, torch.nn.Parameter], 
            optimizers: Dict[str, torch.optim.Optimizer], 
            state: Dict[str, Any]
    ) -> int:
        alpha = torch.sigmoid(params["opacities"])
        is_prune = alpha < self.prune_opa
        n_prune = is_prune.sum().item()
        if n_prune > 0:
            remove(params=params, optimizers=optimizers, state=state, mask=is_prune)
        return n_prune

    @torch.no_grad()
    def _replace_pruned_gs(
        self,
        params: Dict[str, torch.nn.Parameter],
        optimizers: Dict[str, torch.optim.Optimizer],
        state: Dict[str, torch.Tensor],
        n_prune: int,
    ):
        if n_prune <= 0:
            return torch.empty((0), device=params["means"].device, dtype=torch.long)
        
        alpha = torch.sigmoid(params["opacities"])
        weights = alpha.clone().clamp_min(1e-32)

         # 1) Sample `n_prune` from alpha distribution
        chosen_inds = _multinomial_sample(weights, n_prune, replacement=False)  # shape [n_prune]

        return chosen_inds

    @torch.no_grad()
    def _sample_high_grad_gs(
        self,
        params: Dict[str, torch.nn.Parameter],
        optimizers: Dict[str, torch.optim.Optimizer],
        state: Dict[str, torch.Tensor],
        n_prune: int = 0,
    ):
        old_count = params["means"].numel()
        grads = state["grad2d"]

        is_grad_high = grads > self.grow_grad2d
        threshold_count = is_grad_high.sum().item()

        sample_high_grad = int(threshold_count * self.refine_grow_fraction) - n_prune

        growth_count = max(0, min(sample_high_grad, self.max_splats - old_count))

        if growth_count > 0:
            weights = is_grad_high.float() * grads
            chosen_inds = _multinomial_sample(weights, growth_count, replacement=False)

            return chosen_inds
        else:
            return torch.empty((0), device=params["means"].device, dtype=torch.long)
    
    @torch.no_grad()
    def _add_gs(
        self, 
        params: Dict[str, torch.nn.Parameter],
        optimizers: Dict[str, torch.optim.Optimizer],
        state: Dict[str, Any],
        chosen_inds: Tensor,
    ):
        if chosen_inds.numel() == 0:
            return

        old_count = len(params["means"])
        device = params["means"].device

        chosen_inds = torch.unique(chosen_inds)
        assert chosen_inds.max() < old_count, "Index out of bounds"

        # Basic shrink/offset parameters
        scale_offset = math.log(math.sqrt(2.0))  # ~0.3466
        noise_std = 0.5

        # 2) param_fn: Modify p[sampled_inds] in-place, then cat p[sampled_inds] to the end
        # TODO: Try ResGS? 
        def param_fn(name: str, p: Tensor) -> Tensor:
            if name == "means":
                old_means = p[chosen_inds]
                scale = torch.exp(params["scales"][chosen_inds])
                quats = F.normalize(params["quats"][chosen_inds], dim=-1)
                rot_mats = normalized_quat_to_rotmat(quats)
                rand_delta = torch.randn_like(old_means, device=device) * noise_std * scale
                rand_delta = torch.einsum("bij,bj->bi", rot_mats, rand_delta)

                appended = old_means + rand_delta
                p[chosen_inds] = old_means - rand_delta
                p_new = torch.cat([p, appended], dim=0)
            
            elif name == "scales":
                old_log_scales = p[chosen_inds]

                appended = old_log_scales - scale_offset
                p[chosen_inds] = appended
                p_new = torch.cat([p, appended], dim=0)
            
            elif name == "opacities":
                old_raw_opa = p[chosen_inds]
                old_alpha = torch.sigmoid(old_raw_opa)
                new_alpha = 1.0 - torch.sqrt(1.0 - old_alpha.clamp(max=0.9999999))
                new_raw = (new_alpha / (1.0 - new_alpha + 1e-24)).log()

                p[chosen_inds] = new_raw
                p_new = torch.cat([p, new_raw], dim=0)
            
            else:
                chosen_chunk = p[chosen_inds]
                p_new = torch.cat([p, chosen_chunk], dim=0)
            
            return torch.nn.Parameter(p_new, requires_grad=p.requires_grad)

        # 3) optimizer_fn: Append zeros for newly added rows in optimizer state
        def optimizer_fn(key: str, v: Tensor) -> Tensor:
            if v.shape[0] == 0:
                return v
            new_part = torch.zeros_like(v[chosen_inds])
            v_new = torch.cat([v, new_part], dim=0)
            return v_new
        
        _update_param_with_optimizer(param_fn, optimizer_fn, params, optimizers)
        
        # 4) Expand the user state
        new_count = len(params["means"]) - old_count
        if new_count > 0:
            for k, v in state.items():
                if isinstance(v, torch.Tensor) and v.shape[0] == old_count:
                    extra_shape = (len(chosen_inds), *v.shape[1:])
                    v_new = torch.zeros(extra_shape, dtype=v.dtype, device=v.device)
                    state[k] = torch.cat([v, v_new], dim=0)

       


