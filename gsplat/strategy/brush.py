import math
from dataclasses import dataclass
from typing import Any, Dict, Union, Literal

import torch
from torch import Tensor
import torch.nn.functional as F

from .base import Strategy
from .ops import inject_noise_to_position_brush, inject_noise_to_position_brush_v2, _multinomial_sample, remove
from .ops import normalized_quat_to_rotmat, _update_param_with_optimizer, scale_down_largest_dim

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
        v2: bool = False
    ):   
        if v2:
            # 更新默认参数
            self.refine_every = 200
            # self.grow_grad2d = 0.003
            self.refine_grow_fraction = 0.2
            # 更新noise权重、计算新box
            self.mean_noise_weight = 50
            if not hasattr(self, "bound"):
                self.bound = BoundingBox(params["means"], 0.8)
                print("Center:", self.bound.center)
                print("Extent:", self.bound.extent)
                print("Min:", self.bound.min())
                print("Max:", self.bound.max())
                print("Median Size:", self.bound.median_size())
            # 更新opac衰减和scales衰减
            self.opac_decay = 0.004
            self.scales_decay = 0.002
        
        self._update_state(params, state, info, packed=packed)
        
        if step < self.refine_stop_iter:
            if not v2:
                inject_noise_to_position_brush(
                    params=params,
                    optimizers=optimizers,
                    state=state,
                    scaler=(1.0 - step / self.max_steps) * self.mean_noise_weight * lr,
                )
            else:
                inject_noise_to_position_brush_v2(
                    params=params,
                    optimizers=optimizers,
                    state=state,
                    scaler= self.mean_noise_weight * lr * self.bound.median_size(),
                    max_noise=self.bound.median_size()
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
            
            # 衰减opac和scales
            # TODO: 当前衰减会导致无限接近透明，需要修改
            # if v2:
            #     print("===触发衰减===")
            #     with torch.no_grad():
            #         train_t = step / self.max_steps
            #         t_shrink_strength = 1.0 - train_t

            #         minus_opac = self.opac_decay * t_shrink_strength
            #         scale_scaling = 1.0 - self.scales_decay * t_shrink_strength

            #         new_opac = torch.sigmoid(params["opacities"]) - minus_opac 
            #         new_opac = new_opac.clamp(1e-12, 1.0 - 1e-12) 
            #         params["opacities"] = (new_opac / (1.0 - new_opac + 1e-24)).log()

            #         new_scales = (params["scales"].exp() * scale_scaling).log()
            #         params["scales"] = new_scales

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
        # TODO： 当前梯度与Brush最新版本的普通梯度长度存在差异，需要修改
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
            state: Dict[str, Any],
            v2: bool = False
    ) -> int:
        alpha = torch.sigmoid(params["opacities"])
        is_prune = alpha < self.prune_opa

        if v2:
            max_allowed_bounds = max(self.bound.extent) * 100.
            # 删除过远的,过大的，过小的
            is_far = torch.any((params["means"] - self.bound.center) > max_allowed_bounds, dim=1)
            is_big = torch.any(params["scales"] > max_allowed_bounds, dim=1)
            is_small = torch.any(params["scales"] < 1e-10, dim=1)

            is_prune = torch.logical_or(is_prune, is_far)
            is_prune = torch.logical_or(is_prune, is_big)
            is_prune = torch.logical_or(is_prune, is_small)

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
        old_count = len(params["means"])
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
        v2: bool = False
    ):
        if chosen_inds.numel() == 0:
            return

        old_count = len(params["means"])
        device = params["means"].device

        chosen_inds = torch.unique(chosen_inds)
        assert chosen_inds.max() < old_count, "Index out of bounds"

        # Basic shrink/offset parameters
        scale_offset = math.log(math.sqrt(2.0))  # ~0.3466
        noise_std = 0.5 if not v2 else 1.0

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
                if v2:
                    old_exp_scales = old_log_scales.exp()
                    new_exp_scales = scale_down_largest_dim(old_exp_scales, 0.5)
                    p[chosen_inds] = new_exp_scales.log()
                    p_new = torch.cat([p, new_exp_scales.log()], dim=0) 
                else:
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

     
class BoundingBox:
    def __init__(self, means, percentile=0.8):
        self.center = None
        self.extent = None
        self.means = means
        self.percentile = percentile
        self.__get_bounds()

    @torch.no_grad() 
    def __get_bounds(self):
         # Filter out NaN and infinite values
        valid_means = self.means[torch.isfinite(self.means).all(dim=1)]

        # Split into x, y, z values
        x_vals = valid_means[:, 0]
        y_vals = valid_means[:, 1]
        z_vals = valid_means[:, 2]

        # Get upper and lower percentiles
        lower_idx = int((1.0 - self.percentile) / 2.0 * len(x_vals))
        upper_idx = min(len(x_vals) - 1, int((1.0 + self.percentile) / 2.0 * len(x_vals)))

        # Calculate the percentiles
        x_lower, x_upper = torch.kthvalue(x_vals, lower_idx + 1).values, torch.kthvalue(x_vals, upper_idx + 1).values
        y_lower, y_upper = torch.kthvalue(y_vals, lower_idx + 1).values, torch.kthvalue(y_vals, upper_idx + 1).values
        z_lower, z_upper = torch.kthvalue(z_vals, lower_idx + 1).values, torch.kthvalue(z_vals, upper_idx + 1).values

        # Calculate the center and extent
        self.center = torch.tensor([
            (x_lower + x_upper) / 2.0,
            (y_lower + y_upper) / 2.0,
            (z_lower + z_upper) / 2.0
        ]).numpy()
        self.extent = torch.tensor([
            (x_upper - x_lower) / 2.0,
            (y_upper - y_lower) / 2.0,
            (z_upper - z_lower) / 2.0
        ]).numpy()
    
    def min(self):
        return self.center - self.extent

    def max(self):
        return self.center + self.extent
    
    def median_size(self):
        extents_sorted  = sorted(self.extent)
        return extents_sorted[1] * 2.0


