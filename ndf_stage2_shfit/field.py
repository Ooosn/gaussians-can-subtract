"""A trainable one-channel 3D Gaussian field with degree-3 SH."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from plyfile import PlyData, PlyElement
import torch
from torch import nn

from simple_knn._C import distCUDA2
from utils.general_utils import get_expon_lr_func, inverse_sigmoid
from utils.sh_utils import C0


FORMAT = "error_guided_scalar_sh_gaussians"
VERSION = 1


class ScalarSHGaussianField(nn.Module):
    """Gaussian parameters compatible with ``gaussolve_render``.

    Unlike the discarded opacity-only experiment, all standard Gaussian
    parameters and one-channel SH coefficients are learnable.
    """

    def __init__(
        self,
        xyz: torch.Tensor,
        features_dc: torch.Tensor,
        features_rest: torch.Tensor,
        opacity_logits: torch.Tensor,
        log_scaling: torch.Tensor,
        rotation: torch.Tensor,
        *,
        sh_degree: int = 3,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        super().__init__()
        count = int(xyz.shape[0])
        coefficient_count = (int(sh_degree) + 1) ** 2
        expected = {
            "xyz": (count, 3),
            "features_dc": (count, 1, 1),
            "features_rest": (count, coefficient_count - 1, 1),
            "opacity_logits": (count, 1),
            "log_scaling": (count, 3),
            "rotation": (count, 4),
        }
        actual = {
            "xyz": tuple(xyz.shape),
            "features_dc": tuple(features_dc.shape),
            "features_rest": tuple(features_rest.shape),
            "opacity_logits": tuple(opacity_logits.shape),
            "log_scaling": tuple(log_scaling.shape),
            "rotation": tuple(rotation.shape),
        }
        for name, shape in expected.items():
            if actual[name] != shape:
                raise ValueError(f"{name}: expected {shape}, got {actual[name]}")
        self._xyz = nn.Parameter(xyz.float().contiguous())
        self._features_dc = nn.Parameter(features_dc.float().contiguous())
        self._features_rest = nn.Parameter(features_rest.float().contiguous())
        self._opacity = nn.Parameter(opacity_logits.float().contiguous())
        self._scaling = nn.Parameter(log_scaling.float().contiguous())
        self._rotation = nn.Parameter(rotation.float().contiguous())
        self.active_sh_degree = int(sh_degree)
        self.max_sh_degree = int(sh_degree)
        self.metadata = dict(metadata or {})

    @classmethod
    @torch.no_grad()
    def from_points(
        cls,
        points: torch.Tensor,
        *,
        sh_degree: int = 3,
        initial_opacity: float = 0.01,
        initial_value: float = 0.05,
        metadata: dict[str, Any] | None = None,
    ) -> "ScalarSHGaussianField":
        if points.device.type != "cuda":
            raise ValueError("points must be on CUDA")
        points = points.detach().float().contiguous()
        if points.shape[0] < 4:
            raise ValueError("at least four points are required")
        distance_squared = distCUDA2(points).clamp_min(1e-8)
        base_scale = torch.sqrt(distance_squared)
        # Keep the standard 3DGS initialization exactly: distCUDA2 determines
        # every initial scale.  The filtered COLMAP cloud already removed the
        # pathological far points, so an additional quantile clamp would only
        # change the intended initialization.
        q05 = torch.quantile(base_scale, 0.05)
        q95 = torch.quantile(base_scale, 0.95)
        count = points.shape[0]
        dc = torch.full(
            (count, 1, 1), (float(initial_value) - 0.5) / C0, device=points.device
        )
        rest = torch.zeros(
            (count, (sh_degree + 1) ** 2 - 1, 1), device=points.device
        )
        opacity = inverse_sigmoid(
            torch.full((count, 1), float(initial_opacity), device=points.device)
        )
        log_scaling = torch.log(base_scale).unsqueeze(-1).repeat(1, 3)
        rotation = torch.zeros((count, 4), device=points.device)
        rotation[:, 0] = 1.0
        info = dict(metadata or {})
        info.update(
            {
                "initial_count": int(count),
                "scale_q05": float(q05.item()),
                "scale_q95": float(q95.item()),
                "initial_opacity": float(initial_opacity),
                "initial_value": float(initial_value),
            }
        )
        result = cls(
            points,
            dc,
            rest,
            opacity,
            log_scaling,
            rotation,
            sh_degree=sh_degree,
            metadata=info,
        )
        # Match the original 3DGS/NDF training path: start from DC and expose
        # one additional SH degree every 1k steps.
        result.active_sh_degree = 0
        return result

    @classmethod
    @torch.no_grad()
    def from_surface_vertices(cls, vertices: torch.Tensor, **kwargs):
        """Backward-compatible alias for the original mesh-vertex experiment."""

        return cls.from_points(vertices, **kwargs)

    @property
    def get_xyz(self) -> torch.Tensor:
        return self._xyz

    @property
    def get_features(self) -> torch.Tensor:
        return torch.cat((self._features_dc, self._features_rest), dim=1)

    @property
    def get_opacity(self) -> torch.Tensor:
        return torch.sigmoid(self._opacity)

    @property
    def get_scaling(self) -> torch.Tensor:
        return torch.exp(self._scaling)

    @property
    def get_rotation(self) -> torch.Tensor:
        return torch.nn.functional.normalize(self._rotation, dim=-1)

    def get_covariance(self, scaling_modifier: float = 1.0):
        raise RuntimeError("Stage-2 uses CUDA covariance computation")

    def oneupSHdegree(self) -> int:
        if self.active_sh_degree < self.max_sh_degree:
            self.active_sh_degree += 1
        return self.active_sh_degree

    def optimizer(
        self,
        scene_extent: float,
        total_iterations: int = 30_000,
        position_lr_init: float = 0.00016,
        position_lr_final: float = 0.0000016,
        position_lr_delay_mult: float = 0.01,
        feature_rest_lr_scale: float = 1.0,
    ) -> torch.optim.Adam:
        """Build the same basic optimizer and XYZ schedule as the main 3DGS model."""

        xyz_lr_init = float(position_lr_init) * float(scene_extent)
        xyz_lr_final = float(position_lr_final) * float(scene_extent)
        self.xyz_scheduler = get_expon_lr_func(
            lr_init=xyz_lr_init,
            lr_final=xyz_lr_final,
            lr_delay_mult=float(position_lr_delay_mult),
            max_steps=max(1, int(total_iterations)),
        )
        return torch.optim.Adam(
            [
                {"params": [self._xyz], "lr": xyz_lr_init, "name": "xyz"},
                {"params": [self._features_dc], "lr": 2.5e-3, "name": "f_dc"},
                {
                    "params": [self._features_rest],
                    "lr": 2.5e-3 * float(feature_rest_lr_scale),
                    "name": "f_rest",
                },
                {"params": [self._opacity], "lr": 2.5e-2, "name": "opacity"},
                {"params": [self._scaling], "lr": 5e-3, "name": "scaling"},
                {"params": [self._rotation], "lr": 1e-3, "name": "rotation"},
            ],
            lr=0.0,
            eps=1e-15,
        )

    def update_learning_rate(
        self, optimizer: torch.optim.Adam, iteration: int
    ) -> float:
        if not hasattr(self, "xyz_scheduler"):
            raise RuntimeError("optimizer() must be called before update_learning_rate()")
        lr = float(self.xyz_scheduler(int(iteration)))
        for group in optimizer.param_groups:
            if group["name"] == "xyz":
                group["lr"] = lr
                return lr
        raise RuntimeError("optimizer has no xyz parameter group")

    @torch.no_grad()
    def reset_opacity(
        self,
        optimizer: torch.optim.Adam,
        max_opacity: float = 0.01,
    ) -> int:
        """Apply the standard 3DGS opacity reset and clear its Adam moments."""

        current = self.get_opacity
        reset_count = int((current > float(max_opacity)).sum().item())
        capped = torch.minimum(
            current,
            torch.full_like(current, float(max_opacity)),
        )
        replacement = inverse_sigmoid(capped.clamp(1e-6, 1.0 - 1e-6))
        for group in optimizer.param_groups:
            if group["name"] != "opacity":
                continue
            old_parameter = group["params"][0]
            new_parameter = nn.Parameter(replacement.detach().requires_grad_(True))
            state = optimizer.state.pop(old_parameter, None)
            if state is not None:
                for key, value in tuple(state.items()):
                    if torch.is_tensor(value) and tuple(value.shape) == tuple(old_parameter.shape):
                        state[key] = torch.zeros_like(replacement)
                optimizer.state[new_parameter] = state
            group["params"][0] = new_parameter
            self._opacity = new_parameter
            return reset_count
        raise RuntimeError("optimizer has no opacity parameter group")

    @staticmethod
    def _parameter_attributes() -> dict[str, str]:
        return {
            "xyz": "_xyz",
            "f_dc": "_features_dc",
            "f_rest": "_features_rest",
            "opacity": "_opacity",
            "scaling": "_scaling",
            "rotation": "_rotation",
        }

    @staticmethod
    def _quaternion_to_matrix(quaternion: torch.Tensor) -> torch.Tensor:
        q = torch.nn.functional.normalize(quaternion, dim=-1)
        w, x, y, z = q.unbind(-1)
        return torch.stack(
            (
                1 - 2 * (y * y + z * z),
                2 * (x * y - z * w),
                2 * (x * z + y * w),
                2 * (x * y + z * w),
                1 - 2 * (x * x + z * z),
                2 * (y * z - x * w),
                2 * (x * z - y * w),
                2 * (y * z + x * w),
                1 - 2 * (x * x + y * y),
            ),
            dim=-1,
        ).reshape(-1, 3, 3)

    @torch.no_grad()
    def reset_densification_stats(self) -> None:
        count = self.get_xyz.shape[0]
        device = self.get_xyz.device
        self.xyz_gradient_accum = torch.zeros((count, 1), device=device)
        self.denom = torch.zeros((count, 1), device=device)
        self.max_radii2D = torch.zeros(count, device=device)

    @torch.no_grad()
    def add_densification_stats(
        self,
        viewspace_points: torch.Tensor,
        visible_indices: torch.Tensor,
        radii: torch.Tensor,
    ) -> None:
        if not hasattr(self, "xyz_gradient_accum"):
            self.reset_densification_stats()
        indices = visible_indices.reshape(-1).long()
        if indices.numel() == 0 or viewspace_points.grad is None:
            return
        gradient = torch.linalg.vector_norm(viewspace_points.grad[indices, :2], dim=-1)
        self.xyz_gradient_accum[indices, 0] += gradient
        self.denom[indices, 0] += 1
        self.max_radii2D[indices] = torch.maximum(
            self.max_radii2D[indices], radii[indices]
        )

    @torch.no_grad()
    def _append_to_optimizer(
        self,
        optimizer: torch.optim.Adam,
        extensions: dict[str, torch.Tensor],
    ) -> None:
        attributes = self._parameter_attributes()
        for group in optimizer.param_groups:
            name = group["name"]
            old_parameter = group["params"][0]
            extension = extensions[name].detach()
            new_parameter = nn.Parameter(
                torch.cat((old_parameter.detach(), extension), dim=0).requires_grad_(True)
            )
            state = optimizer.state.pop(old_parameter, None)
            if state is not None:
                for key, value in tuple(state.items()):
                    if torch.is_tensor(value) and tuple(value.shape) == tuple(old_parameter.shape):
                        state[key] = torch.cat((value, torch.zeros_like(extension)), dim=0)
                optimizer.state[new_parameter] = state
            group["params"][0] = new_parameter
            setattr(self, attributes[name], new_parameter)

    @torch.no_grad()
    def _keep_in_optimizer(
        self,
        optimizer: torch.optim.Adam,
        keep: torch.Tensor,
    ) -> None:
        attributes = self._parameter_attributes()
        for group in optimizer.param_groups:
            name = group["name"]
            old_parameter = group["params"][0]
            new_parameter = nn.Parameter(old_parameter.detach()[keep].requires_grad_(True))
            state = optimizer.state.pop(old_parameter, None)
            if state is not None:
                for key, value in tuple(state.items()):
                    if torch.is_tensor(value) and tuple(value.shape) == tuple(old_parameter.shape):
                        state[key] = value[keep]
                optimizer.state[new_parameter] = state
            group["params"][0] = new_parameter
            setattr(self, attributes[name], new_parameter)

    @torch.no_grad()
    def densify_and_prune(
        self,
        optimizer: torch.optim.Adam,
        *,
        grad_threshold: float,
        min_opacity: float,
        scene_extent: float,
        percent_dense: float,
        max_points: int,
        max_screen_size: float | None = None,
        max_world_scale_fraction: float = 0.1,
    ) -> dict[str, int]:
        """Apply the standard 3DGS clone/split rule and full size/opacity prune."""

        if not hasattr(self, "xyz_gradient_accum"):
            raise RuntimeError("densification statistics were not initialized")
        before = int(self.get_xyz.shape[0])
        gradients = (self.xyz_gradient_accum / self.denom.clamp_min(1)).squeeze(-1)
        # Match GaussianModel.densify_and_prune: densification is selected by
        # screen-space gradient and scale only.  Opacity is applied afterwards
        # as a prune rule.  Gating candidates by opacity here permanently lost
        # high-error points before their opacity had time to grow.
        eligible = gradients >= float(grad_threshold)
        capacity = max(0, int(max_points) - before)
        eligible_count = int(eligible.sum().item())
        if eligible_count > capacity:
            selected_indices = torch.where(eligible)[0]
            selected_gradients = gradients[selected_indices]
            keep_indices = selected_indices[torch.topk(selected_gradients, capacity).indices]
            eligible.zero_()
            eligible[keep_indices] = True

        large = eligible & (
            self.get_scaling.max(dim=1).values > float(percent_dense) * float(scene_extent)
        )
        small = eligible & ~large
        clone_count = int(small.sum().item())
        split_parent_count = int(large.sum().item())

        old_max_radii = self.max_radii2D.detach().clone()
        appended_max_radii: list[torch.Tensor] = []
        pieces: dict[str, list[torch.Tensor]] = {
            name: [] for name in self._parameter_attributes()
        }
        if clone_count:
            for name, attribute in self._parameter_attributes().items():
                pieces[name].append(getattr(self, attribute).detach()[small])
            # New points have not yet established a screen-space footprint.
            appended_max_radii.append(torch.zeros(clone_count, device=self.get_xyz.device))
        if split_parent_count:
            repeats = 2
            scales = self.get_scaling[large].repeat_interleave(repeats, dim=0)
            offsets = torch.randn_like(scales) * scales
            rotations = self._quaternion_to_matrix(self.get_rotation[large])
            rotations = rotations.repeat_interleave(repeats, dim=0)
            child_xyz = (
                torch.bmm(rotations, offsets.unsqueeze(-1)).squeeze(-1)
                + self.get_xyz[large].repeat_interleave(repeats, dim=0)
            )
            pieces["xyz"].append(child_xyz)
            pieces["f_dc"].append(self._features_dc.detach()[large].repeat_interleave(repeats, 0))
            pieces["f_rest"].append(self._features_rest.detach()[large].repeat_interleave(repeats, 0))
            pieces["opacity"].append(self._opacity.detach()[large].repeat_interleave(repeats, 0))
            pieces["scaling"].append(torch.log(scales / (0.8 * repeats)))
            pieces["rotation"].append(self._rotation.detach()[large].repeat_interleave(repeats, 0))
            appended_max_radii.append(
                torch.zeros(repeats * split_parent_count, device=self.get_xyz.device)
            )

        appended = clone_count + 2 * split_parent_count
        if appended:
            extensions = {
                name: torch.cat(tensors, dim=0) for name, tensors in pieces.items()
            }
            self._append_to_optimizer(optimizer, extensions)
            self.max_radii2D = torch.cat(
                (old_max_radii, torch.cat(appended_max_radii, dim=0)), dim=0
            )

        remove = torch.zeros(self.get_xyz.shape[0], dtype=torch.bool, device=self.get_xyz.device)
        remove[:before] = large
        low_opacity = self.get_opacity.squeeze(-1) < float(min_opacity)
        large_screen = torch.zeros_like(remove)
        large_world = torch.zeros_like(remove)
        if max_screen_size is not None:
            large_screen = self.max_radii2D > float(max_screen_size)
            large_world = self.get_scaling.max(dim=1).values > (
                float(max_world_scale_fraction) * float(scene_extent)
            )
        remove |= low_opacity | large_screen | large_world
        pruned_count = int(remove.sum().item())
        if pruned_count:
            self._keep_in_optimizer(optimizer, ~remove)
        after = int(self.get_xyz.shape[0])
        self.reset_densification_stats()
        return {
            "before": before,
            "eligible": eligible_count,
            "clone_parents": clone_count,
            "split_parents": split_parent_count,
            "created": appended,
            "pruned": pruned_count,
            "after": after,
            "pruned_low_opacity": int(low_opacity.sum().item()),
            "pruned_large_screen": int(large_screen.sum().item()),
            "pruned_large_world": int(large_world.sum().item()),
        }

    @torch.no_grad()
    def prune_only(
        self,
        optimizer: torch.optim.Adam,
        *,
        min_opacity: float,
        scene_extent: float,
        max_screen_size: float | None = None,
        max_world_scale_fraction: float = 0.1,
    ) -> dict[str, int]:
        """Continue pruning after densification has stopped."""

        if not hasattr(self, "max_radii2D"):
            self.reset_densification_stats()
        before = int(self.get_xyz.shape[0])
        low_opacity = self.get_opacity.squeeze(-1) < float(min_opacity)
        large_screen = torch.zeros_like(low_opacity)
        large_world = torch.zeros_like(low_opacity)
        if max_screen_size is not None:
            large_screen = self.max_radii2D > float(max_screen_size)
            large_world = self.get_scaling.max(dim=1).values > (
                float(max_world_scale_fraction) * float(scene_extent)
            )
        remove = low_opacity | large_screen | large_world
        if bool(remove.all()):
            remove[self.get_opacity.squeeze(-1).argmax()] = False
        pruned_count = int(remove.sum().item())
        if pruned_count:
            self._keep_in_optimizer(optimizer, ~remove)
        after = int(self.get_xyz.shape[0])
        self.reset_densification_stats()
        return {
            "before": before,
            "eligible": 0,
            "clone_parents": 0,
            "split_parents": 0,
            "created": 0,
            "pruned": pruned_count,
            "after": after,
            "pruned_low_opacity": int(low_opacity.sum().item()),
            "pruned_large_screen": int(large_screen.sum().item()),
            "pruned_large_world": int(large_world.sum().item()),
        }

    @torch.no_grad()
    def stabilize(
        self,
        *,
        min_scale_factor: float = 0.25,
        max_scale_factor: float = 4.0,
    ) -> None:
        q05 = float(self.metadata["scale_q05"])
        q95 = float(self.metadata["scale_q95"])
        if not 0.0 < float(min_scale_factor) <= float(max_scale_factor):
            raise ValueError("scale factors must satisfy 0 < min <= max")
        self._scaling.clamp_(
            np.log(q05 * float(min_scale_factor)),
            np.log(q95 * float(max_scale_factor)),
        )
        self._opacity.clamp_(-12.0, 12.0)
        self._features_dc.clamp_(-6.0, 6.0)
        self._features_rest.clamp_(-6.0, 6.0)

    @torch.no_grad()
    def pruned(self, opacity_threshold: float) -> "ScalarSHGaussianField":
        keep = self.get_opacity.squeeze(-1) >= float(opacity_threshold)
        if not bool(keep.any()):
            keep[self.get_opacity.squeeze(-1).argmax()] = True
        metadata = dict(self.metadata)
        metadata.update(
            {
                "pruned_from_count": int(keep.numel()),
                "retained_count": int(keep.sum().item()),
                "opacity_threshold": float(opacity_threshold),
            }
        )
        result = ScalarSHGaussianField(
            self._xyz.detach()[keep],
            self._features_dc.detach()[keep],
            self._features_rest.detach()[keep],
            self._opacity.detach()[keep],
            self._scaling.detach()[keep],
            self._rotation.detach()[keep],
            sh_degree=self.max_sh_degree,
            metadata=metadata,
        )
        result.active_sh_degree = self.active_sh_degree
        result.requires_grad_(False)
        return result

    @torch.no_grad()
    def save(self, path: str) -> None:
        payload = {
            "format": FORMAT,
            "version": VERSION,
            "sh_degree": self.max_sh_degree,
            "active_sh_degree": self.active_sh_degree,
            "xyz": self._xyz.detach().cpu(),
            "features_dc": self._features_dc.detach().cpu(),
            "features_rest": self._features_rest.detach().cpu(),
            "opacity_logits": self._opacity.detach().cpu(),
            "log_scaling": self._scaling.detach().cpu(),
            "rotation": self._rotation.detach().cpu(),
            "metadata": self.metadata,
        }
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        torch.save(payload, output)

    @classmethod
    def load(cls, path: str, *, device: str = "cuda", trainable: bool = False):
        payload = torch.load(path, map_location="cpu", weights_only=True)
        if payload.get("format") != FORMAT or payload.get("version") != VERSION:
            raise ValueError("unsupported scalar SH NDF artifact")
        result = cls(
            payload["xyz"].to(device),
            payload["features_dc"].to(device),
            payload["features_rest"].to(device),
            payload["opacity_logits"].to(device),
            payload["log_scaling"].to(device),
            payload["rotation"].to(device),
            sh_degree=int(payload["sh_degree"]),
            metadata=payload.get("metadata", {}),
        )
        result.active_sh_degree = int(
            payload.get("active_sh_degree", payload["sh_degree"])
        )
        result.requires_grad_(trainable)
        return result

    @torch.no_grad()
    def save_rgb_ply(self, path: str) -> None:
        """Write an RGB-compatible inspection PLY by repeating scalar SH."""

        xyz = self._xyz.detach().cpu().numpy()
        normals = np.zeros_like(xyz)
        dc = self._features_dc.detach().repeat(1, 1, 3).transpose(1, 2).flatten(1).cpu().numpy()
        rest = (
            self._features_rest.detach().repeat(1, 1, 3).transpose(1, 2).flatten(1).cpu().numpy()
        )
        opacity = self._opacity.detach().cpu().numpy()
        scaling = self._scaling.detach().cpu().numpy()
        rotation = self._rotation.detach().cpu().numpy()
        names = ["x", "y", "z", "nx", "ny", "nz"]
        names += [f"f_dc_{index}" for index in range(dc.shape[1])]
        names += [f"f_rest_{index}" for index in range(rest.shape[1])]
        names += ["opacity", "scale_0", "scale_1", "scale_2"]
        names += ["rot_0", "rot_1", "rot_2", "rot_3"]
        values = np.concatenate((xyz, normals, dc, rest, opacity, scaling, rotation), axis=1)
        elements = np.empty(values.shape[0], dtype=[(name, "f4") for name in names])
        elements[:] = list(map(tuple, values.astype(np.float32, copy=False)))
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        PlyData([PlyElement.describe(elements, "vertex")]).write(str(output))
