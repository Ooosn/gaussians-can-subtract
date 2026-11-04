"""Scalar NDF field backed by the repository's native ``GaussianModel``.

This deliberately reuses the original 3DGS optimizer, learning-rate schedule,
clone/split implementation, opacity reset, and optimizer-state surgery.  Only
the feature channel count is one instead of RGB.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
from plyfile import PlyData
import torch
from torch import nn

from scene.gaussian_model import GaussianModel
from simple_knn._C import distCUDA2
from utils.general_utils import inverse_sigmoid
from utils.sh_utils import C0


FORMAT = "error_guided_scalar_sh_gaussians"
VERSION = 1


class NativeScalarSHGaussianField(GaussianModel):
    """One-channel SH field using the unmodified native 3DGS machinery."""

    def __init__(self, sh_degree: int, metadata: dict[str, Any] | None = None):
        super().__init__(int(sh_degree), optimizer_type="default")
        self.metadata = dict(metadata or {})
        self.train_test_exp = False
        self.pretrained_exposures = None
        self._last_radii = torch.empty(0, device="cuda")

    @classmethod
    @torch.no_grad()
    def load(cls, path: str | Path, device: str = "cuda") -> "NativeScalarSHGaussianField":
        """Load the exact scalar-SH field saved by :meth:`save`."""

        payload = torch.load(Path(path), map_location="cpu", weights_only=False)
        if payload.get("format") != FORMAT or int(payload.get("version", -1)) != VERSION:
            raise ValueError(f"Unsupported scalar NDF artifact: {path}")
        field = cls(int(payload["sh_degree"]), payload.get("metadata", {}))

        def parameter(name: str) -> nn.Parameter:
            value = payload[name]
            if not torch.is_tensor(value) or not torch.isfinite(value).all():
                raise ValueError(f"Invalid tensor {name!r} in {path}")
            return nn.Parameter(value.to(device=device, dtype=torch.float32).contiguous())

        field._xyz = parameter("xyz")
        field._features_dc = parameter("features_dc")
        field._features_rest = parameter("features_rest")
        field._opacity = parameter("opacity_logits")
        field._scaling = parameter("log_scaling")
        field._rotation = parameter("rotation")
        count = int(field._xyz.shape[0])
        expected_rest = (field.max_sh_degree + 1) ** 2 - 1
        if field._features_dc.shape != (count, 1, 1):
            raise ValueError(f"Expected scalar DC shape {(count, 1, 1)}, got {tuple(field._features_dc.shape)}")
        if field._features_rest.shape != (count, expected_rest, 1):
            raise ValueError(
                f"Expected scalar rest shape {(count, expected_rest, 1)}, "
                f"got {tuple(field._features_rest.shape)}"
            )
        for name, tensor, width in (
            ("xyz", field._xyz, 3),
            ("opacity_logits", field._opacity, 1),
            ("log_scaling", field._scaling, 3),
            ("rotation", field._rotation, 4),
        ):
            if tensor.shape != (count, width):
                raise ValueError(f"Expected {name} shape {(count, width)}, got {tuple(tensor.shape)}")
        field.active_sh_degree = min(
            int(payload.get("active_sh_degree", 0)), field.max_sh_degree
        )
        field.max_radii2D = torch.zeros(count, device=device)
        field.xyz_gradient_accum = torch.zeros((count, 1), device=device)
        field.denom = torch.zeros((count, 1), device=device)
        field.visibility = torch.zeros(count, device=device)
        return field

    @classmethod
    @torch.no_grad()
    def from_gaussolve_vertices(
        cls,
        vertices: torch.Tensor,
        *,
        sh_degree: int = 3,
        initial_opacity: float = 0.01,
        initial_value: float = 0.05,
        metadata: dict[str, Any] | None = None,
    ) -> "NativeScalarSHGaussianField":
        """Initialize through the repository's existing NDF initializer."""

        if abs(float(initial_opacity) - 0.01) > 1e-12:
            raise ValueError("gaussolve_init fixes initial opacity at 0.01")
        if abs(float(initial_value) - 0.05) > 1e-12:
            raise ValueError("gaussolve_init fixes initial scalar SH value at 0.05")
        if vertices.device.type != "cuda":
            raise ValueError("vertices must be on CUDA")
        field = cls(sh_degree, metadata)
        # gaussolve_init needs only the already-aligned mesh vertices.  Passing
        # the exact unique-XYZ set avoids duplicating UV-seam vertex rows.
        field.gaussolve_init(SimpleNamespace(verts=vertices.detach().contiguous()))
        scale = field.get_scaling.max(dim=1).values
        field.metadata.update(
            {
                "initial_count": int(field.get_xyz.shape[0]),
                "scale_q05": float(torch.quantile(scale, 0.05).item()),
                "scale_q95": float(torch.quantile(scale, 0.95).item()),
                "initial_opacity": 0.01,
                "initial_value": 0.05,
                "backend": "existing_gaussolve_ndf",
                "geometry_source": "exact_unique_stage1_mesh_vertices",
            }
        )
        field.active_sh_degree = 0
        return field

    @classmethod
    @torch.no_grad()
    def from_ply_geometry(
        cls,
        path: str | Path,
        *,
        sh_degree: int = 3,
        initial_opacity: float = 0.01,
        initial_value: float = 0.05,
        metadata: dict[str, Any] | None = None,
    ) -> "NativeScalarSHGaussianField":
        """Keep PLY XYZ/scale/rotation exactly and reinitialize field values."""

        path = Path(path)
        vertex = PlyData.read(str(path))["vertex"].data
        names = vertex.dtype.names or ()
        required = ("x", "y", "z", "scale_0", "scale_1", "scale_2")
        missing = [name for name in required if name not in names]
        rotation_names = sorted(
            (name for name in names if name.startswith("rot")),
            key=lambda name: int(name.split("_")[-1]),
        )
        if missing or len(rotation_names) != 4:
            raise ValueError(
                f"PLY geometry is incomplete; missing={missing}, rotations={rotation_names}"
            )
        xyz = np.stack((vertex["x"], vertex["y"], vertex["z"]), axis=1)
        log_scaling = np.stack(
            (vertex["scale_0"], vertex["scale_1"], vertex["scale_2"]), axis=1
        )
        rotation = np.stack([vertex[name] for name in rotation_names], axis=1)
        if not (
            np.isfinite(xyz).all()
            and np.isfinite(log_scaling).all()
            and np.isfinite(rotation).all()
        ):
            raise ValueError("PLY geometry contains non-finite values")

        points = torch.from_numpy(xyz.astype(np.float32, copy=False)).cuda()
        scaling = torch.from_numpy(log_scaling.astype(np.float32, copy=False)).cuda()
        rotations = torch.from_numpy(rotation.astype(np.float32, copy=False)).cuda()
        field = cls(sh_degree, metadata)
        count = int(points.shape[0])
        actual_scale = torch.exp(scaling).max(dim=1).values
        field.metadata.update(
            {
                "initial_count": count,
                "scale_q05": float(torch.quantile(actual_scale, 0.05).item()),
                "scale_q95": float(torch.quantile(actual_scale, 0.95).item()),
                "initial_opacity": float(initial_opacity),
                "initial_value": float(initial_value),
                "backend": "native_gaussian_model",
                "geometry_source": str(path.resolve()),
                "geometry_preserved_exactly": True,
            }
        )
        dc = torch.full(
            (count, 1, 1),
            (float(initial_value) - 0.5) / C0,
            device=points.device,
        )
        rest = torch.zeros(
            (count, (int(sh_degree) + 1) ** 2 - 1, 1), device=points.device
        )
        field._xyz = nn.Parameter(points.contiguous().requires_grad_(True))
        field._features_dc = nn.Parameter(dc.requires_grad_(True))
        field._features_rest = nn.Parameter(rest.requires_grad_(True))
        field._opacity = nn.Parameter(
            inverse_sigmoid(
                torch.full((count, 1), float(initial_opacity), device=points.device)
            ).requires_grad_(True)
        )
        field._scaling = nn.Parameter(scaling.contiguous().requires_grad_(True))
        field._rotation = nn.Parameter(rotations.contiguous().requires_grad_(True))
        field.active_sh_degree = 0
        field.max_radii2D = torch.zeros(count, device=points.device)
        field.xyz_gradient_accum = torch.zeros((count, 1), device=points.device)
        field.denom = torch.zeros((count, 1), device=points.device)
        field.visibility = torch.zeros(count, device=points.device)
        return field

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
    ) -> "NativeScalarSHGaussianField":
        if points.device.type != "cuda":
            raise ValueError("points must be on CUDA")
        points = points.detach().float().contiguous()
        if points.shape[0] < 4:
            raise ValueError("at least four points are required")

        field = cls(sh_degree, metadata)
        count = int(points.shape[0])
        distance_squared = distCUDA2(points).clamp_min(1e-8)
        scale = torch.sqrt(distance_squared)
        field.metadata.update(
            {
                "initial_count": count,
                "scale_q05": float(torch.quantile(scale, 0.05).item()),
                "scale_q95": float(torch.quantile(scale, 0.95).item()),
                "initial_opacity": float(initial_opacity),
                "initial_value": float(initial_value),
                "backend": "native_gaussian_model",
            }
        )

        dc = torch.full(
            (count, 1, 1),
            (float(initial_value) - 0.5) / C0,
            device=points.device,
        )
        rest = torch.zeros(
            (count, (int(sh_degree) + 1) ** 2 - 1, 1), device=points.device
        )
        rotation = torch.zeros((count, 4), device=points.device)
        rotation[:, 0] = 1.0

        field._xyz = nn.Parameter(points.requires_grad_(True))
        field._features_dc = nn.Parameter(dc.requires_grad_(True))
        field._features_rest = nn.Parameter(rest.requires_grad_(True))
        field._opacity = nn.Parameter(
            inverse_sigmoid(
                torch.full((count, 1), float(initial_opacity), device=points.device)
            ).requires_grad_(True)
        )
        field._scaling = nn.Parameter(
            torch.log(scale).unsqueeze(-1).repeat(1, 3).requires_grad_(True)
        )
        field._rotation = nn.Parameter(rotation.requires_grad_(True))
        field.active_sh_degree = 0
        field.max_radii2D = torch.zeros(count, device=points.device)
        field.xyz_gradient_accum = torch.zeros((count, 1), device=points.device)
        field.denom = torch.zeros((count, 1), device=points.device)
        field.visibility = torch.zeros(count, device=points.device)
        return field

    def make_optimizer(
        self,
        scene_extent: float,
        *,
        total_iterations: int,
        position_lr_init: float,
        position_lr_final: float,
        position_lr_delay_mult: float,
        feature_rest_lr_scale: float,
        percent_dense: float,
    ) -> torch.optim.Adam:
        self.spatial_lr_scale = float(scene_extent)
        options = SimpleNamespace(
            percent_dense=float(percent_dense),
            position_lr_init=float(position_lr_init),
            position_lr_final=float(position_lr_final),
            position_lr_delay_mult=float(position_lr_delay_mult),
            position_lr_max_steps=max(1, int(total_iterations)),
            feature_lr=2.5e-3,
            opacity_lr=2.5e-2,
            scaling_lr=5e-3,
            rotation_lr=1e-3,
            exposure_lr_init=0.0,
            exposure_lr_final=0.0,
            exposure_lr_delay_steps=0,
            exposure_lr_delay_mult=1.0,
            iterations=max(1, int(total_iterations)),
        )
        self.training_setup(options)
        self.optimizer.param_groups[2]["lr"] = (
            2.5e-3 * float(feature_rest_lr_scale)
        )
        return self.optimizer

    @torch.no_grad()
    def reset_densification_stats(self) -> None:
        count = int(self.get_xyz.shape[0])
        self.xyz_gradient_accum = torch.zeros((count, 1), device="cuda")
        self.denom = torch.zeros((count, 1), device="cuda")
        self.max_radii2D = torch.zeros(count, device="cuda")

    @torch.no_grad()
    def add_densification_stats(
        self,
        viewspace_points: torch.Tensor,
        visible_indices: torch.Tensor,
        radii: torch.Tensor | None = None,
    ) -> None:
        # The hybrid trainer uses the native two-argument boolean-mask API.
        # Stage-2 field fitting additionally supplies radii and explicit indices.
        if radii is None:
            GaussianModel.add_densification_stats(
                self, viewspace_points, visible_indices
            )
            return
        indices = visible_indices.reshape(-1).long()
        self._last_radii = radii.detach()
        if indices.numel() == 0 or viewspace_points.grad is None:
            return
        self.max_radii2D[indices] = torch.maximum(
            self.max_radii2D[indices], radii[indices]
        )
        GaussianModel.add_densification_stats(self, viewspace_points, indices)

    @torch.no_grad()
    def densify_and_prune(
        self, *args: Any, **kwargs: Any
    ) -> dict[str, int]:
        # Preserve the repository's ordinary GaussianModel ABI when this field
        # is handed to the hybrid trainer.
        if args and not isinstance(args[0], torch.optim.Optimizer):
            return GaussianModel.densify_and_prune(self, *args, **kwargs)

        optimizer = args[0] if args else kwargs.pop("optimizer")
        grad_threshold = kwargs.pop("grad_threshold")
        min_opacity = kwargs.pop("min_opacity")
        scene_extent = kwargs.pop("scene_extent")
        percent_dense = kwargs.pop("percent_dense")
        max_points = kwargs.pop("max_points")
        max_screen_size = kwargs.pop("max_screen_size", None)
        max_world_scale_fraction = kwargs.pop("max_world_scale_fraction", 0.1)
        if kwargs:
            raise TypeError(f"Unexpected densify_and_prune arguments: {sorted(kwargs)}")
        del max_world_scale_fraction  # native code fixes this at 0.1 * extent
        self.optimizer = optimizer
        self.percent_dense = float(percent_dense)
        before = int(self.get_xyz.shape[0])
        threshold = float(grad_threshold)
        if before >= int(max_points):
            threshold = float("inf")
        radii = self._last_radii
        if radii.numel() != before:
            radii = torch.zeros(before, device=self.get_xyz.device)
        row = GaussianModel.densify_and_prune(
            self,
            threshold,
            float(min_opacity),
            float(scene_extent),
            max_screen_size,
            radii,
        )
        return {
            "before": int(row["n_before"]),
            "eligible": int(row["clone_parents"] + row["split_parents"]),
            "clone_parents": int(row["clone_parents"]),
            "split_parents": int(row["split_parents"]),
            "created": int(row["clone_created"] + row["split_created"]),
            "pruned": int(row["opacity_or_size_pruned"]),
            "after": int(row["n_after"]),
            "pruned_low_opacity": -1,
            "pruned_large_screen": -1,
            "pruned_large_world": -1,
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
        self.optimizer = optimizer
        before = int(self.get_xyz.shape[0])
        low = self.get_opacity.squeeze(-1) < float(min_opacity)
        screen = torch.zeros_like(low)
        world = torch.zeros_like(low)
        if max_screen_size is not None:
            screen = self.max_radii2D > float(max_screen_size)
            world = self.get_scaling.max(dim=1).values > (
                float(max_world_scale_fraction) * float(scene_extent)
            )
        remove = low | screen | world
        if bool(remove.all()):
            remove[self.get_opacity.squeeze(-1).argmax()] = False
        pruned = int(remove.sum().item())
        if pruned:
            GaussianModel.prune_points(self, remove)
        self.reset_densification_stats()
        return {
            "before": before,
            "eligible": 0,
            "clone_parents": 0,
            "split_parents": 0,
            "created": 0,
            "pruned": pruned,
            "after": int(self.get_xyz.shape[0]),
            "pruned_low_opacity": int(low.sum().item()),
            "pruned_large_screen": int(screen.sum().item()),
            "pruned_large_world": int(world.sum().item()),
        }

    @torch.no_grad()
    def reset_opacity(
        self, optimizer: torch.optim.Adam, max_opacity: float = 0.01
    ) -> int:
        self.optimizer = optimizer
        count = int((self.get_opacity > float(max_opacity)).sum().item())
        replacement = self.inverse_opacity_activation(
            torch.minimum(
                self.get_opacity,
                torch.full_like(self.get_opacity, float(max_opacity)),
            )
        )
        tensors = self.replace_tensor_to_optimizer(replacement, "opacity")
        self._opacity = tensors["opacity"]
        return count

    @torch.no_grad()
    def pruned(self, opacity_threshold: float) -> "NativeScalarSHGaussianField":
        keep = self.get_opacity.squeeze(-1) >= float(opacity_threshold)
        if not bool(keep.any()):
            keep[self.get_opacity.squeeze(-1).argmax()] = True
        result = NativeScalarSHGaussianField(self.max_sh_degree, self.metadata)
        result.metadata.update(
            {
                "pruned_from_count": int(keep.numel()),
                "pruned_to_count": int(keep.sum().item()),
                "opacity_threshold": float(opacity_threshold),
            }
        )
        result._xyz = nn.Parameter(self._xyz.detach()[keep].clone())
        result._features_dc = nn.Parameter(self._features_dc.detach()[keep].clone())
        result._features_rest = nn.Parameter(self._features_rest.detach()[keep].clone())
        result._opacity = nn.Parameter(self._opacity.detach()[keep].clone())
        result._scaling = nn.Parameter(self._scaling.detach()[keep].clone())
        result._rotation = nn.Parameter(self._rotation.detach()[keep].clone())
        result.active_sh_degree = self.active_sh_degree
        count = int(keep.sum().item())
        result.max_radii2D = torch.zeros(count, device=self.get_xyz.device)
        result.xyz_gradient_accum = torch.zeros((count, 1), device=self.get_xyz.device)
        result.denom = torch.zeros((count, 1), device=self.get_xyz.device)
        result.visibility = torch.zeros(count, device=self.get_xyz.device)
        return result

    @torch.no_grad()
    def reinitialized(
        self, *, initial_opacity: float = 0.01, initial_value: float = 0.05
    ) -> "NativeScalarSHGaussianField":
        """Keep the learned distribution but start the deployable NDF afresh."""

        result = NativeScalarSHGaussianField(self.max_sh_degree, self.metadata)
        count = int(self.get_xyz.shape[0])
        result.metadata.update(
            {
                "purpose": "error_guided_ndf_initialization",
                "learned_distribution_preserved": True,
                "opacity_reinitialized": float(initial_opacity),
                "sh_value_reinitialized": float(initial_value),
            }
        )
        result._xyz = nn.Parameter(self._xyz.detach().clone())
        result._scaling = nn.Parameter(self._scaling.detach().clone())
        result._rotation = nn.Parameter(self._rotation.detach().clone())
        result._features_dc = nn.Parameter(
            torch.full(
                (count, 1, 1),
                (float(initial_value) - 0.5) / C0,
                device=self.get_xyz.device,
            )
        )
        result._features_rest = nn.Parameter(
            torch.zeros(
                (count, (self.max_sh_degree + 1) ** 2 - 1, 1),
                device=self.get_xyz.device,
            )
        )
        result._opacity = nn.Parameter(
            inverse_sigmoid(
                torch.full(
                    (count, 1), float(initial_opacity), device=self.get_xyz.device
                )
            )
        )
        result.active_sh_degree = 0
        result.max_radii2D = torch.zeros(count, device=self.get_xyz.device)
        result.xyz_gradient_accum = torch.zeros((count, 1), device=self.get_xyz.device)
        result.denom = torch.zeros((count, 1), device=self.get_xyz.device)
        result.visibility = torch.zeros(count, device=self.get_xyz.device)
        return result

    def stabilize(self, **_: Any) -> None:
        raise RuntimeError("native backend does not use custom stabilization")

    def save(self, path: str) -> None:
        payload = {
            "format": FORMAT,
            "version": VERSION,
            "sh_degree": self.max_sh_degree,
            "active_sh_degree": self.active_sh_degree,
            "metadata": self.metadata,
            "xyz": self._xyz.detach().cpu(),
            "features_dc": self._features_dc.detach().cpu(),
            "features_rest": self._features_rest.detach().cpu(),
            "opacity_logits": self._opacity.detach().cpu(),
            "log_scaling": self._scaling.detach().cpu(),
            "rotation": self._rotation.detach().cpu(),
        }
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(payload, path)

    def save_rgb_ply(self, path: str) -> None:
        GaussianModel.save_ply(self, path)
