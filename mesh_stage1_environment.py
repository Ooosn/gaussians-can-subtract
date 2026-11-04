"""Small appearance modules for the existing mesh pretrain renderer.

Geometry is never optimized here.  The caller owns the ordinary RGB texture
through :class:`mesh_renderer.MeshRenderer`; this module only adds one of four
low-frequency appearance choices on top of that texture:

* ``texture``: no additional appearance term (the original pretrain baseline),
* ``uv_sh``: low-resolution UV SH color coefficients,
* ``single_env``: one shared low-resolution environment map,
* ``view_env_sh`` / ``view_env_mlp``: a camera-conditioned mixture of a few
  environment maps.

The environment path is intentionally diffuse/ambient only.  It evaluates the
map with the interpolated world normal and performs no specular BRDF, material
decomposition, roughness, or per-pixel material fitting.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import nvdiffrast.torch as dr
import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.sh_utils import C0, C1, C2


VARIANTS = ("texture", "uv_sh", "single_env", "view_env_sh", "view_env_mlp")


def sh_basis_degree2(directions: torch.Tensor) -> torch.Tensor:
    """Return the real degree-0..2 SH basis in the repository's ordering."""

    directions = F.normalize(directions, dim=-1, eps=1e-8)
    x, y, z = directions.unbind(dim=-1)
    xx, yy, zz = x * x, y * y, z * z
    return torch.stack(
        (
            torch.full_like(x, C0),
            -C1 * y,
            C1 * z,
            -C1 * x,
            C2[0] * x * y,
            C2[1] * y * z,
            C2[2] * (2.0 * zz - xx - yy),
            C2[3] * x * z,
            C2[4] * (xx - yy),
        ),
        dim=-1,
    )


@dataclass(frozen=True)
class AppearanceConfig:
    variant: str
    environment_height: int = 16
    environment_width: int = 32
    environment_count: int = 4
    uv_sh_resolution: int = 64
    environment_strength: float = 0.35
    uv_sh_strength: float = 0.15
    gate_hidden: int = 16

    def __post_init__(self) -> None:
        if self.variant not in VARIANTS:
            raise ValueError(f"Unknown Stage-1 variant: {self.variant}")


class Stage1Appearance(nn.Module):
    """View-dependent appearance attached to the existing textured mesh."""

    def __init__(
        self,
        config: AppearanceConfig,
        scene_center: torch.Tensor,
        seed: int = 0,
    ) -> None:
        super().__init__()
        self.config = config
        self.register_buffer("scene_center", scene_center.detach().float().reshape(3))

        self.environment_maps: nn.Parameter | None = None
        self.uv_sh_coefficients: nn.Parameter | None = None
        self.sh_gate: nn.Linear | None = None
        self.mlp_gate: nn.Sequential | None = None

        if config.variant == "uv_sh":
            # The ordinary RGB texture already represents the DC component.
            # Store only degree-1 and degree-2 coefficients (8 bases x RGB).
            self.uv_sh_coefficients = nn.Parameter(
                torch.zeros(
                    1,
                    config.uv_sh_resolution,
                    config.uv_sh_resolution,
                    3 * 8,
                )
            )
        elif config.variant == "single_env":
            self.environment_maps = nn.Parameter(
                torch.zeros(1, 3, config.environment_height, config.environment_width)
            )
        elif config.variant in {"view_env_sh", "view_env_mlp"}:
            self.environment_maps = nn.Parameter(
                torch.zeros(
                    config.environment_count,
                    3,
                    config.environment_height,
                    config.environment_width,
                )
            )
            # A deterministic non-identical gate makes the zero environment
            # bases diverge after the first update while keeping RGB exactly at
            # the original texture at initialization.
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(seed)
                if config.variant == "view_env_sh":
                    self.sh_gate = nn.Linear(9, config.environment_count, bias=True)
                    nn.init.normal_(self.sh_gate.weight, mean=0.0, std=0.12)
                    nn.init.zeros_(self.sh_gate.bias)
                else:
                    self.mlp_gate = nn.Sequential(
                        nn.Linear(3, config.gate_hidden),
                        nn.SiLU(),
                        nn.Linear(config.gate_hidden, config.environment_count),
                    )
                    nn.init.xavier_uniform_(self.mlp_gate[0].weight)
                    nn.init.zeros_(self.mlp_gate[0].bias)
                    nn.init.xavier_uniform_(self.mlp_gate[2].weight)
                    nn.init.zeros_(self.mlp_gate[2].bias)

    @property
    def variant(self) -> str:
        return self.config.variant

    def optimizer_groups(self) -> list[dict]:
        """Return the small appearance-only Adam groups."""

        groups: list[dict] = []
        if self.uv_sh_coefficients is not None:
            groups.append(
                {"params": [self.uv_sh_coefficients], "lr": 5e-3, "name": "uv_sh"}
            )
        if self.environment_maps is not None:
            groups.append(
                {"params": [self.environment_maps], "lr": 5e-3, "name": "environment"}
            )
        if self.sh_gate is not None:
            groups.append({"params": self.sh_gate.parameters(), "lr": 1e-3, "name": "sh_gate"})
        if self.mlp_gate is not None:
            groups.append({"params": self.mlp_gate.parameters(), "lr": 1e-3, "name": "mlp_gate"})
        return groups

    def gate_parameters(self) -> list[nn.Parameter]:
        """Parameters allowed during shared test-view environment adaptation."""

        if self.sh_gate is not None:
            return list(self.sh_gate.parameters())
        if self.mlp_gate is not None:
            return list(self.mlp_gate.parameters())
        return []

    def _camera_direction(self, camera) -> torch.Tensor:
        center = camera.camera_center.to(self.scene_center.device).reshape(3)
        return F.normalize(center - self.scene_center, dim=0, eps=1e-8)

    def _environment_weights(self, camera) -> torch.Tensor:
        assert self.environment_maps is not None
        if self.variant == "single_env":
            return torch.ones(1, device=self.environment_maps.device)
        direction = self._camera_direction(camera)
        if self.sh_gate is not None:
            logits = self.sh_gate(sh_basis_degree2(direction))
        elif self.mlp_gate is not None:
            logits = self.mlp_gate(direction)
        else:
            raise RuntimeError("View-conditioned environment has no gate")
        return torch.softmax(logits, dim=-1)

    def _sample_environment(self, normals: torch.Tensor, camera) -> torch.Tensor:
        """Sample the learned lat-long maps using world-space surface normals."""

        assert self.environment_maps is not None
        normals = F.normalize(normals, dim=-1, eps=1e-8)
        x, y, z = normals.unbind(dim=-1)
        u = torch.remainder(torch.atan2(x, z) / (2.0 * math.pi) + 0.5, 1.0)
        v = torch.acos(y.clamp(-1.0, 1.0)) / math.pi
        grid = torch.stack((u * 2.0 - 1.0, v * 2.0 - 1.0), dim=-1)
        count = self.environment_maps.shape[0]
        sampled = F.grid_sample(
            self.environment_maps,
            grid.expand(count, -1, -1, -1),
            mode="bilinear",
            padding_mode="border",
            align_corners=False,
        )
        weights = self._environment_weights(camera).reshape(count, 1, 1, 1)
        return (sampled * weights).sum(dim=0)

    def _uv_sh_delta(self, renderer, camera, rast: torch.Tensor) -> torch.Tensor:
        assert self.uv_sh_coefficients is not None
        uv_map, _ = dr.interpolate(renderer.uv, rast, renderer.faces_contiguous)
        coefficients = dr.texture(
            self.uv_sh_coefficients.contiguous(), uv_map, filter_mode="linear"
        )
        world_position, _ = dr.interpolate(
            renderer.verts.contiguous(), rast, renderer.faces_contiguous
        )
        camera_center = camera.camera_center.to(world_position.device).reshape(1, 1, 1, 3)
        view_direction = F.normalize(camera_center - world_position, dim=-1, eps=1e-8)
        basis = sh_basis_degree2(view_direction)[..., 1:]
        coefficients = coefficients.reshape(*coefficients.shape[:-1], 3, 8)
        return (coefficients * basis.unsqueeze(-2)).sum(dim=-1).squeeze(0).permute(2, 0, 1)

    def forward(
        self,
        renderer,
        camera,
        rast: torch.Tensor,
        mask: torch.Tensor,
        base_rgb: torch.Tensor,
    ) -> torch.Tensor:
        """Apply the selected appearance model to the existing texture render."""

        if self.variant == "texture":
            return base_rgb

        mask_chw = mask.float()
        if mask_chw.ndim == 3:
            mask_chw = mask_chw.squeeze(0)
        mask_chw = mask_chw.unsqueeze(0)

        if self.variant == "uv_sh":
            delta = self._uv_sh_delta(renderer, camera, rast)
            correction = self.config.uv_sh_strength * torch.tanh(delta)
            return (base_rgb + correction * mask_chw).clamp(0.0, 1.0)

        normals, _ = dr.interpolate(
            renderer.normals.contiguous(), rast, renderer.faces_contiguous
        )
        environment = self._sample_environment(normals, camera)
        illumination = 1.0 + self.config.environment_strength * torch.tanh(environment)
        illumination = illumination * mask_chw + (1.0 - mask_chw)
        return (base_rgb * illumination).clamp(0.0, 1.0)

    def save(self, path: str, extra: dict | None = None) -> None:
        payload = {
            "format": "mesh_stage1_environment_v1",
            "config": asdict(self.config),
            "scene_center": self.scene_center.detach().cpu(),
            "state_dict": {key: value.detach().cpu() for key, value in self.state_dict().items()},
            "extra": extra or {},
        }
        torch.save(payload, path)

    @classmethod
    def load(cls, path: str, device: str = "cuda") -> "Stage1Appearance":
        payload = torch.load(path, map_location="cpu", weights_only=False)
        if payload.get("format") != "mesh_stage1_environment_v1":
            raise ValueError("Unsupported Stage-1 appearance artifact")
        model = cls(
            AppearanceConfig(**payload["config"]),
            payload["scene_center"],
            seed=0,
        )
        model.load_state_dict(payload["state_dict"], strict=True)
        return model.to(device)
