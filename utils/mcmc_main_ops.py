"""Main-Gaussian-only MCMC operations for the isolated hybrid trainer.

This module deliberately has no dependency on ``scene.GaussianModel`` or on
any rasterizer extension.  It implements the relocation equation from 3DGS-
MCMC with ordinary PyTorch tensors and updates an already-created Gaussian
model through its public tensor/optimizer storage.  The isolated trainer is
responsible for deciding *when* these operations run; NDF and mesh fields are
never accepted as arguments here.
"""

from __future__ import annotations

import math
from typing import Any, Dict

import torch


_DEFAULT_DEAD_OPACITY = 0.005
_MAX_RELOCATION_N = 50


def _require_tensor(name: str, value: Any) -> torch.Tensor:
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor, got {type(value).__name__}")
    if value.dtype != torch.float32:
        raise TypeError(f"{name} must use torch.float32, got {value.dtype}")
    if not value.is_floating_point():
        raise TypeError(f"{name} must be floating point")
    return value


def _require_shape(name: str, value: torch.Tensor, shape: tuple[int | None, ...]) -> None:
    if value.ndim != len(shape):
        raise ValueError(f"{name} must have rank {len(shape)}, got shape {tuple(value.shape)}")
    for actual, expected in zip(value.shape, shape):
        if expected is not None and actual != expected:
            raise ValueError(f"{name} must have shape compatible with {shape}, got {tuple(value.shape)}")


def _inverse_sigmoid(value: torch.Tensor) -> torch.Tensor:
    """Match ``utils.general_utils.inverse_sigmoid`` after safe clamping."""

    value = value.clamp(
        min=0.005,
        max=1.0 - torch.finfo(value.dtype).eps,
    )
    return torch.log(value / (1.0 - value))


def _validate_relocation_inputs(
    opacity_old: torch.Tensor,
    scale_old: torch.Tensor,
    counts: torch.Tensor,
    max_n: int,
    validate_finite: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    opacity_old = _require_tensor("opacity_old", opacity_old)
    scale_old = _require_tensor("scale_old", scale_old)
    _require_shape("opacity_old", opacity_old, (None,))
    _require_shape("scale_old", scale_old, (opacity_old.shape[0], 3))
    if not isinstance(counts, torch.Tensor):
        raise TypeError("counts must be a torch.Tensor")
    if counts.ndim != 1 or counts.shape[0] != opacity_old.shape[0]:
        raise ValueError(
            "counts must have shape [P] matching opacity_old; "
            f"got {tuple(counts.shape)} for P={opacity_old.shape[0]}"
        )
    if counts.device != opacity_old.device or scale_old.device != opacity_old.device:
        raise ValueError("opacity_old, scale_old, and counts must share a device")
    if counts.dtype not in (torch.int32, torch.int64):
        raise TypeError(f"counts must be int32 or int64, got {counts.dtype}")
    if not isinstance(max_n, int) or not (1 <= max_n <= _MAX_RELOCATION_N):
        raise ValueError(f"max_n must be an integer in [1, {_MAX_RELOCATION_N}]")
    if validate_finite:
        if not torch.isfinite(opacity_old).all() or not torch.isfinite(scale_old).all():
            raise ValueError("relocation inputs contain non-finite values")
        if opacity_old.numel() and ((opacity_old < 0.0).any() or (opacity_old >= 1.0).any()):
            raise ValueError("opacity_old must lie in [0, 1)")
        if counts.numel() and (counts < 1).any():
            raise ValueError("counts must be at least one")
    # The official wrapper clamps N in-place to [1, N_max-1].  Match that
    # behavior rather than silently allowing a larger polynomial order.
    return opacity_old, scale_old, counts.to(dtype=torch.int64).clamp(max=max_n)


def equation9_relocate(
    opacity_old: torch.Tensor,
    scale_old: torch.Tensor,
    counts: torch.Tensor,
    *,
    max_n: int = _MAX_RELOCATION_N,
    validate_finite: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return relocated opacity and activated scale using MCMC Equation 9.

    ``counts[p]`` is the number of copies represented by Gaussian ``p``.
    The implementation groups points by ``N`` instead of materializing a
    ``[P, N, N]`` tensor.  This keeps peak memory linear in the number of
    points while preserving the official float32 computation.  Full finite
    reductions are opt-in through ``validate_finite``; the training path keeps
    them off to avoid a CPU synchronization over millions of points.
    """

    opacity_old, scale_old, counts = _validate_relocation_inputs(
        opacity_old, scale_old, counts, max_n, validate_finite
    )
    if opacity_old.numel() == 0:
        return opacity_old.clone(), scale_old.clone()

    opacity_new = 1.0 - torch.pow(
        1.0 - opacity_old,
        1.0 / counts.to(dtype=opacity_old.dtype),
    )
    denominator = torch.empty_like(opacity_new)

    # The CUDA reference uses a 51x51 binomial table and loops over i/k.
    # Grouping by N gives identical arithmetic without a huge temporary.
    for n in range(1, max_n + 1):
        selected = counts == n
        if not bool(selected.any()):
            continue
        alpha = opacity_new[selected]
        subtotal = torch.zeros_like(alpha)
        for i in range(1, n + 1):
            k = torch.arange(i, device=alpha.device, dtype=alpha.dtype)
            # Use Python's exact integer binomial coefficients.  This avoids
            # constructing the full reference table for each call.
            coeffs = torch.tensor(
                [float(_binomial(i - 1, int(index))) for index in range(i)],
                device=alpha.device,
                dtype=alpha.dtype,
            )
            signs = torch.where(
                (k.to(dtype=torch.int64) % 2) == 0,
                torch.ones_like(k),
                -torch.ones_like(k),
            )
            terms = (
                coeffs
                * signs
                / torch.sqrt(k + 1.0)
                * alpha.unsqueeze(1).pow(k.unsqueeze(0) + 1.0)
            )
            subtotal = subtotal + terms.sum(dim=1)
        denominator[selected] = subtotal

    if validate_finite:
        if not torch.isfinite(denominator).all():
            raise FloatingPointError("Equation 9 produced a non-finite denominator")
        if (denominator.abs() <= 1e-12).any():
            raise FloatingPointError("Equation 9 denominator is numerically singular")
    scale_factor = opacity_old / denominator
    scale_new = scale_old * scale_factor.unsqueeze(1)
    if validate_finite:
        if not torch.isfinite(opacity_new).all() or not torch.isfinite(scale_new).all():
            raise FloatingPointError("Equation 9 produced non-finite relocation parameters")
        if (scale_new <= 0.0).any():
            raise FloatingPointError("Equation 9 produced non-positive activated scales")
    return opacity_new, scale_new


def _binomial(n: int, k: int) -> int:
    if not (0 <= k <= n):
        return 0
    k = min(k, n - k)
    value = 1
    for index in range(1, k + 1):
        value = value * (n - index + 1) // index
    return value


def _model_tensors(gaussians: Any) -> tuple[torch.Tensor, ...]:
    required = (
        "get_xyz",
        "get_opacity",
        "get_scaling",
        "_xyz",
        "_features_dc",
        "_features_rest",
        "_opacity",
        "_scaling",
        "_rotation",
        "optimizer",
    )
    missing = [name for name in required if not hasattr(gaussians, name)]
    if missing:
        raise AttributeError(f"main Gaussian model is missing required fields: {missing}")
    xyz = _require_tensor("gaussians.get_xyz", gaussians.get_xyz)
    opacity = _require_tensor("gaussians.get_opacity", gaussians.get_opacity)
    scaling = _require_tensor("gaussians.get_scaling", gaussians.get_scaling)
    _require_shape("gaussians.get_xyz", xyz, (None, 3))
    _require_shape("gaussians.get_opacity", opacity, (xyz.shape[0], 1))
    _require_shape("gaussians.get_scaling", scaling, (xyz.shape[0], 3))
    if gaussians.optimizer is None:
        raise RuntimeError("main Gaussian optimizer must exist for MCMC operations")
    for name in ("_xyz", "_features_dc", "_features_rest", "_opacity", "_scaling", "_rotation"):
        value = getattr(gaussians, name)
        if not isinstance(value, torch.Tensor) or value.shape[0] != xyz.shape[0]:
            raise ValueError(f"main raw parameter {name} does not match point count")
    return xyz, opacity, scaling


def _zero_optimizer_moments(gaussians: Any, indices: torch.Tensor) -> None:
    """Reset Adam moments for selected alive MCMC parent rows.

    The current GaussianModel rebases optimizer parameters whenever points are
    appended, so this helper deliberately looks up the *current* parameter
    object each time.  The official MCMC implementation resets only the
    sampled alive parent rows; overwritten dead rows retain their moments so
    they remain exploratory particles.  Missing state is allowed before the
    first optimizer step; existing state must have row-compatible shapes.
    """

    if indices.numel() == 0:
        return
    indices = torch.unique(indices.to(dtype=torch.long))
    for group in gaussians.optimizer.param_groups:
        if len(group.get("params", ())) != 1:
            raise ValueError("MCMC helper requires one tensor parameter per optimizer group")
        parameter = group["params"][0]
        state = gaussians.optimizer.state.get(parameter)
        if state is None:
            continue
        for state_name in ("exp_avg", "exp_avg_sq"):
            moment = state.get(state_name)
            if moment is None:
                continue
            if moment.ndim == 0 or moment.shape[0] != parameter.shape[0]:
                raise ValueError(
                    f"optimizer state {state_name} shape {tuple(moment.shape)} "
                    f"does not match parameter shape {tuple(parameter.shape)}"
                )
            moment[indices] = 0.0


def _optimizer_group_names(gaussians: Any) -> set[str]:
    names = {str(group.get("name", "")) for group in gaussians.optimizer.param_groups}
    required = {"xyz", "f_dc", "f_rest", "opacity", "scaling", "rotation"}
    if not required.issubset(names):
        raise ValueError(f"MCMC helper requires optimizer groups {sorted(required)}, got {sorted(names)}")
    return names


def relocate_and_grow_main(
    gaussians: Any,
    *,
    cap_max: int,
    dead_opacity: float = _DEFAULT_DEAD_OPACITY,
    validate_finite: bool = False,
) -> Dict[str, int]:
    """Relocate dead main Gaussians and grow alive samples up to ``cap_max``.

    This is intentionally main-only.  It follows the official MCMC order:
    relocate rows whose activated opacity is at most 0.005, then append up to
    five percent new rows.  ``densification_postfix`` is used only for the
    append operation so the existing model's optimizer/state bookkeeping is
    preserved without changing the shared model class.  Full finite checks are
    opt-in through ``validate_finite``.
    """

    if not isinstance(cap_max, int) or cap_max <= 0:
        raise ValueError("cap_max must be a positive integer")
    if not (0.0 < float(dead_opacity) < 1.0):
        raise ValueError("dead_opacity must lie in (0, 1)")
    xyz, opacity, scaling = _model_tensors(gaussians)
    _optimizer_group_names(gaussians)
    before = int(xyz.shape[0])

    dead_mask = (opacity[:, 0] <= float(dead_opacity))
    dead_indices = dead_mask.nonzero(as_tuple=True)[0]
    alive_indices = (~dead_mask).nonzero(as_tuple=True)[0]
    relocated = 0

    if dead_indices.numel() and alive_indices.numel():
        probabilities = opacity[alive_indices, 0].clamp_min(0.0)
        probability_sum = probabilities.sum()
        if not torch.isfinite(probability_sum) or probability_sum <= 0.0:
            raise FloatingPointError("cannot sample relocation parents from zero opacity mass")
        sampled_local = torch.multinomial(
            probabilities / probability_sum,
            int(dead_indices.numel()),
            replacement=True,
        )
        source_indices = alive_indices[sampled_local]
        ratio = torch.bincount(source_indices, minlength=before)
        counts = ratio[source_indices].to(dtype=torch.int64) + 1
        new_opacity, new_scaling = equation9_relocate(
            opacity[source_indices, 0],
            scaling[source_indices],
            counts,
            validate_finite=validate_finite,
        )
        raw_opacity = _inverse_sigmoid(new_opacity).unsqueeze(1)
        raw_scaling = torch.log(new_scaling)
        if validate_finite and (
            not torch.isfinite(raw_opacity).all()
            or not torch.isfinite(raw_scaling).all()
        ):
            raise FloatingPointError("relocation produced non-finite raw parameters")
        with torch.no_grad():
            # Dead rows become exact copies of their selected alive parents.
            gaussians._xyz[dead_indices] = gaussians._xyz[source_indices]
            gaussians._features_dc[dead_indices] = gaussians._features_dc[source_indices]
            gaussians._features_rest[dead_indices] = gaussians._features_rest[source_indices]
            gaussians._rotation[dead_indices] = gaussians._rotation[source_indices]
            gaussians._opacity[dead_indices] = raw_opacity
            gaussians._scaling[dead_indices] = raw_scaling
            # The alive parent now represents itself plus the relocated copy.
            gaussians._opacity[source_indices] = raw_opacity
            gaussians._scaling[source_indices] = raw_scaling
        # Match official MCMC: reset only sampled alive parent rows.  Do not
        # clear dead-row moments; those rows are deliberately retained as
        # exploratory particles after relocation.
        _zero_optimizer_moments(gaussians, source_indices)
        relocated = int(dead_indices.numel())

    current = int(gaussians.get_xyz.shape[0])
    target = min(cap_max, int(1.05 * current))
    num_new = max(0, target - current)
    added = 0
    if num_new > 0:
        opacity_now = _require_tensor("gaussians.get_opacity", gaussians.get_opacity)
        scaling_now = _require_tensor("gaussians.get_scaling", gaussians.get_scaling)
        probabilities = opacity_now[:, 0].clamp_min(0.0)
        probability_sum = probabilities.sum()
        if not torch.isfinite(probability_sum) or probability_sum <= 0.0:
            raise FloatingPointError("cannot sample new Gaussian parents from zero opacity mass")
        source_indices = torch.multinomial(
            probabilities / probability_sum,
            num_new,
            replacement=True,
        )
        ratio = torch.bincount(source_indices, minlength=current)
        counts = ratio[source_indices].to(dtype=torch.int64) + 1
        new_opacity, new_scaling = equation9_relocate(
            opacity_now[source_indices, 0],
            scaling_now[source_indices],
            counts,
            validate_finite=validate_finite,
        )
        raw_opacity = _inverse_sigmoid(new_opacity).unsqueeze(1)
        raw_scaling = torch.log(new_scaling)
        if validate_finite and (
            not torch.isfinite(raw_opacity).all()
            or not torch.isfinite(raw_scaling).all()
        ):
            raise FloatingPointError("new Gaussian relocation produced non-finite raw parameters")
        with torch.no_grad():
            gaussians._opacity[source_indices] = raw_opacity
            gaussians._scaling[source_indices] = raw_scaling
            new_xyz = gaussians._xyz[source_indices].detach().clone()
            new_features_dc = gaussians._features_dc[source_indices].detach().clone()
            new_features_rest = gaussians._features_rest[source_indices].detach().clone()
            new_rotation = gaussians._rotation[source_indices].detach().clone()
            new_tmp_radii = torch.zeros(
                (num_new,),
                device=new_xyz.device,
                dtype=gaussians.max_radii2D.dtype,
            )
            # ``load_ply`` checkpoints in the shared model may not have
            # created this optional transient field.  The shared postfix
            # intentionally treats ``None`` as "no current radii buffer";
            # initialize only the missing attribute and leave all visibility
            # and statistics propagation to that postfix.
            if not hasattr(gaussians, "tmp_radii"):
                gaussians.tmp_radii = None
            gaussians.densification_postfix(
                new_xyz,
                new_features_dc,
                new_features_rest,
                raw_opacity.detach().clone(),
                raw_scaling.detach().clone(),
                new_rotation,
                new_tmp_radii,
            )
        _zero_optimizer_moments(gaussians, source_indices)
        added = int(num_new)

    after = int(gaussians.get_xyz.shape[0])
    # Periodic seeding is intentionally independent and may already have
    # crossed the MCMC growth cap.  The caller records that condition as
    # ``seed_over_cap``; MCMC itself must not silently prune those seeds.
    if before <= cap_max and after > cap_max:
        raise AssertionError(f"MCMC add-new exceeded cap: after={after}, cap={cap_max}")
    return {
        "before": before,
        "dead": int(dead_indices.numel()),
        "relocated": relocated,
        "added": added,
        "after": after,
        "cap": int(cap_max),
    }


def main_mcmc_regularizer(
    gaussians: Any,
    *,
    opacity_reg: float,
    scale_reg: float,
) -> torch.Tensor:
    """Return the official MCMC regularizer for the main GS only."""

    if opacity_reg < 0.0 or scale_reg < 0.0:
        raise ValueError("MCMC regularizer weights must be non-negative")
    _, opacity, scaling = _model_tensors(gaussians)
    value = float(opacity_reg) * torch.abs(opacity).mean()
    value = value + float(scale_reg) * torch.abs(scaling).mean()
    if not torch.isfinite(value):
        raise FloatingPointError("MCMC regularizer is non-finite")
    return value


def apply_main_mcmc_noise(
    gaussians: Any,
    *,
    xyz_lr: float,
    noise_lr: float,
    validate_finite: bool = False,
) -> None:
    """Apply the official opacity-gated covariance-shaped SGLD noise."""

    if not math.isfinite(float(xyz_lr)) or xyz_lr < 0.0:
        raise ValueError("xyz_lr must be finite and non-negative")
    if not math.isfinite(float(noise_lr)) or noise_lr < 0.0:
        raise ValueError("noise_lr must be finite and non-negative")
    xyz, opacity, scaling = _model_tensors(gaussians)
    rotation = _require_tensor("gaussians.get_rotation", gaussians.get_rotation)
    _require_shape("gaussians.get_rotation", rotation, (xyz.shape[0], 4))
    if xyz.numel() == 0 or float(xyz_lr) == 0.0 or float(noise_lr) == 0.0:
        return None

    from utils.general_utils import build_scaling_rotation

    with torch.no_grad():
        covariance_factor = build_scaling_rotation(scaling, rotation)
        covariance = covariance_factor @ covariance_factor.transpose(1, 2)
        # Official MCMC uses k=100, x0=.995 on (1-opacity), so low-opacity
        # points receive noise and opaque points are effectively stationary.
        gate_input = 1.0 - opacity
        gate = 1.0 / (1.0 + torch.exp(-100.0 * (gate_input - 0.995)))
        noise = torch.randn_like(xyz) * gate * float(noise_lr) * float(xyz_lr)
        noise = torch.bmm(covariance, noise.unsqueeze(-1)).squeeze(-1)
        xyz.add_(noise)
    if validate_finite and (not torch.isfinite(noise).all() or not torch.isfinite(xyz).all()):
        raise FloatingPointError("MCMC SGLD noise produced non-finite XYZ")
