"""Portable exact-render state bundles and deterministic decomposition helpers.

The training process owns the live objects and calls :func:`save_exact_bundle`
at the final report boundary.  This module deliberately contains no scene,
drive-letter, or checkpoint-path assumptions.  A manifest stores relative
paths whenever possible; loaders still accept the absolute paths emitted by
older exact-report revisions.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Iterable, Mapping, Sequence

import torch


FORMAT = "training_exact_render_contract_v2"
VERSION = 2
TEST_VIEW_FORMAT = "training_report_exact_test_view_v1"
CAMERA_TENSOR_FIELDS = (
    "world_view_transform",
    "full_proj_transform",
    "camera_center",
    "open_gl_view",
    "open_gl_mvp",
)
MESH_STATE_NAMES = (
    "verts",
    "verts_h",
    "faces",
    "faces_contiguous",
    "uv",
    "normals",
    "texture",
    "vertex_colors",
    "alpha",
    "texture_alpha",
    "vertex_alpha",
    "ranges",
    "default_bg",
)
_SAFE_NAME_RE = re.compile(r"[^A-Za-z0-9_.-]+")


def sha256(path: str | os.PathLike[str]) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _safe_name(value: Any) -> str:
    name = _SAFE_NAME_RE.sub("_", str(value)).strip("._")
    if not name:
        raise ValueError(f"invalid empty image name: {value!r}")
    return name


def _cpu_tensor(value: Any, name: str, *, allow_none: bool = False) -> torch.Tensor | None:
    if value is None and allow_none:
        return None
    if not torch.is_tensor(value):
        raise TypeError(f"exact render field {name!r} must be a tensor")
    result = value.detach().to(device="cpu", dtype=torch.float32).contiguous().clone()
    if not bool(torch.isfinite(result).all().item()):
        raise ValueError(f"exact render field {name!r} contains NaN/Inf")
    return result


def _jsonable(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if torch.is_tensor(value):
        return {
            "tensor": True,
            "shape": [int(item) for item in value.shape],
            "dtype": str(value.dtype),
            "device": str(value.device),
        }
    return repr(value)


def _atomic_torch_save(payload: Any, path: Path) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)
    return _artifact_record(path, path.parent)


def _artifact_record(path: Path, relative_to: Path) -> dict[str, Any]:
    path = path.resolve()
    base = relative_to.resolve()
    relative = os.path.relpath(path, base).replace(os.sep, "/")
    return {
        "path": relative,
        "relative_path": relative,
        "sha256": sha256(path),
        "size_bytes": int(path.stat().st_size),
    }


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(_jsonable(value), ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    os.replace(temporary, path)


def _camera_payload(viewpoint: Any) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "image_name": str(viewpoint.image_name),
        "width": int(viewpoint.image_width),
        "height": int(viewpoint.image_height),
        "FoVx": float(viewpoint.FoVx),
        "FoVy": float(viewpoint.FoVy),
        "znear": float(viewpoint.znear),
        "zfar": float(viewpoint.zfar),
    }
    for name in CAMERA_TENSOR_FIELDS:
        if not hasattr(viewpoint, name):
            raise AttributeError(f"camera is missing exact field {name}: {payload['image_name']}")
        payload[name] = _cpu_tensor(getattr(viewpoint, name), f"camera.{name}")
    return payload


def save_test_view(
    exact_report_dir: str | os.PathLike[str],
    iteration: int,
    viewpoint: Any,
    image: torch.Tensor,
    gt_image: torch.Tensor | None,
    psnr_value: float | None = None,
    *,
    overwrite: bool = False,
    extra: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Save one metric-participating test view and return its manifest record."""

    report_dir = Path(exact_report_dir).resolve()
    safe_name = _safe_name(viewpoint.image_name)
    view_path = report_dir / "test_views" / f"{safe_name}.pt"
    if view_path.exists() and not overwrite:
        raise FileExistsError(f"refusing to overwrite exact test view: {view_path}")
    render_cpu = _cpu_tensor(image, "render")
    gt_cpu = _cpu_tensor(gt_image, "gt", allow_none=True)
    camera = _camera_payload(viewpoint)
    payload = {
        "format": TEST_VIEW_FORMAT,
        "version": 1,
        "iteration": int(iteration),
        "split": "test",
        "image_name": camera["image_name"],
        "camera": camera,
        "render": render_cpu,
        "gt": gt_cpu,
        "psnr": None if psnr_value is None else float(psnr_value),
    }
    if extra:
        payload["extra"] = dict(extra)
    _atomic_torch_save(payload, view_path)
    record = _artifact_record(view_path, report_dir)
    record.update(
        {
            "image_name": camera["image_name"],
            "width": camera["width"],
            "height": camera["height"],
            "metric_render_shape": [int(item) for item in render_cpu.shape],
            "metric_gt_shape": None if gt_cpu is None else [int(item) for item in gt_cpu.shape],
            "psnr": None if psnr_value is None else float(psnr_value),
        }
    )
    return record


def _save_main_state(gaussians: Any, path: Path) -> dict[str, Any] | None:
    if gaussians is None:
        return None
    names = ("_xyz", "_features_dc", "_features_rest", "_opacity", "_scaling", "_rotation")
    tensors: dict[str, torch.Tensor | None] = {}
    identities: dict[str, bool] = {}
    requires_grad: dict[str, bool] = {}
    for name in names:
        value = getattr(gaussians, name, None)
        tensors[name] = _cpu_tensor(value, name, allow_none=True)
        identities[name] = isinstance(value, torch.nn.Parameter)
        requires_grad[name] = bool(value.requires_grad) if torch.is_tensor(value) else False
    payload: dict[str, Any] = {
        "format": "gaussian_model_exact_render_state_v1",
        "version": 1,
        "active_sh_degree": int(getattr(gaussians, "active_sh_degree", 0)),
        "max_sh_degree": int(getattr(gaussians, "max_sh_degree", 0)),
        "optimizer_type": str(getattr(gaussians, "optimizer_type", "default")),
        "train_test_exp": bool(getattr(gaussians, "train_test_exp", False)),
        "spatial_lr_scale": float(getattr(gaussians, "spatial_lr_scale", 0.0)),
        "tensors": tensors,
        "parameter_identity": identities,
        "parameter_requires_grad": requires_grad,
    }
    reduced_degrees = getattr(gaussians, "_reduced_sh_degrees", None)
    if torch.is_tensor(reduced_degrees):
        payload["reduced_sh_degrees"] = _cpu_tensor(
            reduced_degrees, "reduced_sh_degrees"
        )
    exposure = getattr(gaussians, "_exposure", None)
    if torch.is_tensor(exposure):
        payload["exposure"] = _cpu_tensor(exposure, "exposure")
        identities["_exposure"] = isinstance(exposure, torch.nn.Parameter)
        requires_grad["_exposure"] = bool(exposure.requires_grad)
        payload["exposure_mapping"] = {
            str(key): int(value) for key, value in getattr(gaussians, "exposure_mapping", {}).items()
        }
    pretrained = getattr(gaussians, "pretrained_exposures", None)
    if pretrained is not None:
        payload["pretrained_exposures"] = {
            str(key): _cpu_tensor(value, f"pretrained_exposure:{key}")
            for key, value in pretrained.items()
        }
    return _atomic_torch_save(payload, path)


def _state_artifact(path: Path, report_dir: Path) -> dict[str, Any]:
    return _artifact_record(path, report_dir)


def _gaussolve_contract(gaussolves: Any) -> dict[str, Any] | None:
    if gaussolves is None:
        return None
    return {
        "type": type(gaussolves).__module__ + "." + type(gaussolves).__qualname__,
        "count": int(gaussolves.get_xyz.shape[0]),
        "max_sh_degree": int(gaussolves.max_sh_degree),
        "active_sh_degree": int(gaussolves.active_sh_degree),
        "features_dc_shape": [int(item) for item in gaussolves._features_dc.shape],
        "features_rest_shape": [int(item) for item in gaussolves._features_rest.shape],
    }


def _render_contract(render_func: Any, render_args: Sequence[Any] | None) -> dict[str, Any] | None:
    if render_func is None or render_args is None:
        return None
    args = tuple(render_args)
    labels = [
        "pipe",
        "background",
        "scaling_modifier",
        "separate_sh",
        "override_color",
        "use_trained_exp",
        "mesh_depth_tolerance",
        "gaussolves",
        "gaussolve_mode",
        "ndf_no_mesh_depth_cutoff",
    ]
    values: dict[str, Any] = {}
    for index, label in enumerate(labels):
        if index >= len(args):
            break
        value = args[index]
        if label == "pipe":
            values[label] = _jsonable(vars(value)) if hasattr(value, "__dict__") else _jsonable(value)
        elif label == "background":
            values[label] = _cpu_tensor(value, label).tolist() if torch.is_tensor(value) else _jsonable(value)
        elif label == "gaussolves":
            values[label] = _gaussolve_contract(value)
        elif label == "override_color":
            values[label] = None if value is None else _jsonable(value)
        else:
            values[label] = _jsonable(value)
    owner = getattr(render_func, "__self__", None)
    return {
        "function": {
            "module": str(getattr(render_func, "__module__", "")),
            "qualname": str(getattr(render_func, "__qualname__", repr(render_func))),
        },
        "render_args_tuple_labels": labels[: len(args)],
        "render_args": values,
        "synchronous_streams": None if owner is None else bool(getattr(owner, "synchronous_streams", False)),
    }


def _normalise_record(record: Mapping[str, Any], report_dir: Path) -> dict[str, Any]:
    result = dict(record)
    path_value = record.get("relative_path") or record.get("path") or record.get("absolute_path")
    if path_value is None:
        raise ValueError(f"test view record has no path: {record}")
    path = Path(str(path_value))
    if not path.is_absolute():
        path = report_dir / path
    result.update(_artifact_record(path, report_dir))
    return result


def save_exact_bundle(
    report_dir: str | os.PathLike[str],
    *,
    iteration: int,
    dataset: Any | None = None,
    opt: Any | None = None,
    pipe: Any | None = None,
    scene: Any | None = None,
    report_render_func: Any | None = None,
    report_render_args: Sequence[Any] | None = None,
    gaussians: Any | None = None,
    mesh_renderer: Any | None = None,
    mesh_appearance: Any | None = None,
    gaussolves: Any | None = None,
    test_view_records: Iterable[Mapping[str, Any]] = (),
    state_dir: str | os.PathLike[str] | None = None,
    extra_manifest: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Save a portable state manifest and optional golden test-view tensors.

    ``report_dir`` owns ``render_contract.json`` and ``test_views``.  State
    tensors are written under ``state_dir`` when supplied, otherwise beside
    the manifest.  Thus a default final asset can always retain its exact
    state even when no ``--exact_report_dir`` was requested.
    """

    report_root = Path(report_dir).resolve()
    state_root = Path(state_dir).resolve() if state_dir is not None else report_root
    report_root.mkdir(parents=True, exist_ok=True)
    state_root.mkdir(parents=True, exist_ok=True)
    records = [] if test_view_records is None else [_normalise_record(record, report_root) for record in test_view_records]
    if scene is not None:
        expected = [str(view.image_name) for view in scene.getTestCameras()]
        actual = [str(record["image_name"]) for record in records]
        if records and actual != expected:
            raise RuntimeError(f"exact report test order mismatch: {actual} != {expected}")

    state_paths: dict[str, dict[str, Any] | None] = {}
    main_path = state_root / "main_render_state.pt"
    state_paths["main_render_state"] = _save_main_state(gaussians, main_path)
    if state_paths["main_render_state"] is not None:
        state_paths["main_render_state"] = _state_artifact(main_path, report_root)

    if mesh_renderer is None:
        state_paths["mesh_exact_state"] = None
    else:
        if not hasattr(mesh_renderer, "exact_render_state"):
            raise TypeError("mesh renderer lacks exact_render_state()")
        mesh_path = state_root / "mesh_exact_state.pt"
        state_paths["mesh_exact_state"] = _atomic_torch_save(mesh_renderer.exact_render_state(), mesh_path)
        state_paths["mesh_exact_state"] = _state_artifact(mesh_path, report_root)

    if mesh_appearance is None:
        state_paths["stage1_appearance"] = None
    else:
        appearance_path = state_root / "stage1_appearance.pt"
        try:
            mesh_appearance.save(str(appearance_path), extra={"iteration": int(iteration), "exact_render_state": True})
        except TypeError:
            mesh_appearance.save(str(appearance_path))
        state_paths["stage1_appearance"] = _state_artifact(appearance_path, report_root)

    if gaussolves is None:
        state_paths["ndf_native_scalar"] = None
    else:
        if not hasattr(gaussolves, "save"):
            raise TypeError("NDF object lacks save() for exact state")
        ndf_path = state_root / "ndf_native_scalar.pt"
        gaussolves.save(str(ndf_path))
        state_paths["ndf_native_scalar"] = _state_artifact(ndf_path, report_root)

    dataset_values = _jsonable(vars(dataset)) if dataset is not None and hasattr(dataset, "__dict__") else _jsonable(dataset)
    opt_values = _jsonable(vars(opt)) if opt is not None and hasattr(opt, "__dict__") else _jsonable(opt)
    pipe_values = _jsonable(vars(pipe)) if pipe is not None and hasattr(pipe, "__dict__") else _jsonable(pipe)
    manifest: dict[str, Any] = {
        "format": FORMAT,
        "version": VERSION,
        "iteration": int(iteration),
        "final_iteration": None if opt is None else int(getattr(opt, "iterations", iteration)),
        "test_order": records,
        "render_contract": _render_contract(report_render_func, report_render_args),
        "dataset_params": dataset_values,
        "optimization_params": opt_values,
        "pipeline_params": pipe_values,
        "state_paths": state_paths,
        "ndf_contract": {
            "native_scalar": gaussolves is not None,
            "loader": "NativeScalarSHGaussianField.load" if gaussolves is not None else None,
            "save_method": "NativeScalarSHGaussianField.save" if gaussolves is not None else None,
            "gaussolve_rgb_ply_used": False,
            **({} if gaussolves is None else _gaussolve_contract(gaussolves)),
        },
        "notes": [
            "State is saved after the final training report and before scene.save/prune/optimizer/final PLY export.",
            "Per-view tensors, when requested, are the clamped/cropped tensors consumed by the training metric.",
            "Manifest paths are relative to the manifest directory; loaders accept legacy absolute paths.",
        ],
    }
    if dataset is not None:
        for attr in (
            "source_path", "model_path", "images", "depths", "resolution", "data_device", "eval",
            "white_background", "train_test_exp", "sh_degree", "gaussolve_sh_degree", "mesh_path",
            "gaussolve_mode", "ndf_no_mesh_depth_cutoff", "mesh_depth_tolerance", "synchronous_render_streams",
        ):
            if hasattr(dataset, attr):
                manifest[attr] = _jsonable(getattr(dataset, attr))
    if opt is not None:
        for attr in ("iterations", "use_sigmoid_rgb", "use_sigmoid_alpha", "mesh_depth_tolerance"):
            if hasattr(opt, attr):
                manifest[attr] = _jsonable(getattr(opt, attr))
    if extra_manifest:
        manifest.update(dict(extra_manifest))
    manifest_path = report_root / "render_contract.json"
    _write_json(manifest_path, manifest)
    manifest["contract_path"] = str(manifest_path)
    manifest["contract_sha256"] = sha256(manifest_path)
    return manifest


def _resolve_artifact(base: Path, value: Any) -> Path:
    if isinstance(value, Mapping):
        value = value.get("relative_path") or value.get("path") or value.get("absolute_path")
    if value is None:
        raise ValueError("artifact record has no path")
    path = Path(str(value))
    return path if path.is_absolute() else (base / path).resolve()


def load_exact_bundle(
    report_dir: str | os.PathLike[str],
    *,
    map_location: str | torch.device = "cpu",
    load_states: bool = False,
    verify_paths: bool = True,
) -> dict[str, Any]:
    """Load a manifest and resolve both new relative and legacy absolute paths."""

    given = Path(report_dir)
    manifest_path = given if given.suffix.lower() == ".json" else given / "render_contract.json"
    manifest_path = manifest_path.resolve()
    if not manifest_path.is_file():
        raise FileNotFoundError(manifest_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8-sig"))
    base = manifest_path.parent
    records = []
    for record in manifest.get("test_order", []):
        item = dict(record)
        item["resolved_path"] = str(_resolve_artifact(base, record))
        if verify_paths and not Path(item["resolved_path"]).is_file():
            raise FileNotFoundError(item["resolved_path"])
        records.append(item)
    state_paths: dict[str, dict[str, Any] | None] = {}
    states: dict[str, Any] = {}
    for name, record in manifest.get("state_paths", {}).items():
        if record is None:
            state_paths[name] = None
            continue
        item = dict(record) if isinstance(record, Mapping) else {"path": record}
        resolved = _resolve_artifact(base, item)
        item["resolved_path"] = str(resolved)
        state_paths[name] = item
        if verify_paths and not resolved.is_file():
            raise FileNotFoundError(resolved)
        if load_states:
            states[name] = torch.load(resolved, map_location=map_location, weights_only=False)
    return {
        "report_dir": str(base),
        "manifest_path": str(manifest_path),
        "manifest": manifest,
        "test_order": records,
        "state_paths": state_paths,
        "states": states,
    }


def _view_payload_checks(payload: Any, source: Path) -> list[str]:
    errors: list[str] = []
    if not isinstance(payload, Mapping):
        return [f"{source}: payload is not a mapping"]
    if payload.get("format") not in (TEST_VIEW_FORMAT, "training_report_exact_test_view_v1"):
        errors.append(f"{source}: unexpected format {payload.get('format')!r}")
    camera = payload.get("camera")
    if not isinstance(camera, Mapping):
        errors.append(f"{source}: missing camera mapping")
    else:
        for key in ("image_name", "width", "height", "FoVx", "FoVy", "znear", "zfar") + CAMERA_TENSOR_FIELDS:
            if key not in camera:
                errors.append(f"{source}: camera missing {key}")
            elif torch.is_tensor(camera[key]) and not bool(torch.isfinite(camera[key]).all().item()):
                errors.append(f"{source}: camera {key} contains NaN/Inf")
    for key in ("render", "gt"):
        value = payload.get(key)
        if value is not None:
            if not torch.is_tensor(value):
                errors.append(f"{source}: {key} is not a tensor")
            elif not bool(torch.isfinite(value).all().item()):
                errors.append(f"{source}: {key} contains NaN/Inf")
    return errors


def verify_exact_bundle(
    bundle_or_report_dir: Mapping[str, Any] | str | os.PathLike[str],
    *,
    load_states: bool = False,
    raise_on_error: bool = False,
) -> dict[str, Any]:
    """Verify manifest paths, cameras, finite tensors, and optional state files."""

    try:
        bundle = (
            dict(bundle_or_report_dir)
            if isinstance(bundle_or_report_dir, Mapping) and "manifest" in bundle_or_report_dir
            else load_exact_bundle(bundle_or_report_dir, load_states=load_states, verify_paths=False)
        )
        manifest = bundle["manifest"]
        base = Path(bundle["report_dir"])
        errors: list[str] = []
        if manifest.get("format") not in (FORMAT, "training_exact_render_contract_v1"):
            errors.append(f"unexpected manifest format: {manifest.get('format')!r}")
        records = bundle.get("test_order", [])
        if not records and manifest.get("test_order"):
            records = [dict(item, resolved_path=str(_resolve_artifact(base, item))) for item in manifest["test_order"]]
        for record in records:
            source = Path(record.get("resolved_path") or _resolve_artifact(base, record))
            if not source.is_file():
                errors.append(f"missing test view: {source}")
                continue
            expected_hash = record.get("sha256") if isinstance(record, Mapping) else None
            if expected_hash and sha256(source) != expected_hash:
                errors.append(f"test view sha256 mismatch: {source}")
            try:
                payload = torch.load(source, map_location="cpu", weights_only=False)
                errors.extend(_view_payload_checks(payload, source))
            except Exception as error:
                errors.append(f"{source}: load failed: {type(error).__name__}: {error}")
        for name, record in bundle.get("state_paths", {}).items():
            if record is None:
                continue
            source = Path(record.get("resolved_path") or _resolve_artifact(base, record))
            if not source.is_file():
                errors.append(f"missing state {name}: {source}")
                continue
            expected_hash = record.get("sha256") if isinstance(record, Mapping) else None
            if expected_hash and sha256(source) != expected_hash:
                errors.append(f"state {name} sha256 mismatch: {source}")
            if load_states:
                try:
                    torch.load(source, map_location="cpu", weights_only=False)
                except Exception as error:
                    errors.append(f"state {name} load failed: {type(error).__name__}: {error}")
        result = {
            "schema": "exact_render_state_verification_v1",
            "status": "PASS" if not errors else "FAIL",
            "errors": errors,
            "view_count": len(records),
            "state_count": sum(record is not None for record in bundle.get("state_paths", {}).values()),
            "manifest_path": bundle.get("manifest_path"),
        }
    except Exception as error:
        result = {
            "schema": "exact_render_state_verification_v1",
            "status": "FAIL",
            "errors": [f"{type(error).__name__}: {error}"],
            "view_count": 0,
            "state_count": 0,
        }
    if raise_on_error and result["status"] != "PASS":
        raise ValueError("exact bundle verification failed: " + "; ".join(result["errors"]))
    return result


def camera_from_saved_payload(
    payload: Mapping[str, Any],
    *,
    device: str | torch.device = "cuda",
    gt: torch.Tensor | None = None,
) -> Any:
    """Construct a lightweight camera directly from one saved view payload."""

    camera_payload = payload.get("camera", payload)
    if not isinstance(camera_payload, Mapping):
        raise TypeError("saved camera payload must be a mapping")
    camera = SimpleNamespace()
    camera.image_name = str(camera_payload["image_name"])
    camera.uid = int(camera_payload.get("uid", 0))
    camera.image_width = int(camera_payload["width"])
    camera.image_height = int(camera_payload["height"])
    camera.FoVx = float(camera_payload["FoVx"])
    camera.FoVy = float(camera_payload["FoVy"])
    camera.znear = float(camera_payload.get("znear", 0.01))
    camera.zfar = float(camera_payload.get("zfar", 100.0))
    target = torch.device(device)
    for name in CAMERA_TENSOR_FIELDS:
        value = camera_payload.get(name)
        if value is None:
            setattr(camera, name, None)
        elif torch.is_tensor(value):
            setattr(camera, name, value.detach().to(device=target, dtype=torch.float32).contiguous())
        else:
            setattr(camera, name, torch.as_tensor(value, dtype=torch.float32, device=target).contiguous())
    saved_gt = gt if gt is not None else payload.get("gt")
    camera.original_image = None if saved_gt is None else saved_gt.detach().to(device=target, dtype=torch.float32).contiguous()
    camera.mask = None
    camera.depth_reliable = False
    return camera


def decompose_components(
    Cf: torch.Tensor,
    Cg: torch.Tensor,
    Tf: torch.Tensor,
    Cm: torch.Tensor,
    D: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Return the fixed scalar-NDF decomposition from one forward pass."""

    tensors = {"Cf": Cf, "Cg": Cg, "Tf": Tf, "Cm": Cm, "D": D}
    for name, value in tensors.items():
        if not torch.is_tensor(value):
            raise TypeError(f"{name} must be a tensor")
        if not bool(torch.isfinite(value).all().item()):
            raise ValueError(f"{name} contains NaN/Inf")
    H = Cf + Tf * Cm
    erase = D * Tf * Cm
    refill = D * (Cg - Cf)
    mesh_contribution = (1.0 - D) * Tf * Cm
    residual_contribution = Cf + refill
    final = H - erase + refill
    return {
        "Cf": Cf,
        "Cg": Cg,
        "Tf": Tf,
        "Cm": Cm,
        "D": D,
        "H": H,
        "Erase": erase,
        "Refill": refill,
        "MeshContribution": mesh_contribution,
        "ResidualContribution": residual_contribution,
        "Final": final,
    }


def scalarize_replicated_channels(value: torch.Tensor, name: str = "features") -> torch.Tensor:
    """Convert scalar or bit-identical RGB channels to one native channel.

    RGB PLY export is only a serialization convenience.  A non-identical RGB
    field cannot be losslessly interpreted as a scalar NDF and is rejected;
    averaging would silently change the render contract.
    """

    if not torch.is_tensor(value) or value.ndim < 1:
        raise TypeError(f"{name} must be a tensor with a channel dimension")
    channels = int(value.shape[-1])
    if channels == 1:
        return value.contiguous()
    if channels != 3:
        raise ValueError(f"{name} must have scalar or replicated RGB channels, got {tuple(value.shape)}")
    if not (torch.equal(value[..., :1], value[..., 1:2]) and torch.equal(value[..., :1], value[..., 2:3])):
        raise ValueError(f"{name} RGB channels are not bit-identical; refusing lossy scalar conversion")
    return value[..., :1].contiguous()


__all__ = [
    "CAMERA_TENSOR_FIELDS",
    "FORMAT",
    "VERSION",
    "camera_from_saved_payload",
    "decompose_components",
    "load_exact_bundle",
    "save_exact_bundle",
    "save_test_view",
    "scalarize_replicated_channels",
    "sha256",
    "verify_exact_bundle",
]
