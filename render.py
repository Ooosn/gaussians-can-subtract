"""Verify or replay portable exact-render bundles without rebuilding a Scene.

Usage examples::

    python render.py verify --report-dir REPORT
    python render.py decompose --report-dir REPORT --output-dir OUT

The replay path constructs a lightweight camera from each saved payload and
calls ``MeshGsRender.render_wrapper`` directly.  The displayed Final image is
always the saved training-report tensor; the wrapper result is retained as a
diagnostic replay tensor.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import sys
from types import SimpleNamespace
from typing import Any, Mapping, Sequence

import torch

from exact_render_state import (
    CAMERA_TENSOR_FIELDS,
    camera_from_saved_payload,
    decompose_components,
    load_exact_bundle,
    sha256,
    verify_exact_bundle,
)


PASS_STATUS = "PASS_FLOAT32_REPLAY"
VIEW_TENSOR_KEYS = (
    "Cf",
    "Cg",
    "Tf",
    "Cm",
    "D",
    "H",
    "Erase",
    "Refill",
    "MeshContribution",
    "ResidualContribution",
    "Final",
)


def _dump(path: Path | None, value: Mapping[str, Any]) -> None:
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n")
    os.replace(temporary, path)


def _parse_views(values: Sequence[str] | None) -> set[str] | None:
    if not values:
        return None
    result: set[str] = set()
    for value in values:
        for item in str(value).split(","):
            item = item.strip()
            if item:
                result.add(item if item.endswith(".png") else item + ".png")
    return result or None


def _restore_main(state: Mapping[str, Any], GaussianModel: Any, device: torch.device) -> Any:
    model = GaussianModel(int(state["max_sh_degree"]), state.get("optimizer_type", "default"))
    identities = state.get("parameter_identity", {})
    requires_grad = state.get("parameter_requires_grad", {})
    for name, value in state.get("tensors", {}).items():
        if value is None:
            setattr(model, name, None)
            continue
        restored = value.to(device=device, dtype=torch.float32).contiguous()
        if bool(identities.get(name, False)):
            restored = torch.nn.Parameter(restored, requires_grad=bool(requires_grad.get(name, False)))
        setattr(model, name, restored)
    model.active_sh_degree = int(state["active_sh_degree"])
    model.max_sh_degree = int(state["max_sh_degree"])
    model.train_test_exp = bool(state.get("train_test_exp", False))
    model.spatial_lr_scale = float(state.get("spatial_lr_scale", 0.0))
    if "reduced_sh_degrees" in state:
        model._reduced_sh_degrees = state["reduced_sh_degrees"].to(
            device=device, dtype=torch.int32
        ).contiguous()
    if "exposure" in state:
        exposure = state["exposure"].to(device=device, dtype=torch.float32).contiguous()
        model._exposure = torch.nn.Parameter(exposure, requires_grad=bool(requires_grad.get("_exposure", False)))
        model.exposure_mapping = {str(key): int(value) for key, value in state.get("exposure_mapping", {}).items()}
    if "pretrained_exposures" in state:
        model.pretrained_exposures = {
            str(key): value.to(device=device, dtype=torch.float32).contiguous()
            for key, value in state["pretrained_exposures"].items()
        }
    return model


def _pipe_from_manifest(manifest: Mapping[str, Any]) -> Any:
    values = manifest.get("pipeline_params") or manifest.get("render_contract", {}).get("render_args", {}).get("pipe") or {}
    if not isinstance(values, Mapping):
        values = {}
    pipe = SimpleNamespace(**dict(values))
    for name, default in (
        ("debug", False),
        ("compute_cov3D_python", False),
        ("convert_SHs_python", False),
        ("antialiasing", False),
    ):
        if not hasattr(pipe, name):
            setattr(pipe, name, default)
    return pipe


def _psnr(image: torch.Tensor, gt: torch.Tensor) -> float:
    mse = ((image - gt) ** 2).reshape(image.shape[0], -1).mean(dim=1)
    value = 20.0 * torch.log10(1.0 / torch.sqrt(mse))
    return float(value.mean().item())


def _png(path: Path, tensor: torch.Tensor) -> dict[str, Any]:
    import numpy as np
    from PIL import Image

    value = tensor.detach().to(device="cpu", dtype=torch.float32).clamp(0.0, 1.0)
    if value.ndim != 3 or value.shape[0] not in (1, 3):
        raise ValueError(f"cannot save image {path}: shape={tuple(value.shape)}")
    array = value.numpy()
    if value.shape[0] == 1:
        image = Image.fromarray(np.rint(array[0] * 255.0).astype(np.uint8), mode="L")
    else:
        image = Image.fromarray(np.rint(array.transpose(1, 2, 0) * 255.0).astype(np.uint8), mode="RGB")
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path)
    return {"path": str(path), "sha256": sha256(path), "size_bytes": int(path.stat().st_size), "width": image.width, "height": image.height, "mode": image.mode}


def _camera_equal(camera: Any, payload: Mapping[str, Any]) -> dict[str, Any]:
    checks: dict[str, bool] = {
        "image_name": str(camera.image_name) == str(payload["image_name"]),
        "width": int(camera.image_width) == int(payload["width"]),
        "height": int(camera.image_height) == int(payload["height"]),
        "FoVx": float(camera.FoVx) == float(payload["FoVx"]),
        "FoVy": float(camera.FoVy) == float(payload["FoVy"]),
        "znear": float(camera.znear) == float(payload["znear"]),
        "zfar": float(camera.zfar) == float(payload["zfar"]),
    }
    tensor_checks: dict[str, bool] = {}
    for name in CAMERA_TENSOR_FIELDS:
        actual = getattr(camera, name)
        expected = payload.get(name)
        if actual is None or expected is None:
            tensor_checks[name] = actual is None and expected is None
        else:
            tensor_checks[name] = torch.equal(actual.detach().cpu(), expected.detach().cpu())
    if not all(checks.values()) or not all(tensor_checks.values()):
        raise RuntimeError(f"camera mismatch: scalar={checks}, tensor={tensor_checks}")
    return {"scalar_torch_equal": checks, "tensor_torch_equal": tensor_checks, "all_torch_equal": True}


def _prepare_runtime(bundle: Mapping[str, Any], device: torch.device) -> tuple[Any, Any, Any, Any, tuple[Any, ...], Mapping[str, Any]]:
    repo = Path(__file__).resolve().parent
    if str(repo) not in sys.path:
        sys.path.insert(0, str(repo))
    from scene import GaussianModel
    from mesh_renderer import MeshRenderer
    from mesh_stage1_environment import Stage1Appearance
    from ndf_stage2_shfit.native_field import NativeScalarSHGaussianField
    from gaussolve_package.meshgs_render import MeshGsRender

    manifest = bundle["manifest"]
    states = bundle.get("states", {})
    main_state = states.get("main_render_state")
    if main_state is None:
        raise RuntimeError("exact bundle has no main_render_state")
    main = _restore_main(main_state, GaussianModel, device)
    mesh_state = states.get("mesh_exact_state")
    mesh = MeshRenderer.from_exact_render_state(mesh_state, device=str(device), use_opengl=False) if mesh_state is not None else None
    appearance = None
    appearance_record = bundle.get("state_paths", {}).get("stage1_appearance")
    if appearance_record is not None:
        appearance = Stage1Appearance.load(appearance_record["resolved_path"], device=str(device)).eval()
        for parameter in appearance.parameters():
            parameter.requires_grad_(False)
    ndf = None
    ndf_record = bundle.get("state_paths", {}).get("ndf_native_scalar")
    if ndf_record is not None:
        ndf = NativeScalarSHGaussianField.load(ndf_record["resolved_path"], device=str(device))
    render_contract = manifest.get("render_contract") or {}
    args_contract = render_contract.get("render_args") or {}
    background = torch.tensor(args_contract.get("background", [1.0, 1.0, 1.0] if manifest.get("white_background", False) else [0.0, 0.0, 0.0]), dtype=torch.float32, device=device)
    render_args = (
        _pipe_from_manifest(manifest),
        background,
        float(args_contract.get("scaling_modifier", 1.0)),
        bool(args_contract.get("separate_sh", False)),
        args_contract.get("override_color"),
        bool(args_contract.get("use_trained_exp", manifest.get("train_test_exp", False))),
        float(args_contract.get("mesh_depth_tolerance", manifest.get("mesh_depth_tolerance", 0.05))),
        ndf,
        bool(args_contract.get("gaussolve_mode", manifest.get("gaussolve_mode", ndf is not None))),
        bool(args_contract.get("ndf_no_mesh_depth_cutoff", manifest.get("ndf_no_mesh_depth_cutoff", False))),
    )
    synchronous = manifest.get("synchronous_render_streams")
    if synchronous is None:
        synchronous = render_contract.get("synchronous_streams", False)
    renderer = MeshGsRender(mesh, appearance, synchronous_streams=bool(synchronous))
    return main, mesh, appearance, ndf, render_args, renderer


def _selected_records(bundle: Mapping[str, Any], views: set[str] | None) -> list[Mapping[str, Any]]:
    records = list(bundle.get("test_order", []))
    if views is None:
        return records
    selected = [record for record in records if str(record["image_name"]) in views]
    missing = views - {str(record["image_name"]) for record in selected}
    if missing:
        raise ValueError(f"requested views are absent from bundle: {sorted(missing)}")
    return selected


def decompose(
    report_dir: str | os.PathLike[str],
    *,
    output_dir: str | os.PathLike[str] | None = None,
    output_json: str | os.PathLike[str] | None = None,
    device: str = "cuda",
    views: Sequence[str] | None = None,
    resume: bool = False,
) -> dict[str, Any]:
    bundle = load_exact_bundle(report_dir, map_location="cpu", load_states=True)
    structural = verify_exact_bundle(bundle)
    if structural["status"] != "PASS":
        raise ValueError("exact bundle structural verification failed: " + "; ".join(structural["errors"]))
    target = torch.device(device)
    if target.type != "cuda":
        raise ValueError("decompose requires a CUDA device; use verify for CPU-only manifest checks")
    selected = _selected_records(bundle, _parse_views(views))
    destination = Path(output_dir).resolve() if output_dir is not None else Path(bundle["report_dir"]).resolve() / "decomposition"
    destination.mkdir(parents=True, exist_ok=True)
    main, _, _, _, render_args, renderer = _prepare_runtime(bundle, target)
    manifest = bundle["manifest"]
    records: list[dict[str, Any]] = []
    eps = float(torch.finfo(torch.float32).eps)
    for record in selected:
        image_name = str(record["image_name"])
        source = Path(record["resolved_path"])
        payload = torch.load(source, map_location="cpu", weights_only=False)
        view_dir = destination / Path(image_name).stem
        metadata_path = view_dir / "metadata.json"
        if resume and metadata_path.is_file():
            previous = json.loads(metadata_path.read_text(encoding="utf-8"))
            if previous.get("status") == PASS_STATUS:
                records.append(previous)
                continue
        camera = camera_from_saved_payload(payload, device=target)
        _camera_equal(camera, payload["camera"])
        with torch.no_grad():
            package = renderer.render_wrapper(camera, main, *render_args, return_components=True)
        components = package.get("hybrid_components")
        if not isinstance(components, Mapping):
            raise RuntimeError(f"{image_name}: wrapper did not return hybrid_components")
        required = {"Cf", "Cg", "Tf", "Cm", "D"}
        if not required.issubset(components):
            raise RuntimeError(f"{image_name}: missing hybrid components {sorted(required - set(components))}")
        component_values_full = {key: components[key] for key in required}
        decomposition_full = decompose_components(**component_values_full)
        component_values = component_values_full
        decomposition = decomposition_full
        dissolve = component_values_full["D"]
        if not torch.isfinite(dissolve).all() or not bool(((dissolve >= 0.0) & (dissolve <= 1.0)).all().item()):
            raise RuntimeError(f"{image_name}: D is non-finite or outside [0,1]")
        expected = payload["render"].to(device=target, dtype=torch.float32).contiguous()
        expected_gt = payload.get("gt")
        if expected_gt is None:
            expected_gt = camera.original_image
        expected_gt = expected_gt.to(device=target, dtype=torch.float32).contiguous()
        replay_full = package["render"].to(device=target, dtype=torch.float32).clamp(0.0, 1.0).contiguous()
        camera_gt = None if camera.original_image is None else camera.original_image.to(device=target, dtype=torch.float32).contiguous()
        if bool(manifest.get("train_test_exp", False)):
            crop_start = component_values_full["Cf"].shape[-1] // 2
            component_values = {key: value[..., crop_start:] for key, value in component_values_full.items()}
            decomposition = {key: value[..., crop_start:] for key, value in decomposition_full.items()}
            if replay_full.shape[-1] != expected.shape[-1]:
                replay = replay_full[..., replay_full.shape[-1] // 2:]
            else:
                replay = replay_full
            if camera_gt is not None and camera_gt.shape[-1] != expected_gt.shape[-1]:
                camera_gt = camera_gt[..., camera_gt.shape[-1] // 2:]
        else:
            replay = replay_full
        if replay.shape != expected.shape:
            raise RuntimeError(f"{image_name}: replay/report shape mismatch {tuple(replay.shape)} != {tuple(expected.shape)}")
        if camera_gt is not None and camera_gt.shape != expected_gt.shape:
            raise RuntimeError(f"{image_name}: camera/report GT shape mismatch {tuple(camera_gt.shape)} != {tuple(expected_gt.shape)}")
        final_error = float((replay - expected).abs().max().item())
        formula_raw_error = float((decomposition_full["Final"] - package["render"]).abs().max().item())
        formula_clamped = decomposition["Final"].clamp(0.0, 1.0)
        formula_error = float((formula_clamped - expected).abs().max().item())
        gt_error = float((camera_gt - expected_gt).abs().max().item()) if camera_gt is not None else 0.0
        training_psnr = None if payload.get("psnr") is None else float(payload["psnr"])
        replay_psnr = _psnr(replay, expected_gt)
        psnr_difference = None if training_psnr is None else replay_psnr - training_psnr
        if not torch.isfinite(replay).all() or final_error > eps or formula_raw_error > 2.0 * eps or formula_error > 2.0 * eps:
            raise RuntimeError(f"{image_name}: exact replay tolerance failed: final={final_error}, formula_raw={formula_raw_error}, formula_clamped={formula_error}")
        if training_psnr is not None and psnr_difference != 0.0:
            raise RuntimeError(f"{image_name}: GPU PSNR mismatch: replay={replay_psnr}, training={training_psnr}")
        view_dir.mkdir(parents=True, exist_ok=True)
        save_values = {
            "GT": expected_gt,
            "Cm": component_values["Cm"],
            "Mesh": component_values["Cm"],
            "D": component_values["D"],
            # Paper display convention: black=dissolve, white=retain.
            "NDF": 1.0 - component_values["D"],
            "Cg": component_values["Cg"],
            "H": decomposition["H"],
            "Erase": decomposition["Erase"],
            "Refill": decomposition["Refill"],
            "MeshContribution": decomposition["MeshContribution"],
            "ResidualContribution": decomposition["ResidualContribution"],
            "Residual": decomposition["ResidualContribution"],
            "Final": expected,
            "Ours": expected,
            "offline_wrapper_Final": replay,
        }
        pngs = {name: _png(view_dir / f"{name}.png", value) for name, value in save_values.items()}
        tensor_payload = dict(decomposition)
        tensor_payload.update({"Final_training_report": expected, "Final_offline_wrapper": replay, "GT_training_report": expected_gt})
        component_path = view_dir / "decomposition.pt"
        torch.save({key: value.detach().to(device="cpu", dtype=torch.float32).contiguous() for key, value in tensor_payload.items()}, component_path)
        metadata = {
            "schema": "exact_render_state_replay_view_v1",
            "status": PASS_STATUS,
            "image_name": image_name,
            "source_payload": str(source),
            "final_png_source": "training_report payload['render']",
            "offline_wrapper_final_png": str(view_dir / "offline_wrapper_Final.png"),
            "camera": _camera_equal(camera, payload["camera"]),
            "identity": {
                "formula": "H=Cf+Tf*Cm; Erase=D*Tf*Cm; Refill=D*(Cg-Cf); MeshContribution=(1-D)*Tf*Cm; ResidualContribution=Cf+Refill; Final=H-Erase+Refill",
                "final_max_abs": final_error,
                "formula_raw_vs_wrapper_max_abs": formula_raw_error,
                "formula_clamped_vs_training_max_abs": formula_error,
                "gt_max_abs": gt_error,
                "torch_equal_final": torch.equal(replay, expected),
                "torch_equal_gt": torch.equal(camera_gt, expected_gt) if camera_gt is not None else None,
                "float32_eps": eps,
                "formula_tolerance": 2.0 * eps,
            },
            "psnr": {"training_report_gpu": training_psnr, "offline_wrapper_gpu": replay_psnr, "difference": psnr_difference, "status": "PASS_EXACT" if psnr_difference == 0.0 else "UNAVAILABLE"},
            "D": {"min": float(component_values["D"].min().item()), "max": float(component_values["D"].max().item()), "range": "[0,1]", "display": "NDF.png stores 1-D (black=dissolve, white=retain)"},
            "pngs": pngs,
            "components": {"path": str(component_path), "sha256": sha256(component_path), "size_bytes": int(component_path.stat().st_size), "keys": sorted(tensor_payload)},
        }
        _dump(metadata_path, metadata)
        records.append(metadata)
    mean_training = None if not records or any(record.get("psnr", {}).get("training_report_gpu") is None for record in records) else sum(record["psnr"]["training_report_gpu"] for record in records) / len(records)
    mean_replay = None if not records else sum(record["psnr"]["offline_wrapper_gpu"] for record in records) / len(records)
    result = {
        "schema": "exact_render_state_replay_v1",
        "status": PASS_STATUS,
        "report_dir": str(Path(bundle["report_dir"])),
        "output_dir": str(destination),
        "view_count": len(records),
        "views": records,
        "identity": {
            "max_final_abs": max((record["identity"]["final_max_abs"] for record in records), default=0.0),
            "max_formula_raw_abs": max((record["identity"]["formula_raw_vs_wrapper_max_abs"] for record in records), default=0.0),
            "max_formula_clamped_abs": max((record["identity"]["formula_clamped_vs_training_max_abs"] for record in records), default=0.0),
            "all_final_torch_equal": all(record["identity"]["torch_equal_final"] for record in records),
            "all_gt_torch_equal": all(record["identity"]["torch_equal_gt"] for record in records),
        },
        "psnr": {"training_report_gpu_mean": mean_training, "offline_wrapper_gpu_mean": mean_replay, "difference": None if mean_training is None else mean_replay - mean_training, "all_view_differences_exact_zero": all(record["psnr"]["difference"] == 0.0 for record in records)},
        "source_manifest": str(bundle["manifest_path"]),
        "render_path": "MeshGsRender.render_wrapper(..., return_components=True)",
    }
    output_json_path = Path(output_json).resolve() if output_json is not None else destination / "replay_summary.json"
    _dump(output_json_path, result)
    result["output_json"] = str(output_json_path)
    return result


def _common_arguments(parser: argparse.ArgumentParser, *, output: bool = False) -> None:
    parser.add_argument("--report-dir", required=True)
    if output:
        parser.add_argument("--output-dir")
        parser.add_argument("--output-json")
        parser.add_argument("--device", default="cuda")
        parser.add_argument("--views", nargs="*", help="View names, optionally comma-separated")
        parser.add_argument("--resume", action="store_true")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    verify_parser = subparsers.add_parser("verify")
    _common_arguments(verify_parser, output=True)
    decompose_parser = subparsers.add_parser("decompose")
    _common_arguments(decompose_parser, output=True)
    args = parser.parse_args(argv)
    if args.command == "verify":
        bundle = load_exact_bundle(args.report_dir, load_states=False, verify_paths=False)
        result = verify_exact_bundle(bundle)
        _dump(None if args.output_json is None else Path(args.output_json), result)
        print(json.dumps(result, ensure_ascii=False, sort_keys=True), flush=True)
        return 0 if result["status"] == "PASS" else 1
    try:
        result = decompose(args.report_dir, output_dir=args.output_dir, output_json=args.output_json, device=args.device, views=args.views, resume=args.resume)
        print(json.dumps({"status": result["status"], "view_count": result["view_count"], "output_dir": result["output_dir"]}, ensure_ascii=False, sort_keys=True), flush=True)
        return 0
    except Exception as error:
        result = {"schema": "exact_render_state_replay_v1", "status": "FAIL", "error_type": type(error).__name__, "error": str(error)}
        _dump(None if args.output_json is None else Path(args.output_json), result)
        print(json.dumps(result, ensure_ascii=False, sort_keys=True), flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
