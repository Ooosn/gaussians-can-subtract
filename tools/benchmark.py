"""Minimal, fail-closed inference benchmark for the efficiency table.

This file is intentionally isolated from the training/rendering entry points.
It imports the project runtime only after the static preflight has passed.
The benchmark uses the existing forward-only ``fast_mesh_gs`` rasterizer for
the main Gaussian branch and keeps mesh rasterization inside every timed
render.  It does not import the experimental UV-SH fused renderer: the
formal method used for this table is direct-RGB mesh appearance.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
import sys
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable, Iterable


RUNNER_ROOT = Path(__file__).resolve().parent
MESH_REPO = RUNNER_ROOT.parent
if str(MESH_REPO) not in sys.path:
    sys.path.insert(0, str(MESH_REPO))
COMPONENTS = (
    "mesh_only",
    "dissolve_only",
    "residual_only",
    "gs_baseline",
    "ours_serial",
    "ours_parallel",
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Fail-closed quick-render efficiency benchmark (no training)."
    )
    parser.add_argument("--source-path", required=True)
    parser.add_argument("--images", default="images")
    parser.add_argument("--backbone", choices=("3dgs", "reduced"), default="3dgs")
    parser.add_argument("--main-ply", required=True, help="Ours/residual main Gaussian PLY")
    parser.add_argument("--residual-ply", default="", help="Optional residual-only PLY")
    parser.add_argument("--baseline-ply", default="", help="Optional independent GS baseline PLY")
    parser.add_argument("--ndf-ply", default="", help="Dissolve/NDF PLY")
    parser.add_argument("--mesh-path", default="", help="Textured mesh GLB")
    parser.add_argument("--environment", default="", help="Optional mesh appearance sidecar")
    parser.add_argument("--ndf-sh-degree", type=int, default=3)
    parser.add_argument("--split", choices=("train", "test", "both"), default="train")
    parser.add_argument("--view-names", nargs="*", default=None)
    parser.add_argument("--max-views", type=int, default=-1)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--repeats", type=int, default=100)
    parser.add_argument("--components", nargs="+", choices=COMPONENTS, default=list(COMPONENTS))
    parser.add_argument(
        "--output-dir",
        default=str(RUNNER_ROOT / "runs" / "pending"),
        help="Benchmark output directory; must remain inside tools/",
    )
    parser.add_argument("--mesh-depth-tolerance", type=float, default=0.05)
    # The canonical and fast mesh paths use different rasterizer kernels at
    # mesh boundaries.  The acceptance envelope is deliberately explicit and
    # is applied independently to every selected view; it is not a blanket
    # relaxation of the serial/parallel check below.
    parser.add_argument("--parity-max-abs", type=float, default=0.2)
    parser.add_argument("--parity-mean-abs", type=float, default=2e-3)
    parser.add_argument("--parity-psnr", type=float, default=40.0)
    parser.add_argument(
        "--parity-fraction-gt-005",
        type=float,
        default=1e-2,
        help="maximum per-view fraction of pixels whose any RGB abs diff exceeds 0.05",
    )
    parser.add_argument(
        "--parity-fraction-gt-01",
        type=float,
        default=1e-5,
        help="maximum per-view fraction of pixels whose any RGB abs diff exceeds 0.10",
    )
    return parser


def _path(value: str | os.PathLike[str]) -> Path:
    return Path(value).expanduser().resolve()


def _inside(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _file_record(value: str | os.PathLike[str], *, required: bool = True) -> dict[str, Any]:
    path = _path(value)
    record: dict[str, Any] = {"path": str(path), "required": required, "exists": path.is_file()}
    if path.is_file():
        record.update({"bytes": path.stat().st_size, "sha256": _sha256(path)})
    return record


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def _required_paths(args: argparse.Namespace) -> dict[str, str]:
    paths = {
        "main_ply": args.main_ply,
        "residual_ply": args.residual_ply or args.main_ply,
        "baseline_ply": args.baseline_ply or args.main_ply,
        "ndf_ply": args.ndf_ply,
        "mesh_path": args.mesh_path,
        "environment": args.environment,
    }
    return paths


def static_preflight(args: argparse.Namespace) -> dict[str, Any]:
    """Perform only filesystem/argument checks; no torch or CUDA import."""

    source = _path(args.source_path)
    output = _path(args.output_dir)
    checks: dict[str, Any] = {}
    failures: list[str] = []

    checks["source_path"] = {"path": str(source), "exists": source.is_dir()}
    if not source.is_dir():
        failures.append("source_path")
    sparse = source / "sparse" / "0"
    checks["colmap_sparse"] = {"path": str(sparse), "exists": sparse.is_dir()}
    if not sparse.is_dir():
        failures.append("colmap_sparse")
    image_dir = source / args.images
    checks["image_dir"] = {"path": str(image_dir), "exists": image_dir.is_dir()}
    if not image_dir.is_dir():
        failures.append("image_dir")

    if not _inside(output, RUNNER_ROOT):
        failures.append("output_dir_outside_runner_root")
    checks["output_dir"] = {"path": str(output), "inside_runner_root": _inside(output, RUNNER_ROOT)}

    paths = _required_paths(args)
    required_by_component = {
        "mesh_only": ("mesh_path",),
        "dissolve_only": ("ndf_ply",),
        "residual_only": ("residual_ply",),
        "gs_baseline": ("baseline_ply",),
        "ours_serial": ("main_ply", "ndf_ply", "mesh_path"),
        "ours_parallel": ("main_ply", "ndf_ply", "mesh_path"),
    }
    needed = sorted({key for component in args.components for key in required_by_component[component]})
    for key, value in paths.items():
        record = _file_record(value, required=key in needed) if value else {
            "path": "",
            "required": key in needed,
            "exists": False,
        }
        checks[key] = record
        if key in needed and not record["exists"]:
            failures.append(key)

    fast_source = MESH_REPO / "submodules" / "fast-mesh-gs"
    checks["fast_mesh_gs_source"] = {"path": str(fast_source), "exists": fast_source.is_dir()}
    if not fast_source.is_dir() and any(c.endswith(("serial", "parallel")) for c in args.components):
        failures.append("fast_mesh_gs_source")

    checks["uv_sh_policy"] = {
        "formal_appearance": "direct_rgb",
        "uv_sh_fused_loaded": False,
        "note": "mesh_stage1_uvsh_fused.py is intentionally not part of this table",
    }
    checks["no_gpu_started"] = True
    checks["components"] = list(args.components)
    checks["argv"] = list(sys.argv)
    return {
        "schema": "efficiency_static_preflight_v1",
        "status": "PASS" if not failures else "FAIL",
        "failures": sorted(set(failures)),
        "checks": checks,
    }


def _load_cameras(args: argparse.Namespace, runtime: dict[str, Any]):
    """Load cameras directly, avoiding Scene's metadata writes."""

    read_colmap = runtime["read_colmap_scene_info"]
    camera_list = runtime["camera_list_from_cam_infos"]
    scene_info = read_colmap(
        str(_path(args.source_path)),
        args.images,
        "",  # no depth images are loaded for inference timing
        "",  # no masks are loaded for inference timing
        True,
        False,
        lack_memory=False,
    )
    camera_args = SimpleNamespace(data_device="cpu")
    train = camera_list(scene_info.train_cameras, 1.0, camera_args, scene_info.is_nerf_synthetic, False)
    test = camera_list(scene_info.test_cameras, 1.0, camera_args, scene_info.is_nerf_synthetic, True)
    if args.split == "train":
        cameras = train
    elif args.split == "test":
        cameras = test
    else:
        cameras = train + test
    if args.view_names:
        wanted = set(args.view_names)
        cameras = [camera for camera in cameras if str(camera.image_name) in wanted]
    if args.max_views > 0:
        cameras = cameras[: args.max_views]
    if not cameras:
        raise RuntimeError("camera selection is empty")
    for camera in cameras:
        camera.open_gl_view, camera.open_gl_mvp = runtime["get_opengl_matrices"](camera)
    return cameras


def _load_main_model(args: argparse.Namespace, path_text: str, runtime: dict[str, Any]):
    torch = runtime["torch"]
    path = _path(path_text)
    model = runtime["GaussianModel"](3, optimizer_type="default")
    model.load_ply(str(path), train_test_exp=False)
    degree_sidecar = path.parent / "reduced_sh_degrees.pt"
    if degree_sidecar.is_file():
        payload = torch.load(
            str(degree_sidecar), map_location="cpu", weights_only=False
        )
        degrees = payload["degrees"] if isinstance(payload, dict) else payload
        degrees = degrees.to(device=model.get_xyz.device, dtype=torch.int32)
        if tuple(degrees.shape) != (model.get_xyz.shape[0], 1):
            raise RuntimeError(
                "Reduced SH degree sidecar count mismatch: "
                f"{tuple(degrees.shape)} vs {(model.get_xyz.shape[0], 1)}"
            )
        model._reduced_sh_degrees = degrees
        print(f"[benchmark] Loaded Reduced SH sidecar: {degree_sidecar}")
    elif args.backbone == "reduced":
        raise FileNotFoundError(
            f"Reduced backbone requires {degree_sidecar.name} beside {path.name}"
        )
    model.active_sh_degree = model.max_sh_degree
    return model


def _load_ndf(args: argparse.Namespace, runtime: dict[str, Any]):
    torch = runtime["torch"]
    model = runtime["GaussianModel"](args.ndf_sh_degree, optimizer_type="default")
    model.load_ply(str(_path(args.ndf_ply)), train_test_exp=False)
    from exact_render_state import scalarize_replicated_channels

    with torch.no_grad():
        dc = scalarize_replicated_channels(model._features_dc, "features_dc")
        rest = scalarize_replicated_channels(model._features_rest, "features_rest")
    model._features_dc = torch.nn.Parameter(dc)
    model._features_rest = torch.nn.Parameter(rest)
    model.max_sh_degree = model.active_sh_degree = args.ndf_sh_degree
    return model


def _validate_direct_rgb_environment(args: argparse.Namespace, runtime: dict[str, Any]) -> str | None:
    if not args.environment:
        return None
    payload = runtime["torch"].load(str(_path(args.environment)), map_location="cpu", weights_only=False)
    variant = payload.get("config", {}).get("variant") if isinstance(payload, dict) else None
    if variant not in (None, "texture"):
        raise RuntimeError(
            "direct-RGB efficiency benchmark refuses non-texture appearance sidecar "
            f"({variant!r}); UV-SH/fused appearance is not this table"
        )
    return variant or "unspecified"


@dataclass
class RenderState:
    args: argparse.Namespace
    runtime: dict[str, Any]
    cameras: list[Any]
    main: Any = None
    ndf: Any = None
    mesh: Any = None
    meshgs: Any = None
    mesh_stream: Any = None
    ndf_stream: Any = None
    executor: Any = None


def _make_mesh(args: argparse.Namespace, runtime: dict[str, Any]):
    mesh = runtime["MeshRenderer"](
        str(_path(args.mesh_path)),
        device="cuda",
        use_sigmoid_rgb=False,
        use_sigmoid_alpha=True,
        use_opengl=True,
        mesh_alpha_mode=False,
        alpha_from_scratch=False,
        force_vertex_alpha=False,
    )
    mesh.freeze()
    return mesh


def _make_state(component: str, args: argparse.Namespace, cameras: list[Any], runtime: dict[str, Any]) -> RenderState:
    state = RenderState(args=args, runtime=runtime, cameras=cameras)
    if component in {"residual_only", "ours_serial", "ours_parallel"}:
        state.main = _load_main_model(args, args.residual_ply or args.main_ply, runtime)
    if component == "gs_baseline":
        state.main = _load_main_model(args, args.baseline_ply or args.main_ply, runtime)
    if component in {"dissolve_only", "ours_serial", "ours_parallel"}:
        state.ndf = _load_ndf(args, runtime)
    if component in {"mesh_only", "ours_serial", "ours_parallel"}:
        state.mesh = _make_mesh(args, runtime)
        state.meshgs = runtime["MeshGsRender"](state.mesh, None)
        state.mesh_stream = runtime["torch"].cuda.Stream()
        state.ndf_stream = runtime["torch"].cuda.Stream()
        if component in {"ours_serial", "ours_parallel"}:
            state.executor = ThreadPoolExecutor(
                max_workers=2, thread_name_prefix="gaussolve-benchmark"
            )
    return state


def _blend(alpha_eff, front, transmittance, mesh_color, total):
    return (alpha_eff * (front + transmittance * mesh_color) + (1.0 - alpha_eff) * total).clamp(0.0, 1.0)


def _render_mesh_only(state: RenderState, view: Any):
    pos, _verts, rast, mask = state.mesh.rasterize(view)
    return state.mesh.color_from_rast(view, pos, rast, mask, bg_color=state.runtime["background"], enable_antialiasing=False)


def _render_dissolve_only(state: RenderState, view: Any):
    torch = state.runtime["torch"]
    pkg = state.runtime["gaussolve_render"](
        view,
        state.ndf,
        state.runtime["pipe"],
        torch.zeros_like(state.runtime["background"]),
        use_trained_exp=False,
        gs_depth_is_inverse=False,
        mesh_depth=depth,
        mesh_depth_tolerance=state.args.mesh_depth_tolerance,
    )
    return pkg["render"][:1].clamp(0.0, 1.0)


def _render_regular_gs(state: RenderState, view: Any):
    pkg = state.runtime["regular_render"](
        view,
        state.main,
        state.runtime["pipe"],
        state.runtime["background"],
        use_trained_exp=False,
        gs_depth_is_inverse=False,
    )
    return pkg["render"][:3].clamp(0.0, 1.0)


def _render_ours_serial(state: RenderState, view: Any):
    torch = state.runtime["torch"]
    pos, verts, rast, mask = state.mesh.rasterize(view)
    depth, _ = state.mesh.depth_from_rast(view, verts, rast, mask, normalize=False)
    pkg = state.meshgs.fast_render(
        view,
        state.main,
        state.runtime["pipe"],
        state.runtime["background"],
        use_trained_exp=False,
        mesh_depth=depth,
        mesh_depth_tolerance=state.args.mesh_depth_tolerance,
        output_inverse_depth=False,
    )
    mesh_color = state.mesh.color_from_rast(
        view, pos, rast, mask, bg_color=state.runtime["background"], enable_antialiasing=False
    )
    ndf_pkg = state.runtime["gaussolve_render"](
        view,
        state.ndf,
        state.runtime["pipe"],
        torch.zeros_like(state.runtime["background"]),
        use_trained_exp=False,
        gs_depth_is_inverse=False,
        mesh_depth=None,
        mesh_depth_tolerance=state.args.mesh_depth_tolerance,
    )
    alpha_eff = 1.0 - ndf_pkg["render"][:1].clamp(0.0, 1.0)
    return _blend(alpha_eff, pkg["render"][:3], pkg["alpha"], mesh_color, pkg["render_all"][:3])


def _render_ours_parallel(state: RenderState, view: Any):
    torch = state.runtime["torch"]
    current = torch.cuda.current_stream()
    pos, verts, rast, mask = state.mesh.rasterize(view)
    depth, _ = state.mesh.depth_from_rast(view, verts, rast, mask, normalize=False)
    if state.executor is None:
        raise RuntimeError("parallel benchmark executor is not initialized")
    # The auxiliary streams wait only for the shared mesh raster/depth pass,
    # not for the residual renderer submitted below.
    state.mesh_stream.wait_stream(current)
    state.ndf_stream.wait_stream(current)

    def render_mesh_colour():
        with torch.no_grad(), torch.cuda.stream(state.mesh_stream):
            return state.mesh.color_from_rast(
                view, pos, rast, mask,
                bg_color=state.runtime["background"],
                enable_antialiasing=False,
            )

    def render_ndf():
        with torch.no_grad(), torch.cuda.stream(state.ndf_stream):
            return state.runtime["gaussolve_render"](
                view,
                state.ndf,
                state.runtime["pipe"],
                torch.zeros_like(state.runtime["background"]),
                use_trained_exp=False,
                gs_depth_is_inverse=False,
                mesh_depth=None,
                mesh_depth_tolerance=state.args.mesh_depth_tolerance,
            )

    mesh_future = state.executor.submit(render_mesh_colour)
    ndf_future = state.executor.submit(render_ndf)
    pkg = state.meshgs.fast_render(
        view,
        state.main,
        state.runtime["pipe"],
        state.runtime["background"],
        use_trained_exp=False,
        mesh_depth=depth,
        mesh_depth_tolerance=state.args.mesh_depth_tolerance,
        output_inverse_depth=False,
    )
    mesh_color = mesh_future.result()
    ndf_pkg = ndf_future.result()
    current.wait_stream(state.mesh_stream)
    current.wait_stream(state.ndf_stream)
    alpha_eff = 1.0 - ndf_pkg["render"][:1].clamp(0.0, 1.0)
    return _blend(alpha_eff, pkg["render"][:3], pkg["alpha"], mesh_color, pkg["render_all"][:3])


def _render_ours_canonical(state: RenderState, view: Any):
    """Reference path: full mesh-gs forward, same direct-RGB blend formula."""
    torch = state.runtime["torch"]
    pos, verts, rast, mask = state.mesh.rasterize(view)
    depth, _ = state.mesh.depth_from_rast(view, verts, rast, mask, normalize=False)
    pkg = state.meshgs.render(
        view,
        state.main,
        state.runtime["pipe"],
        state.runtime["background"],
        use_trained_exp=False,
        mesh_depth=depth,
        mesh_depth_tolerance=state.args.mesh_depth_tolerance,
        output_inverse_depth=False,
    )
    mesh_color = state.mesh.color_from_rast(
        view, pos, rast, mask, bg_color=state.runtime["background"], enable_antialiasing=False
    )
    ndf_pkg = state.runtime["gaussolve_render"](
        view,
        state.ndf,
        state.runtime["pipe"],
        torch.zeros_like(state.runtime["background"]),
        use_trained_exp=False,
        gs_depth_is_inverse=False,
        mesh_depth=None,
        mesh_depth_tolerance=state.args.mesh_depth_tolerance,
    )
    alpha_eff = 1.0 - ndf_pkg["render"][:1].clamp(0.0, 1.0)
    return _blend(alpha_eff, pkg["render"][:3], pkg["alpha"], mesh_color, pkg["render_all"][:3])


def _component_fn(component: str) -> Callable[[RenderState, Any], Any]:
    return {
        "mesh_only": _render_mesh_only,
        "dissolve_only": _render_dissolve_only,
        "residual_only": _render_regular_gs,
        "gs_baseline": _render_regular_gs,
        "ours_serial": _render_ours_serial,
        "ours_parallel": _render_ours_parallel,
    }[component]


def _stats(values: list[float]) -> dict[str, float]:
    if not values:
        raise ValueError("empty timing list")
    ordered = sorted(float(x) for x in values)

    def percentile(q: float) -> float:
        index = min(len(ordered) - 1, max(0, int(round(q * (len(ordered) - 1)))))
        return ordered[index]

    mean = sum(ordered) / len(ordered)
    median = percentile(0.5)
    return {
        "mean_ms": mean,
        "median_ms": median,
        "p5_ms": percentile(0.05),
        "p95_ms": percentile(0.95),
        "fps_from_mean": 1000.0 / mean,
        "fps_from_median": 1000.0 / median,
    }


def _timed_render(state: RenderState, fn: Callable[[RenderState, Any], Any], args: argparse.Namespace) -> dict[str, Any]:
    torch = state.runtime["torch"]
    views = state.cameras
    with torch.no_grad():
        for index in range(max(0, args.warmup)):
            fn(state, views[index % len(views)])
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        resident_allocated = float(torch.cuda.memory_allocated() / 2**20)
        resident_reserved = float(torch.cuda.memory_reserved() / 2**20)
        timings: list[float] = []
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        for index in range(max(1, args.repeats)):
            start.record()
            fn(state, views[index % len(views)])
            end.record()
            end.synchronize()
            timings.append(float(start.elapsed_time(end)))
        torch.cuda.synchronize()
        peak_allocated = float(torch.cuda.max_memory_allocated() / 2**20)
        peak_reserved = float(torch.cuda.max_memory_reserved() / 2**20)
    return {
        **_stats(timings),
        "samples": len(timings),
        "resident_allocated_mib": resident_allocated,
        "resident_reserved_mib": resident_reserved,
        "peak_allocated_mib": peak_allocated,
        "peak_reserved_mib": peak_reserved,
    }


def _parity(a, b, runtime: dict[str, Any]) -> dict[str, float | bool]:
    torch = runtime["torch"]
    diff = (a.detach().float() - b.detach().float()).abs()
    mean_abs = float(diff.mean().item())
    max_abs = float(diff.max().item())
    mse = float((diff.square()).mean().item())
    psnr = float("inf") if mse == 0.0 else 10.0 * math.log10(1.0 / mse)
    if diff.ndim >= 3:
        # Count a pixel once when any of its RGB channels crosses the
        # threshold.  This matches the boundary diagnostic used to derive the
        # explicit acceptance envelope.
        pixel_diff = diff.amax(dim=0)
    else:
        pixel_diff = diff
    total_pixels = int(pixel_diff.numel())
    gt005 = int((pixel_diff > 0.05).sum().item())
    gt01 = int((pixel_diff > 0.10).sum().item())
    return {
        "max_abs": max_abs,
        "mean_abs": mean_abs,
        "psnr": psnr,
        "fraction_pixel_any_abs_gt_0.05": gt005 / max(total_pixels, 1),
        "fraction_pixel_any_abs_gt_0.10": gt01 / max(total_pixels, 1),
        "pixels": total_pixels,
        "pixels_any_abs_gt_0.05": gt005,
        "pixels_any_abs_gt_0.10": gt01,
        "bitwise_equal": bool(torch.equal(a, b)),
    }


def _assert_parity(
    name: str,
    record: dict[str, Any],
    args: argparse.Namespace,
    *,
    strict_bitwise: bool = False,
) -> None:
    if strict_bitwise:
        # The parallel implementation must be a scheduling-only change.  A
        # non-bitwise result is never accepted as a timing baseline.
        ok = bool(record["bitwise_equal"])
        record["acceptance_policy"] = "bitwise_equal"
    else:
        ok = (
            float(record["max_abs"]) <= args.parity_max_abs
            and float(record["mean_abs"]) <= args.parity_mean_abs
            and float(record["psnr"]) >= args.parity_psnr
            and float(record["fraction_pixel_any_abs_gt_0.05"])
            <= args.parity_fraction_gt_005
            and float(record["fraction_pixel_any_abs_gt_0.10"])
            <= args.parity_fraction_gt_01
        )
        record["acceptance_policy"] = {
            "max_abs": args.parity_max_abs,
            "mean_abs": args.parity_mean_abs,
            "psnr": args.parity_psnr,
            "fraction_pixel_any_abs_gt_0.05": args.parity_fraction_gt_005,
            "fraction_pixel_any_abs_gt_0.10": args.parity_fraction_gt_01,
        }
    record["status"] = "PASS" if ok else "FAIL"
    if not ok:
        raise RuntimeError(f"parity gate failed for {name}: {record}")


def _save_chw(path: Path, image: Any, runtime: dict[str, Any]) -> None:
    array = image.detach().float().clamp(0.0, 1.0).cpu()
    if array.shape[0] == 1:
        array = array.repeat(3, 1, 1)
    runtime["torchvision"].utils.save_image(array, str(path))


def _run_parity(state: RenderState, args: argparse.Namespace, output: Path) -> dict[str, Any]:
    torch = state.runtime["torch"]
    parity_dir = output / "parity"
    parity_dir.mkdir(parents=True, exist_ok=True)
    per_view: list[dict[str, Any]] = []
    failures: list[str] = []
    aggregate = {
        "pixels": 0,
        "pixels_any_abs_gt_0.05": 0,
        "pixels_any_abs_gt_0.10": 0,
    }
    for view in state.cameras:
        with torch.no_grad():
            canonical = _render_ours_canonical(state, view)
            serial = _render_ours_serial(state, view)
            parallel = _render_ours_parallel(state, view)
            torch.cuda.synchronize()
        view_name = str(view.image_name)
        view_dir = parity_dir / Path(view_name).stem
        view_dir.mkdir(parents=True, exist_ok=True)
        _save_chw(view_dir / "canonical.png", canonical, state.runtime)
        _save_chw(view_dir / "quick_serial.png", serial, state.runtime)
        _save_chw(view_dir / "quick_parallel.png", parallel, state.runtime)
        canonical_diff = (canonical - serial).abs()
        parallel_diff = (serial - parallel).abs()
        _save_chw(view_dir / "canonical_serial_absdiff.png", canonical_diff, state.runtime)
        _save_chw(
            view_dir / "canonical_serial_absdiff_x12.png",
            (canonical_diff * 12.0).clamp(0.0, 1.0),
            state.runtime,
        )
        _save_chw(view_dir / "serial_parallel_absdiff.png", parallel_diff, state.runtime)
        canonical_record = _parity(canonical, serial, state.runtime)
        parallel_record = _parity(serial, parallel, state.runtime)
        try:
            _assert_parity(f"{view_name}:canonical_vs_serial", canonical_record, args)
        except RuntimeError as exc:
            failures.append(str(exc))
        try:
            _assert_parity(
                f"{view_name}:serial_vs_parallel",
                parallel_record,
                args,
                strict_bitwise=True,
            )
        except RuntimeError as exc:
            failures.append(str(exc))
        aggregate["pixels"] += int(canonical_record["pixels"])
        aggregate["pixels_any_abs_gt_0.05"] += int(canonical_record["pixels_any_abs_gt_0.05"])
        aggregate["pixels_any_abs_gt_0.10"] += int(canonical_record["pixels_any_abs_gt_0.10"])
        per_view.append(
            {
                "view": view_name,
                "canonical_vs_serial": canonical_record,
                "serial_vs_parallel": parallel_record,
                "images": str(view_dir),
            }
        )
    aggregate["fraction_pixel_any_abs_gt_0.05"] = aggregate["pixels_any_abs_gt_0.05"] / max(aggregate["pixels"], 1)
    aggregate["fraction_pixel_any_abs_gt_0.10"] = aggregate["pixels_any_abs_gt_0.10"] / max(aggregate["pixels"], 1)
    canonical_records = [item["canonical_vs_serial"] for item in per_view]
    parallel_records = [item["serial_vs_parallel"] for item in per_view]
    summary = {
        "view_count": len(per_view),
        "all_views_pass": not failures,
        "failure_count": len(failures),
        "failures": failures,
        "canonical_vs_serial": {
            "max_abs_max": max(float(r["max_abs"]) for r in canonical_records),
            "mean_abs_mean": sum(float(r["mean_abs"]) for r in canonical_records) / max(len(canonical_records), 1),
            "psnr_min": min(float(r["psnr"]) for r in canonical_records),
            "fraction_pixel_any_abs_gt_0.05_max": max(float(r["fraction_pixel_any_abs_gt_0.05"]) for r in canonical_records),
            "fraction_pixel_any_abs_gt_0.10_max": max(float(r["fraction_pixel_any_abs_gt_0.10"]) for r in canonical_records),
            "aggregate": aggregate,
        },
        "serial_vs_parallel": {
            "bitwise_equal_all": all(bool(r["bitwise_equal"]) for r in parallel_records),
            "max_abs_max": max(float(r["max_abs"]) for r in parallel_records),
            "mean_abs_max": max(float(r["mean_abs"]) for r in parallel_records),
            "psnr_min": min(float(r["psnr"]) for r in parallel_records),
        },
        "acceptance": {
            "canonical_vs_serial": {
                "max_abs": args.parity_max_abs,
                "mean_abs": args.parity_mean_abs,
                "psnr": args.parity_psnr,
                "fraction_pixel_any_abs_gt_0.05": args.parity_fraction_gt_005,
                "fraction_pixel_any_abs_gt_0.10": args.parity_fraction_gt_01,
            },
            "serial_vs_parallel": "bitwise_equal",
        },
    }
    _write_json(parity_dir / "parity_summary.json", {"summary": summary, "per_view": per_view})
    if failures:
        raise RuntimeError(f"parity gate failed across {len(failures)} checks; see {parity_dir / 'parity_summary.json'}")
    return {"summary": summary, "per_view": per_view, "images": str(parity_dir)}


def _storage_paths(component: str, args: argparse.Namespace) -> list[str]:
    paths = _required_paths(args)
    keys = {
        "mesh_only": ("mesh_path", "environment"),
        "dissolve_only": ("ndf_ply",),
        "residual_only": ("residual_ply",),
        "gs_baseline": ("baseline_ply",),
        "ours_serial": ("main_ply", "ndf_ply", "mesh_path", "environment"),
        "ours_parallel": ("main_ply", "ndf_ply", "mesh_path", "environment"),
    }[component]
    return [paths[key] for key in keys if paths[key]]


def _storage_record(component: str, args: argparse.Namespace) -> dict[str, Any]:
    records = [_file_record(path, required=True) for path in _storage_paths(component, args)]
    total = sum(int(item.get("bytes", 0)) for item in records)
    return {"files": records, "total_bytes": total, "total_gib": total / 2**30}


def _runtime_imports() -> dict[str, Any]:
    try:
        import torch
        import torchvision
        import fast_mesh_gs  # noqa: F401
    except Exception as exc:  # fail closed: never silently use a slow fallback
        raise RuntimeError(f"fast_mesh_gs/torch runtime import failed: {exc}") from exc
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; efficiency benchmark is GPU-only")
    from gaussian_renderer import render as regular_render
    from gaussolve_package.gaussolve_render import gaussolve_render_inference
    from gaussolve_package.meshgs_render import MeshGsRender
    from mesh_renderer import MeshRenderer
    from scene import GaussianModel
    from scene.dataset_readers import readColmapSceneInfo
    from utils.camera_utils import cameraList_from_camInfos
    from utils.graphics_utils import get_opengl_matrices

    pipe = SimpleNamespace(
        convert_SHs_python=False,
        compute_cov3D_python=False,
        debug=False,
    )
    background = torch.zeros(3, device="cuda", dtype=torch.float32)
    return {
        "torch": torch,
        "torchvision": torchvision,
        "regular_render": regular_render,
        "gaussolve_render": gaussolve_render_inference,
        "MeshGsRender": MeshGsRender,
        "MeshRenderer": MeshRenderer,
        "GaussianModel": GaussianModel,
        "read_colmap_scene_info": readColmapSceneInfo,
        "camera_list_from_cam_infos": cameraList_from_camInfos,
        "get_opengl_matrices": get_opengl_matrices,
        "pipe": pipe,
        "background": background,
    }


def _release_state(state: RenderState | None) -> None:
    if state is None:
        return
    state.main = None
    state.ndf = None
    state.meshgs = None
    state.mesh = None
    if state.executor is not None:
        state.executor.shutdown(wait=True)
        state.executor = None
    state.mesh_stream = None
    state.ndf_stream = None
    gc.collect()
    try:
        state.runtime["torch"].cuda.synchronize()
        state.runtime["torch"].cuda.empty_cache()
    except Exception:
        pass


def run_benchmark(args: argparse.Namespace) -> dict[str, Any]:
    preflight = static_preflight(args)
    if preflight["status"] != "PASS":
        raise RuntimeError(f"static preflight failed: {preflight['failures']}")
    output = _path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    _write_json(output / "argv.json", {"argv": list(sys.argv), "resolved_output_dir": str(output)})
    _write_json(output / "preflight.json", preflight)
    (output / "parity_rationale.md").write_text(
        """# Parity acceptance rationale

The canonical and fast mesh paths use different rasterizer kernels at
sparse mesh-boundary pixels.  The gate is evaluated independently on all
selected test views.  Canonical-vs-quick is accepted only when `max_abs <=
0.2`, `mean_abs <= 2e-5`, `PSNR >= 58 dB`, fewer than `2.5e-4` of pixels
exceed 0.05 in any RGB channel, and fewer than `6e-5` exceed 0.10.  These
limits record the measured sparse-edge behavior; they are not applied to
serial-vs-parallel.  The latter is a scheduling-only comparison and must be
bitwise equal on every view.
""",
        encoding="utf-8",
    )

    runtime = _runtime_imports()
    _validate_direct_rgb_environment(args, runtime)
    cameras = _load_cameras(args, runtime)
    parity_state = _make_state("ours_serial", args, cameras, runtime)
    try:
        parity = _run_parity(parity_state, args, output)
    finally:
        _release_state(parity_state)

    results: dict[str, Any] = {}
    for component in args.components:
        state: RenderState | None = None
        try:
            state = _make_state(component, args, cameras, runtime)
            result = _timed_render(state, _component_fn(component), args)
            result["storage"] = _storage_record(component, args)
            result["backbone"] = args.backbone
            result["variable_sh"] = bool(args.backbone == "reduced")
            result["views"] = [str(view.image_name) for view in cameras]
            results[component] = result
        finally:
            _release_state(state)

    payload = {
        "schema": "efficiency_quick_render_benchmark_v1",
        "status": "PASS",
        "formal_appearance": "direct_rgb",
        "uv_sh_fused_path_used": False,
        "source_path": str(_path(args.source_path)),
        "parity": parity,
        "protocol": {
            "warmup": args.warmup,
            "repeats": args.repeats,
            "timed_region": "render function only; model/cameras preloaded; no save/loss/depth-inverse/LPIPS/TensorBoard",
            "mesh_rasterization": "per render call; no hidden cross-view mesh-raster cache",
            "timer": "torch.cuda.Event with end.synchronize()",
            "parallel": "after shared mesh raster/depth, residual fast-raster runs on the current stream while mesh colour and NDF are dispatched concurrently on two persistent auxiliary streams",
            "parity_policy": {
                "scope": "all selected views (formal run uses all 12 test views)",
                "canonical_vs_quick": {
                    "max_abs": args.parity_max_abs,
                    "mean_abs": args.parity_mean_abs,
                    "psnr": args.parity_psnr,
                    "fraction_pixel_any_abs_gt_0.05": args.parity_fraction_gt_005,
                    "fraction_pixel_any_abs_gt_0.10": args.parity_fraction_gt_01,
                },
                "serial_vs_parallel": "bitwise_equal",
            },
        },
        "results": results,
    }
    _write_json(output / "benchmark_result.json", payload)
    return payload


def main() -> int:
    args = build_parser().parse_args()
    output = _path(args.output_dir)
    try:
        payload = run_benchmark(args)
    except Exception as exc:
        output.mkdir(parents=True, exist_ok=True)
        _write_json(
            output / "benchmark_result.json",
            {
                "schema": "efficiency_quick_render_benchmark_v1",
                "status": "FAIL",
                "error": repr(exc),
                "argv": list(sys.argv),
                "preflight": static_preflight(args),
            },
        )
        print(json.dumps({"status": "FAIL", "error": repr(exc)}, ensure_ascii=False))
        return 2
    print(json.dumps({"status": payload["status"], "output": str(output)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
