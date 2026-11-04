#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import os
import csv
import json
import torch
import torchvision
from plyfile import PlyData
from datetime import datetime
from random import randint
from utils.gaussolve_utils_initfilter import tv_loss
from utils.colmap_init_stability_filter import (
    build_colmap_initialization_stability_mask,
    write_initialization_filter_audit,
)
from trimesh.constants import ToleranceMesh
from utils.loss_utils import l1_loss, ssim
from gaussian_renderer import render as regular_render, network_gui
from lpipsPyTorch import lpips
from gaussolve_package.meshgs_render import MeshGsRender
from gaussolve_package.gaussolve_render import gaussolve_render
from gaussolve_package.gaussolve_model import collect_bad_mesh_vertices_multiview
from utils.sh_utils import C0, eval_sh
from simple_knn._C import distCUDA2
import sys
from scene import Scene, GaussianModel
from utils.general_utils import inverse_sigmoid, safe_state, get_expon_lr_func
import uuid
from tqdm import tqdm
from utils.image_utils import psnr
from utils.graphics_utils import get_opengl_matrices
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams
from mesh_renderer import MeshRenderer
from mesh_aware_split import MeshAwareSplitProjector
from mesh_stage1_environment import Stage1Appearance
from ndf_stage2_shfit.native_field import NativeScalarSHGaussianField
from exact_render_state import save_exact_bundle, save_test_view
from utils.gaussolve_utils_initfilter import clone_gaussians_for_patching_multiview, initialize_empty_train_gaussians_from_source, prune_gaussians_on_good_mesh_multiview, seed_gaussians_for_patching_multiview
from utils.mcmc_main_ops import (
    apply_main_mcmc_noise,
    main_mcmc_regularizer,
    relocate_and_grow_main,
)
try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False

try:
    from fused_ssim import fused_ssim
    FUSED_SSIM_AVAILABLE = True
except:
    FUSED_SSIM_AVAILABLE = False

try:
    from diff_gaussian_rasterization import SparseGaussianAdam
    SPARSE_ADAM_AVAILABLE = True
except:
    SPARSE_ADAM_AVAILABLE = False


def _apply_pretrained_main_reset_factorial(
    pretrained_main_gaussians,
    *,
    expected_count,
    color_mode,
    opacity_mode,
    opacity_reset_value,
):
    """Apply one isolated color/opacity initialization arm.

    This helper deliberately operates only on the freshly loaded full-SH3
    ordinary-GS PLY.  It keeps geometry (XYZ, raw scaling, raw rotation)
    byte-for-byte unchanged and performs every appearance/opacity transform
    before ``training_setup`` creates optimizer slots.
    """
    if color_mode not in {"keep", "reset"}:
        raise ValueError(f"unsupported pretrained main color mode: {color_mode!r}")
    if opacity_mode not in {"keep", "reset"}:
        raise ValueError(
            f"unsupported pretrained main opacity mode: {opacity_mode!r}"
        )
    if opacity_mode == "reset":
        if opacity_reset_value is None or not (0.0 < float(opacity_reset_value) < 1.0):
            raise ValueError(
                "opacity reset mode requires pretrained_main_opacity_reset_value in (0, 1)"
            )
    elif opacity_reset_value is not None:
        raise ValueError(
            "pretrained_main_opacity_reset_value must be omitted when opacity mode is keep"
        )

    expected_shapes = {
        "xyz": (int(expected_count), 3),
        "features_dc": (int(expected_count), 1, 3),
        "features_rest": (int(expected_count), 15, 3),
        "opacity": (int(expected_count), 1),
        "scaling": (int(expected_count), 3),
        "rotation": (int(expected_count), 4),
    }
    tensor_names = tuple(expected_shapes)
    before = {}
    for tensor_name in tensor_names:
        tensor = getattr(pretrained_main_gaussians, f"_{tensor_name}")
        if tuple(tensor.shape) != expected_shapes[tensor_name]:
            raise ValueError(
                "pretrained main Gaussian tensor shape mismatch for "
                f"{tensor_name}: expected {expected_shapes[tensor_name]}, "
                f"got {tuple(tensor.shape)}"
            )
        if not bool(torch.isfinite(tensor.detach()).all().item()):
            raise ValueError(
                f"pretrained main Gaussian tensor contains non-finite values: {tensor_name}"
            )
        before[tensor_name] = tensor.detach().clone()

    if int(pretrained_main_gaussians.get_xyz.shape[0]) != int(expected_count):
        raise ValueError(
            "pretrained main Gaussian count mismatch: "
            f"expected {expected_count}, got {pretrained_main_gaussians.get_xyz.shape[0]}"
        )
    if int(pretrained_main_gaussians.max_sh_degree) != 3:
        raise ValueError("pretrained main initialization requires full SH degree 3")
    if int(pretrained_main_gaussians.active_sh_degree) != 3:
        raise ValueError("pretrained main Gaussian PLY did not load full SH3 coefficients")

    if color_mode == "reset":
        # SH coefficients of exactly zero render the neutral RGB=.5 baseline
        # after the renderer's C0 conversion.
        pretrained_main_gaussians._features_dc = torch.nn.Parameter(
            torch.zeros_like(before["features_dc"]).requires_grad_(True)
        )
        pretrained_main_gaussians._features_rest = torch.nn.Parameter(
            torch.zeros_like(before["features_rest"]).requires_grad_(True)
        )

    if opacity_mode == "reset":
        target = torch.full_like(before["opacity"], float(opacity_reset_value))
        reset_raw_opacity = inverse_sigmoid(target)
        pretrained_main_gaussians._opacity = torch.nn.Parameter(
            reset_raw_opacity.requires_grad_(True)
        )

    # Geometry must remain an exact row-preserving transfer from the source
    # PLY.  Keep arms also require exact identity for their untouched fields.
    for tensor_name in ("xyz", "scaling", "rotation"):
        if not torch.equal(
            getattr(pretrained_main_gaussians, f"_{tensor_name}").detach(),
            before[tensor_name],
        ):
            raise AssertionError(
                f"pretrained main geometry changed during reset factorial: {tensor_name}"
            )
    if color_mode == "keep":
        for tensor_name in ("features_dc", "features_rest"):
            if not torch.equal(
                getattr(pretrained_main_gaussians, f"_{tensor_name}").detach(),
                before[tensor_name],
            ):
                raise AssertionError(
                    f"pretrained main keep color changed: {tensor_name}"
                )
    else:
        if not bool((pretrained_main_gaussians._features_dc.detach() == 0).all().item()):
            raise AssertionError("pretrained main reset DC color is not exactly zero")
        if not bool((pretrained_main_gaussians._features_rest.detach() == 0).all().item()):
            raise AssertionError("pretrained main reset SH-rest color is not exactly zero")
    if opacity_mode == "keep":
        if not torch.equal(pretrained_main_gaussians._opacity.detach(), before["opacity"]):
            raise AssertionError("pretrained main keep opacity changed")
    else:
        expected_raw_opacity = inverse_sigmoid(
            torch.full_like(before["opacity"], float(opacity_reset_value))
        )
        if not torch.equal(
            pretrained_main_gaussians._opacity.detach(), expected_raw_opacity
        ):
            raise AssertionError("pretrained main reset opacity target is not exact")

    for tensor_name in tensor_names:
        tensor = getattr(pretrained_main_gaussians, f"_{tensor_name}")
        if tuple(tensor.shape) != expected_shapes[tensor_name]:
            raise AssertionError(
                f"pretrained main tensor shape changed after mode transform: {tensor_name}"
            )
        if not bool(torch.isfinite(tensor.detach()).all().item()):
            raise AssertionError(
                f"pretrained main tensor became non-finite after mode transform: {tensor_name}"
            )
    return {
        "modes": {"color": color_mode, "opacity": opacity_mode},
        "value": None
        if opacity_mode == "keep"
        else float(opacity_reset_value),
        "count": int(expected_count),
        "activeSH": int(pretrained_main_gaussians.active_sh_degree),
        "fresh_optimizer": True,
        "optimizer_resume": False,
    }


def _build_hybrid_structure_contract(
    *,
    iterations,
    hybrid_start_iteration,
    pretrained_main_start_iteration,
    hybrid_local_iterations,
    main_densify_until_global,
    ndf_densify_until_global,
    mesh_joint_start_global,
    seed_gaussians_interval,
    seed_gaussians_gen_num_per_view,
    seed_gaussians_start_iter,
):
    """Validate and materialize the multistart seeded global schedule.

    The pure-3DGS checkpoint boundary, optimizer/depth schedules, evaluation
    iterations, and all structure cutoffs are expressed in global iteration
    coordinates.  Local hybrid lengths are derived only for auditability.
    """
    values = {
        "iterations": int(iterations),
        "hybrid_start_iteration": int(hybrid_start_iteration),
        "pretrained_main_start_iteration": int(pretrained_main_start_iteration),
        "hybrid_local_iterations": int(hybrid_local_iterations),
        "main_densify_until_global": int(main_densify_until_global),
        "ndf_densify_until_global": int(ndf_densify_until_global),
        "mesh_joint_start_global": int(mesh_joint_start_global),
        "seed_gaussians_interval": int(seed_gaussians_interval),
        "seed_gaussians_gen_num_per_view": int(seed_gaussians_gen_num_per_view),
        "seed_gaussians_start_iter": int(seed_gaussians_start_iter),
    }
    if values["iterations"] != 30000:
        raise ValueError("multistart contract requires iterations=30000")
    if values["hybrid_start_iteration"] not in {6000, 9000, 12000}:
        raise ValueError("hybrid_start_iteration must be one of {6000, 9000, 12000}")
    if values["pretrained_main_start_iteration"] != values["hybrid_start_iteration"]:
        raise ValueError("pretrained_main_start_iteration must equal hybrid_start_iteration")
    if values["hybrid_local_iterations"] != (
        values["iterations"] - values["hybrid_start_iteration"]
    ):
        raise ValueError("hybrid_local_iterations must equal iterations-hybrid_start_iteration")
    if values["main_densify_until_global"] != 15000:
        raise ValueError("multistart contract requires main_densify_until_global=15000")
    if values["ndf_densify_until_global"] != 15000:
        raise ValueError("multistart contract requires ndf_densify_until_global=15000")
    if values["mesh_joint_start_global"] != 20000:
        raise ValueError("multistart contract requires mesh_joint_start_global=20000")
    if not (
        values["hybrid_start_iteration"]
        < values["main_densify_until_global"]
        <= values["mesh_joint_start_global"]
        <= values["iterations"]
    ):
        raise ValueError("global structure boundaries must be ordered start < densify < joint <= iterations")
    if values["seed_gaussians_interval"] != 1500:
        raise ValueError("multistart contract requires seed_gaussians_interval=1500")
    if values["seed_gaussians_gen_num_per_view"] != 300:
        raise ValueError("multistart contract requires seed_gaussians_gen_num_per_view=300")
    if values["seed_gaussians_start_iter"] > values["hybrid_start_iteration"]:
        raise ValueError("seed_gaussians_start_iter must be <= hybrid_start_iteration")

    global_start = values["hybrid_start_iteration"]
    main_global = values["main_densify_until_global"]
    ndf_global = values["ndf_densify_until_global"]
    joint_global = values["mesh_joint_start_global"]
    local = {
        "hybrid_start": 0,
        "main_densify_until_exclusive": main_global - global_start,
        "ndf_densify_until_exclusive": ndf_global - global_start,
        "mesh_joint_start": joint_global - global_start,
        "hybrid_end": values["hybrid_local_iterations"],
    }
    global_boundaries = {
        "hybrid_start": global_start,
        "main_densify_until_exclusive": main_global,
        "ndf_densify_until_exclusive": ndf_global,
        "mesh_joint_start": joint_global,
        "hybrid_end": values["iterations"],
    }
    expected_seed_iterations = [
        iteration
        for iteration in range(global_start + 1, main_global)
        if iteration % values["seed_gaussians_interval"] == 0
        and iteration >= values["seed_gaussians_start_iter"]
    ]
    return {
        "schema": "hybrid_structure_contract_v2",
        "mode": "multistart_seeded_global15k",
        "values": values,
        "local_boundaries": local,
        "global_boundaries": global_boundaries,
        "boundary_semantics": "cutoffs are exclusive; the last active 100-step round is cutoff-100",
        "global_lr_schedule": "unchanged",
        "global_depth_schedule": "unchanged",
        "main_after_cutoff": "train_and_cycle_prune_only",
        "ndf_after_cutoff": "train_and_cycle_opacity_cleanup_only",
        "seed_events": {
            "interval": values["seed_gaussians_interval"],
            "gen_num_per_view": values["seed_gaussians_gen_num_per_view"],
            "expected_iterations": expected_seed_iterations,
            "actual_iterations": [],
            "gating": "hybrid_active_and_main_structure_active_and_iteration_gt_start",
            "support_filter": "raw_depth_min_views_2",
        },
    }


def _write_hybrid_structure_contract(metrics_output_dir, contract):
    """Persist and print one machine-readable structure schedule record."""
    contract_path = os.path.join(metrics_output_dir, "hybrid_structure_contract.json")
    with open(contract_path, "w", encoding="utf-8") as contract_file:
        json.dump(contract, contract_file, indent=2, sort_keys=True)
        contract_file.write("\n")
    print(
        "[HybridStructureContract] "
        + json.dumps(contract, sort_keys=True, separators=(",", ":"))
    )
    print(f"[HybridStructureContract] path={contract_path}")
    return contract_path

def training(dataset, opt, pipe, testing_iterations, saving_iterations, checkpoint_iterations, checkpoint, debug_from, load_iteration, task_path, seedfix_args, hybrid_args, full_train_metrics=False):
    
    if not SPARSE_ADAM_AVAILABLE and opt.optimizer_type == "sparse_adam":
        sys.exit(f"Trying to use sparse adam but it is not installed, please install the correct rasterizer using pip install [3dgs_accel].")
    if hybrid_args.forbid_periodic_seeding and opt.seed_gaussians:
        raise ValueError("paired initialization comparison forbids periodic Gaussian/NDF seeding")
    standard_main_prune_only = bool(getattr(opt, "standard_main_prune_only", True))
    main_mcmc = bool(getattr(opt, "main_mcmc", False))
    mcmc_cap_max = int(getattr(opt, "mcmc_cap_max", -1))
    mcmc_opacity_reg = float(getattr(opt, "mcmc_opacity_reg", 0.0))
    mcmc_scale_reg = float(getattr(opt, "mcmc_scale_reg", 0.0))
    mcmc_noise_lr = float(getattr(opt, "mcmc_noise_lr", 0.0))
    mcmc_relocate_from_global = int(
        getattr(opt, "mcmc_relocate_from_global", 0)
    )
    mcmc_relocate_until_global = int(
        getattr(opt, "mcmc_relocate_until_global", 0)
    )
    mcmc_interval = int(getattr(opt, "mcmc_interval", 0))
    if main_mcmc:
        if not standard_main_prune_only:
            raise ValueError(
                "main_mcmc requires --standard_main_prune_only so visibility "
                "cycle/self-prunes cannot delete MCMC particles"
            )
        if not hybrid_args.pretrained_main_ply_path:
            raise ValueError("main_mcmc requires --pretrained_main_ply_path")
        if mcmc_cap_max <= 0:
            raise ValueError("main_mcmc requires --mcmc_cap_max > 0")
        if min(mcmc_opacity_reg, mcmc_scale_reg, mcmc_noise_lr) < 0.0:
            raise ValueError("MCMC regularizers and noise_lr must be non-negative")
        if mcmc_relocate_from_global < 0 or mcmc_relocate_until_global <= mcmc_relocate_from_global:
            raise ValueError("MCMC relocation global interval is invalid")
        if mcmc_interval <= 0:
            raise ValueError("main_mcmc requires --mcmc_interval > 0")
    disable_screen_size_prune = bool(
        getattr(opt, "disable_screen_size_prune", False)
    )
    deterministic_camera_schedule = bool(
        getattr(opt, "deterministic_camera_schedule", False)
    )
    freeze_ndf_training = bool(getattr(opt, "freeze_ndf_training", False))
    freeze_ndf_geometry = bool(getattr(opt, "freeze_ndf_geometry", False))
    detach_ndf_alpha_from_main = bool(
        getattr(opt, "detach_ndf_alpha_from_main", False)
    )
    ndf_lr_scale = float(getattr(opt, "ndf_lr_scale", 1.0))
    ndf_freeze_until_iter = int(getattr(opt, "ndf_freeze_until_iter", 0) or 0)
    hybrid_start_iteration = int(
        getattr(opt, "hybrid_start_iteration", 0) or 0
    )
    if hybrid_start_iteration < 0 or hybrid_start_iteration >= int(opt.iterations):
        raise ValueError(
            "hybrid_start_iteration must be >= 0 and < total iterations"
        )
    pretrained_main_start_iteration = int(
        getattr(hybrid_args, "pretrained_main_start_iteration", 0) or 0
    )
    if pretrained_main_start_iteration < 0 or pretrained_main_start_iteration >= int(opt.iterations):
        raise ValueError(
            "pretrained_main_start_iteration must be >= 0 and < total iterations"
        )
    structure_contract = _build_hybrid_structure_contract(
        iterations=int(opt.iterations),
        hybrid_start_iteration=hybrid_start_iteration,
        pretrained_main_start_iteration=pretrained_main_start_iteration,
        hybrid_local_iterations=int(hybrid_args.hybrid_local_iterations),
        main_densify_until_global=int(hybrid_args.main_densify_until_global),
        ndf_densify_until_global=int(hybrid_args.ndf_densify_until_global),
        mesh_joint_start_global=int(hybrid_args.mesh_joint_start_global),
        seed_gaussians_interval=int(opt.seed_gaussians_interval),
        seed_gaussians_gen_num_per_view=int(opt.seed_gaussians_gen_num_per_view),
        seed_gaussians_start_iter=int(opt.seed_gaussians_start_iter),
    )
    main_densify_until_iteration = int(
        structure_contract["global_boundaries"]["main_densify_until_exclusive"]
    )
    ndf_densify_until_iteration = int(
        structure_contract["global_boundaries"]["ndf_densify_until_exclusive"]
    )
    mesh_joint_start_iteration = int(
        structure_contract["global_boundaries"]["mesh_joint_start"]
    )
    if int(getattr(opt, "mesh_start_finetune_iteration", -1)) != mesh_joint_start_iteration:
        raise ValueError(
            "mesh_start_finetune_iteration must equal the contract global mesh joint boundary"
        )
    if main_mcmc:
        if mcmc_relocate_from_global < hybrid_start_iteration + 1:
            raise ValueError(
                "MCMC relocation must begin after the pretrained hybrid handoff"
            )
        if mcmc_relocate_until_global > main_densify_until_iteration:
            raise ValueError(
                "MCMC relocation must end no later than the main densification cutoff"
            )
        if mcmc_relocate_from_global % mcmc_interval != 0:
            raise ValueError(
                "MCMC relocation start must align with --mcmc_interval"
            )
    if int(getattr(opt, "densification_interval", -1)) != 100:
        raise ValueError("multistart contract requires densification_interval=100")
    if not bool(getattr(opt, "seed_gaussians", False)):
        raise ValueError("multistart contract requires --seed_gaussians")
    if bool(getattr(hybrid_args, "forbid_periodic_seeding", False)):
        raise ValueError("multistart contract forbids --forbid_periodic_seeding")
    hybrid_active = hybrid_start_iteration == 0
    if ndf_lr_scale <= 0.0:
        raise ValueError("ndf_lr_scale must be > 0")
    if ndf_freeze_until_iter < 0:
        raise ValueError("ndf_freeze_until_iter must be >= 0")
    if freeze_ndf_training and ndf_freeze_until_iter > 0:
        raise ValueError("freeze_ndf_training and ndf_freeze_until_iter are mutually exclusive")
    densify_split_mode = str(getattr(opt, "densify_split_mode", "random"))
    densify_split_max_offset_fraction = float(
        getattr(opt, "densify_split_max_offset_fraction", 0.05)
    )
    mesh_aware_split = bool(getattr(opt, "mesh_aware_split", False))
    mesh_aware_split_nearest_faces = int(
        getattr(opt, "mesh_aware_split_nearest_faces", 1)
    )
    if mesh_aware_split_nearest_faces not in {1, 2, 3}:
        raise ValueError("mesh_aware_split_nearest_faces must be one of {1, 2, 3}")
    if densify_split_mode not in {"random", "deterministic_principal"}:
        raise ValueError(f"unsupported densify_split_mode: {densify_split_mode!r}")
    if densify_split_max_offset_fraction <= 0.0:
        raise ValueError("densify_split_max_offset_fraction must be > 0")

    first_iter = 0
    tb_writer = prepare_output_and_logger(dataset)
    # 如果指定了 task_path，则将输出（尤其是 TensorBoard 日志）写到 model_path/task_path 子目录
    task_output_dir = None
    if task_path is not None:
        task_output_dir = os.path.join(dataset.model_path, task_path)
        os.makedirs(task_output_dir, exist_ok=True)
        if tb_writer:
            tb_writer.close()
            tb_writer = SummaryWriter(task_output_dir)
    metrics_output_dir = task_output_dir if task_output_dir is not None else dataset.model_path
    os.makedirs(metrics_output_dir, exist_ok=True)
    _write_hybrid_structure_contract(metrics_output_dir, structure_contract)

    def record_seed_event(iteration):
        """Record an invoked periodic seed round in the persisted contract."""
        actual_iterations = structure_contract["seed_events"]["actual_iterations"]
        iteration = int(iteration)
        if iteration not in actual_iterations:
            actual_iterations.append(iteration)
            _write_hybrid_structure_contract(metrics_output_dir, structure_contract)
        print(
            "[SeedSchedule] actual_event_iteration=%d expected=%s"
            % (iteration, structure_contract["seed_events"]["expected_iterations"])
        )

    print(
        "[SeedSchedule] expected_event_iterations="
        + json.dumps(structure_contract["seed_events"]["expected_iterations"])
    )
    metrics_output_path = os.path.join(metrics_output_dir, "eval_metrics.csv")
    densify_min_opacity = float(getattr(opt, "densify_min_opacity", 0.005))
    if densify_min_opacity < 0.0:
        raise ValueError("densify_min_opacity must be >= 0")
    densification_log_path = None
    if bool(getattr(opt, "log_densification_events", False)):
        densification_log_path = os.path.join(metrics_output_dir, "densification_events.csv")
        if os.path.exists(densification_log_path):
            raise FileExistsError(
                f"refusing to reuse densification event log: {densification_log_path}"
            )
        with open(densification_log_path, "w", encoding="utf-8") as f_densify:
            csv.writer(f_densify).writerow(
                [
                    "component",
                    "iteration",
                    "n_before",
                    "n_after",
                    "clone_parents",
                    "clone_created",
                    "split_parents",
                    "split_created",
                    "split_net_added",
                    "opacity_or_size_pruned",
                    "stats_json",
                ]
            )

    mcmc_event_log_path = None
    if main_mcmc:
        mcmc_event_log_path = os.path.join(
            metrics_output_dir, "main_mcmc_events.csv"
        )
        if os.path.exists(mcmc_event_log_path):
            raise FileExistsError(
                f"refusing to reuse MCMC event log: {mcmc_event_log_path}"
            )
        with open(mcmc_event_log_path, "w", encoding="utf-8", newline="") as f_mcmc:
            csv.writer(f_mcmc).writerow(
                [
                    "iteration",
                    "before",
                    "dead",
                    "relocated",
                    "added",
                    "after",
                    "cap",
                    "seed_over_cap",
                ]
            )

    def log_densification_event(component, iteration, stats):
        if densification_log_path is None:
            return
        # GaussianModel and NativeScalarSHGaussianField intentionally expose
        # slightly different names.  Normalize their existing return dicts to
        # one CSV schema without changing either implementation.
        n_before = int(stats.get("n_before", stats.get("before", 0)))
        n_after = int(stats.get("n_after", stats.get("after", 0)))
        clone_parents = int(stats.get("clone_parents", 0))
        clone_created = int(stats.get("clone_created", clone_parents))
        split_parents = int(stats.get("split_parents", 0))
        split_created = int(
            stats.get("split_created", 2 * split_parents)
        )
        split_net_added = int(
            stats.get("split_net_added", split_created - split_parents)
        )
        pruned = int(
            stats.get("opacity_or_size_pruned", stats.get("pruned", 0))
        )
        with open(densification_log_path, "a", encoding="utf-8") as f_densify:
            csv.writer(f_densify).writerow(
                [
                    component,
                    int(iteration),
                    n_before,
                    n_after,
                    clone_parents,
                    clone_created,
                    split_parents,
                    split_created,
                    split_net_added,
                    pruned,
                    json.dumps(stats, sort_keys=True, separators=(",", ":")),
                ]
            )

    def log_main_mcmc_event(iteration, stats):
        if mcmc_event_log_path is None:
            return
        with open(mcmc_event_log_path, "a", encoding="utf-8", newline="") as f_mcmc:
            csv.writer(f_mcmc).writerow(
                [
                    int(iteration),
                    int(stats["before"]),
                    int(stats["dead"]),
                    int(stats["relocated"]),
                    int(stats["added"]),
                    int(stats["after"]),
                    int(stats["cap"]),
                    int(stats["seed_over_cap"]),
                ]
            )
    # Keep every diagnostic produced by this isolated entry inside the
    # experiment-specific seedfix directory.  Never reuse the paper code's
    # shared ``outputs/debug`` paths across control runs.
    os.makedirs(seedfix_args.log_dir, exist_ok=True)
    # Overwrite per run to avoid mixing stale metrics from previous interrupted runs.
    with open(metrics_output_path, "w", encoding="utf-8") as f_metrics:
        f_metrics.write("iteration,split,l1,psnr,ssim,lpips\n")

    gaussians = GaussianModel(dataset.sh_degree, opt.optimizer_type)
    scene = Scene(
        dataset,
        gaussians,
        load_iteration=load_iteration,
        shuffle=not deterministic_camera_schedule,
    )
    gaussians.training_setup(opt)
    gaussians.densify_split_mode = densify_split_mode
    gaussians.densify_split_max_offset_fraction = densify_split_max_offset_fraction
    if checkpoint:
        (model_params, first_iter) = torch.load(checkpoint)
        gaussians.restore(model_params, opt)

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    # 溶解场相关模式开关：
    # - gaussolve_mode: 使用高斯场作为溶解场
    # - mesh_alpha_mode: 使用 mesh / 贴图 alpha 作为溶解场
    gaussolve_mode = bool(dataset.gaussolve_mode)
    mesh_alpha_mode = bool(dataset.mesh_alpha_mode)
    force_vertex_alpha = bool(dataset.force_vertex_alpha)
    mesh_alpha_from_scratch = bool(dataset.mesh_alpha_from_scratch)
    gaussolve_sup_enable = bool(opt.gaussolve_sup_enable)
    gaussolve_sup_weight = float(opt.gaussolve_sup_weight)
    gaussolve_sup_loss_scaling = bool(
        getattr(opt, "gaussolve_sup_loss_scaling", False)
    )
    gaussolve_sup_relerr_thr = float(opt.gaussolve_sup_relerr_thr)
    gaussolve_sup_relerr_temp = float(opt.gaussolve_sup_relerr_temp)
    gaussolve_sup_hard_relerr_thr = float(opt.gaussolve_sup_hard_relerr_thr)

    # 两者不能同时为 True
    if gaussolve_mode and mesh_alpha_mode:
        raise ValueError("gaussolve_mode 和 mesh_alpha_mode 不能同时为 True，请只开启其中一个。")

    # Initialize mesh renderer if mesh_path is provided
    mesh_renderer = None
    surface_split_query = None
    mesh_appearance = None
    appearance_optimizer = None
    appearance_base_lrs = []
    appearance_lr_scheduler = None
    if hasattr(dataset, 'mesh_path') and dataset.mesh_path and os.path.exists(dataset.mesh_path):
        texture_path = getattr(dataset, 'mesh_texture_path', None)
        if texture_path and not os.path.exists(texture_path):
            texture_path = None
        mesh_renderer = MeshRenderer(
            dataset.mesh_path,
            texture_path,
            device="cuda",
            use_sigmoid_rgb=opt.use_sigmoid_rgb,
            use_sigmoid_alpha=opt.use_sigmoid_alpha,
            mesh_alpha_mode=mesh_alpha_mode,
            # 当显式指定从 scratch 学 alpha 时，完全忽略 mesh / 纹理里自带的 alpha 通道
            alpha_from_scratch=mesh_alpha_from_scratch,
            force_vertex_alpha=force_vertex_alpha,
        )
        print(f"\nMesh renderer initialized with mesh: {dataset.mesh_path}")
        if mesh_aware_split:
            surface_split_query = MeshAwareSplitProjector(
                dataset.mesh_path,
                scene.cameras_extent,
                nearest_faces=mesh_aware_split_nearest_faces,
            )
            print(
                "[MeshAwareSplit] enabled for main GS only: "
                f"nearest_faces={surface_split_query.nearest_faces}, "
                "mode=minimal_same_side_projection, "
                f"tau={surface_split_query.tau:.6f}, mesh={dataset.mesh_path}"
            )
        if opt.mesh_train:
            mesh_renderer.training_setup(opt)
            mesh_renderer.unfreeze()
            print(f"\nMesh parameter training enabled. Current LR: {mesh_renderer.get_learning_rate():.6e}")
        else:
            mesh_renderer.freeze()

        for view in scene.getTrainCameras():
            view.open_gl_view, view.open_gl_mvp = get_opengl_matrices(view)
        for view in scene.getTestCameras():
            view.open_gl_view, view.open_gl_mvp = get_opengl_matrices(view)

        if hybrid_args.stage1_appearance_path:
            mesh_appearance = Stage1Appearance.load(
                hybrid_args.stage1_appearance_path, device="cuda"
            )
            if mesh_appearance.variant not in {"uv_sh", "texture"}:
                raise ValueError(
                    "paired downstream comparison expects a UV-SH or ordinary-texture Stage-1 sidecar, "
                    f"got {mesh_appearance.variant!r}"
                )
            mesh_appearance.eval()
            for parameter in mesh_appearance.parameters():
                parameter.requires_grad_(False)
            print(
                f"[Stage1] Loaded frozen {mesh_appearance.variant} appearance sidecar: "
                f"{hybrid_args.stage1_appearance_path}"
            )
    
    synchronous_render_streams = bool(
        getattr(opt, "synchronous_render_streams", False)
    )
    print(
        "[RenderStreams] mode="
        + ("synchronous_default" if synchronous_render_streams else "historical_auxiliary")
    )
    print(
        "[PruneSchedule] screen_size_prune="
        + ("disabled" if disable_screen_size_prune else "historical")
    )
    print(
        "[CameraSchedule] mode="
        + ("sequential_sorted" if deterministic_camera_schedule else "historical_random_pop")
    )
    print(
        "[NDFSchedule] mode="
        + ("frozen_initialized_field" if freeze_ndf_training else "trainable")
    )
    if freeze_ndf_geometry:
        print("[NDFSchedule] geometry=frozen")
    if detach_ndf_alpha_from_main:
        print("[NDFSchedule] main-image alpha gradient=detached")
    print(
        "[NDFSchedule] lr_scale=%.6g freeze_until=%d"
        % (ndf_lr_scale, ndf_freeze_until_iter)
    )
    print(
        "[HybridSchedule] start=%d phase=%s"
        % (hybrid_start_iteration, "hybrid" if hybrid_active else "pure_3dgs")
    )
    print(
        "[MainMCMC] mode=%s cap=%d relocate=[%d,%d) interval=%d "
        "regularizer_from=%d noise_from=%d "
        "periodic_main_opacity_reset=disabled "
        "post_cutoff_main_opacity_cleanup=disabled"
        % (
            "enabled" if main_mcmc else "disabled",
            mcmc_cap_max,
            mcmc_relocate_from_global,
            mcmc_relocate_until_global,
            mcmc_interval,
            hybrid_start_iteration + 1,
            hybrid_start_iteration + 1,
        )
    )
    meshgs_render = MeshGsRender(
        mesh_renderer,
        mesh_appearance,
        synchronous_streams=synchronous_render_streams,
    )
    # The frozen rebuttal training contract uses the historical raw depth
    # moment for seeding.  Do not construct the conditional-depth diagnostic
    # renderer on this path.  Conditional M/A remains available only through
    # the separate depth-rendering diagnostics.
    gaussolves = None
    if gaussolve_mode is True:
        if mesh_renderer is None:
            print("\nWarning: gaussolve_mode=True requires mesh renderer. Disabling gaussolve branch.")
            gaussolve_mode = False
        else:
            gaussolves = GaussianModel(dataset.gaussolve_sh_degree, opt.optimizer_type)
            gaussolves.train_test_exp = dataset.train_test_exp

    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)

    use_sparse_adam = opt.optimizer_type == "sparse_adam" and SPARSE_ADAM_AVAILABLE 
    depth_schedule_max_steps = int(
        getattr(opt, "depth_schedule_max_steps", 0) or 0
    )
    if depth_schedule_max_steps <= 0:
        depth_schedule_max_steps = int(opt.iterations)
    if depth_schedule_max_steps < int(opt.iterations):
        raise ValueError(
            "depth_schedule_max_steps must be >= iterations so a prefix run "
            "cannot silently accelerate the depth schedule"
        )
    print(
        "[DepthSchedule] init=%.6g final=%.6g max_steps=%d iterations=%d"
        % (
            float(opt.depth_l1_weight_init),
            float(opt.depth_l1_weight_final),
            depth_schedule_max_steps,
            int(opt.iterations),
        )
    )
    depth_l1_weight = get_expon_lr_func(
        opt.depth_l1_weight_init,
        opt.depth_l1_weight_final,
        max_steps=depth_schedule_max_steps,
    )

    ema_loss_for_log = 0.0
    ema_Ll1depth_for_log = 0.0

    # 创建两个 CUDA stream 用于并行执行
    stream1 = torch.cuda.Stream()  # 用于 mesh color 渲染
    stream2 = torch.cuda.Stream()  # 用于 gaussolve 渲染
    
    new_collection = False

    # loss 阈值
    train_cameras = scene.getTrainCameras()
    
    source_gaussians = gaussians

    mesh_pretrain_iterations = getattr(opt, "mesh_pretrain_iterations", 0)
    if mesh_appearance is not None and opt.mesh_train and mesh_pretrain_iterations > 0:
        for parameter in mesh_appearance.parameters():
            parameter.requires_grad_(True)
        appearance_groups = mesh_appearance.optimizer_groups()
        if appearance_groups:
            appearance_optimizer = torch.optim.Adam(appearance_groups, lr=0.0, eps=1e-8)
            print(
                f"[Stage1] {mesh_appearance.variant} auxiliary appearance participates "
                "in mesh pretraining optimizer."
            )
        else:
            # The ordinary-texture sidecar intentionally has no auxiliary
            # parameters.  Its RGB texture is owned by MeshRenderer's optimizer;
            # never construct Adam with an empty appearance parameter list.
            print("[Stage1] texture sidecar has no auxiliary appearance optimizer groups.")
    elif mesh_appearance is not None:
        # The Stage-1 checkpoint is already trained.  Keep both texture and
        # UV-SH frozen until the common joint-finetune boundary.
        mesh_renderer.freeze_all()
        if mesh_renderer.optimizer is not None:
            mesh_renderer.optimizer.state.clear()
            mesh_renderer.optimizer = None
        mesh_renderer.lr_scheduler = None

    if mesh_renderer is not None and opt.mesh_train and mesh_renderer.optimizer is not None and mesh_pretrain_iterations > 0:
        print(f"\nStarting mesh pretraining for {mesh_pretrain_iterations} iterations...")
        
        mesh_viewpoint_stack = train_cameras.copy()
        mesh_viewpoint_indices = list(range(len(mesh_viewpoint_stack)))
        mesh_pretrain_bar = tqdm(range(mesh_pretrain_iterations), desc="Mesh pretraining", dynamic_ncols=True)
        ema_mesh_loss_for_log = 0.0
        smoothed_cycle_loss = None
        cycle_idx = 0

        for mesh_iteration in range(1, mesh_pretrain_iterations + 1):
            if not mesh_viewpoint_stack:
                mesh_viewpoint_stack = train_cameras.copy()
                mesh_viewpoint_indices = list(range(len(mesh_viewpoint_stack)))
            mesh_rand_idx = randint(0, len(mesh_viewpoint_indices) - 1)
            viewpoint_cam = mesh_viewpoint_stack.pop(mesh_rand_idx)
            _ = mesh_viewpoint_indices.pop(mesh_rand_idx)

            bg = torch.rand((3), device="cuda") if opt.random_background else background
            pos, verts_view, rast, mask = mesh_renderer.rasterize(viewpoint_cam)
            image = meshgs_render.mesh_color_from_rast(
                viewpoint_cam,
                pos,
                rast,
                mask,
                bg_color=bg,
                enable_antialiasing=True,
            )

            if viewpoint_cam.mask is not None:
                alpha_mask = viewpoint_cam.mask.cuda()
                image *= alpha_mask

            gt_image = viewpoint_cam.original_image.cuda()
            image = image * mask
            gt_image = gt_image * mask
            Ll1 = l1_loss(image, gt_image)
            if FUSED_SSIM_AVAILABLE:
                ssim_value = fused_ssim(image.unsqueeze(0), gt_image.unsqueeze(0))
            else:
                ssim_value = ssim(image, gt_image)

            loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim_value)
            tv_w = 1e-5  # 建议从 1e-6 ~ 1e-4 试，噪声重就调大
            loss_tv = 0.0
            if mesh_renderer.texture is not None:
                # Apply activation based on parameterization mode
                if mesh_renderer.use_sigmoid_rgb:
                    tex01 = torch.sigmoid(mesh_renderer.texture)  # texture 是 logits 参数
                else:
                    tex01 = mesh_renderer._clamp_color(mesh_renderer.texture)  # texture 已经是 [0,1]
                # tex01: [H,W,3] -> [3,H,W]
                # loss_tv = tv_loss(tex01.permute(2, 0, 1))

            loss = loss + tv_w * loss_tv
            loss.backward()

            with torch.no_grad():
                # Record per-camera mesh pretrain loss for cycle-wise averaging.
                viewpoint_cam.mesh_pretrain_loss = float(loss.item())
                mesh_renderer.update_learning_rate(mesh_iteration)
                mesh_renderer.optimizer.step()
                mesh_renderer.optimizer.zero_grad(set_to_none=True)
                if appearance_optimizer is not None:
                    appearance_optimizer.step()
                    appearance_optimizer.zero_grad(set_to_none=True)
                # Keep Gaussian grads empty during mesh-only warmup.

                ema_mesh_loss_for_log = 0.4 * loss.item() + 0.6 * ema_mesh_loss_for_log
                postfix = {"Loss": f"{ema_mesh_loss_for_log:.7f}"}
                if len(mesh_viewpoint_stack) == 0:
                    cycle_idx += 1
                    cycle_losses = [
                        getattr(cam, "mesh_pretrain_loss", None)
                        for cam in train_cameras
                    ]
                    cycle_losses = [v for v in cycle_losses if v is not None]
                    if cycle_losses:
                        avg_cycle_loss = sum(cycle_losses) / len(cycle_losses)
                        if smoothed_cycle_loss is None:
                            smoothed_cycle_loss = avg_cycle_loss
                        else:
                            smoothed_cycle_loss = 0.9 * smoothed_cycle_loss + 0.1 * avg_cycle_loss
                        print(
                            f"\n[Mesh pretrain cycle {cycle_idx}] avg loss: {avg_cycle_loss:.7f}, "
                            f"smoothed: {smoothed_cycle_loss:.7f}"
                        )
                        postfix["CycleAvg"] = f"{avg_cycle_loss:.7f}"
                        postfix["CycleEMA"] = f"{smoothed_cycle_loss:.7f}"
                        if tb_writer:
                            tb_writer.add_scalar("mesh_pretrain/cycle_avg_loss", avg_cycle_loss, cycle_idx)
                            tb_writer.add_scalar("mesh_pretrain/cycle_ema_loss", smoothed_cycle_loss, cycle_idx)
                if mesh_iteration % 10 == 0:
                    mesh_pretrain_bar.set_postfix(postfix)
                    mesh_pretrain_bar.update(10)
                if tb_writer:
                    tb_writer.add_scalar("mesh_pretrain/loss", loss.item(), mesh_iteration)
                    tb_writer.add_scalar("mesh_pretrain/l1", Ll1.item(), mesh_iteration)

        mesh_pretrain_bar.close()
        # mesh 预训练结果：优先存到 task_output_dir，否则保持原行为
        mesh_pretrain_dir = task_output_dir if 'task_output_dir' in locals() and task_output_dir is not None else dataset.model_path
        os.makedirs(mesh_pretrain_dir, exist_ok=True)
        mesh_renderer.save_model(os.path.join(mesh_pretrain_dir, "mesh_pretrained.glb"))
        if mesh_appearance is not None:
            mesh_appearance.save(
                os.path.join(mesh_pretrain_dir, "mesh_pretrained.environment.pt"),
                extra={"stage": "mesh_pretrain", "iterations": mesh_pretrain_iterations},
            )
        print("Mesh pretraining complete.")

        # 预训练结束后，先冻结 mesh，释放优化器状态，节省中间阶段的显存和算力
        mesh_renderer.freeze_all()
        if mesh_renderer.optimizer is not None:
            mesh_renderer.optimizer.state.clear()
            mesh_renderer.optimizer = None
        mesh_renderer.lr_scheduler = None
        if mesh_appearance is not None:
            for parameter in mesh_appearance.parameters():
                parameter.requires_grad_(False)
        appearance_optimizer = None


    # One-shot init from repository gaussians:
    # multi-view union mask -> add once to a fresh train_gaussians.

    # Isolated initialization-only contract.  The source row identity is
    # validated against points3D.txt before this mask can be applied.  Existing
    # Gaussian scales/features/opacities/rotations are copied by source index;
    # no parameter is recomputed here.
    if not train_cameras:
        raise RuntimeError("Initialization filter requires at least one training camera")
    train_center = torch.stack(
        [camera.camera_center.detach().double().cpu() for camera in train_cameras], dim=0
    ).mean(dim=0).numpy()
    sparse_dir = os.path.join(dataset.source_path, "sparse", "0")
    initialization_mask_cpu, initialization_report = build_colmap_initialization_stability_mask(
        source_gaussians.get_xyz.detach().float().cpu().numpy(),
        sparse_dir,
        train_center=train_center,
        cameras_extent=scene.cameras_extent,
    )
    rejected_mask_cpu = ~initialization_mask_cpu
    source_scale_cpu = source_gaussians.get_scaling.detach().float().cpu().numpy()
    rejected_max_scale = source_scale_cpu[rejected_mask_cpu].max(axis=1) if rejected_mask_cpu.any() else None
    initialization_report["rejected_scale_max_gt_0_5"] = (
        int((rejected_max_scale > 0.5).sum()) if rejected_max_scale is not None else 0
    )
    initialization_report["rejected_scale_max_gt_10"] = (
        int((rejected_max_scale > 10.0).sum()) if rejected_max_scale is not None else 0
    )
    initialization_report["rejected_scale_max_gt_50"] = (
        int((rejected_max_scale > 50.0).sum()) if rejected_max_scale is not None else 0
    )
    initialization_report["rejected_scale_max_gt_100"] = (
        int((rejected_max_scale > 100.0).sum()) if rejected_max_scale is not None else 0
    )
    # Apply the same geometric rule to every source; the exact PLY/TXT row
    # contract is checked first and non-canonical counts are recorded directly.
    canonical_source_count = 47555
    allow_unlocked_source = bool(
        getattr(opt, "initfilter_allow_unlocked_source", True)
    )
    requested_source_count = int(
        getattr(opt, "initfilter_expected_source_count", -1)
    )
    actual_source_count = int(initialization_report["source_count"])
    if requested_source_count <= 0:
        requested_source_count = actual_source_count
    if requested_source_count > 0 and actual_source_count != requested_source_count:
        raise RuntimeError(
            f"InitFilter launcher expects {requested_source_count} source rows, got "
            f"{actual_source_count}"
        )

    canonical_contract = actual_source_count == canonical_source_count
    if not canonical_contract and not allow_unlocked_source:
        raise RuntimeError(
            "InitFilter source is not the canonical 47,555-row rebuttal model. "
            "Use --initfilter_allow_unlocked_source together with "
            "--initfilter_expected_source_count only for an explicitly pinned new dataset."
        )
    if not canonical_contract and requested_source_count <= 0:
        raise RuntimeError(
            "A non-canonical InitFilter source requires a positive, launcher-pinned "
            "--initfilter_expected_source_count."
        )

    if canonical_contract:
        expected_source_count = canonical_source_count
        expected_filtered_count = 46652
        expected_rejected_count = 903
        expected_outside_high_angle_retained = 3
        expected_scale_stats = {"gt_0_5": 903, "gt_10": 559, "gt_50": 292, "gt_100": 180}
        initialization_report["contract_mode"] = "canonical_saintanne_exact"
    else:
        expected_source_count = actual_source_count
        expected_filtered_count = int(initialization_report["stable_count"])
        expected_rejected_count = int(initialization_report["rejected_count"])
        expected_outside_high_angle_retained = int(
            initialization_report["outside_2r_high_angle_retained_count"]
        )
        expected_scale_stats = {
            "gt_0_5": initialization_report["rejected_scale_max_gt_0_5"],
            "gt_10": initialization_report["rejected_scale_max_gt_10"],
            "gt_50": initialization_report["rejected_scale_max_gt_50"],
            "gt_100": initialization_report["rejected_scale_max_gt_100"],
        }
        initialization_report["contract_mode"] = "explicit_new_source_same_rule"
        initialization_report["launcher_pinned_source_count"] = requested_source_count
    initialization_report["locked_expected_source_count"] = expected_source_count
    initialization_report["locked_expected_filtered_count"] = expected_filtered_count
    initialization_report["locked_expected_rejected_count"] = expected_rejected_count
    initialization_report["locked_expected_outside_2r_high_angle_retained_count"] = expected_outside_high_angle_retained
    initialization_report["locked_expected_rejected_scale_counts"] = expected_scale_stats
    if initialization_report["source_count"] != expected_source_count:
        raise RuntimeError(
            f"Locked InitFilter expects {expected_source_count} source rows, got "
            f"{initialization_report['source_count']}"
        )
    if initialization_report["stable_count"] != expected_filtered_count:
        raise RuntimeError(
            f"InitFilter expects {expected_filtered_count} retained rows "
            f"({expected_rejected_count} rejected), got "
            f"{initialization_report['stable_count']} retained and "
            f"{initialization_report['rejected_count']} rejected"
        )
    if initialization_report["rejected_count"] != expected_rejected_count:
        raise RuntimeError(
            f"Locked InitFilter expects {expected_rejected_count} rejected rows, got "
            f"{initialization_report['rejected_count']}"
        )
    if initialization_report["outside_2r_high_angle_retained_count"] != expected_outside_high_angle_retained:
        raise RuntimeError(
            f"InitFilter expects exactly {expected_outside_high_angle_retained} outside-2R "
            "points retained by angle>=1.5deg, got "
            f"{initialization_report['outside_2r_high_angle_retained_count']}"
        )
    actual_scale_stats = {
        "gt_0_5": initialization_report["rejected_scale_max_gt_0_5"],
        "gt_10": initialization_report["rejected_scale_max_gt_10"],
        "gt_50": initialization_report["rejected_scale_max_gt_50"],
        "gt_100": initialization_report["rejected_scale_max_gt_100"],
    }
    if actual_scale_stats != expected_scale_stats:
        raise RuntimeError(
            f"Locked InitFilter rejected-scale contract mismatch: expected {expected_scale_stats}, "
            f"got {actual_scale_stats}"
        )

    audit_root = task_output_dir if task_output_dir is not None else dataset.model_path
    filtered_ply_path = os.path.join(
        audit_root, "initialization_filter_audit", "filtered_source_model.ply"
    )
    initialization_report["filtered_source_model_ply"] = os.path.abspath(filtered_ply_path)
    mask_path, report_path = write_initialization_filter_audit(
        initialization_mask_cpu,
        initialization_report,
        os.path.join(audit_root, "initialization_filter_audit"),
    )
    print(
        "[InitFilter] locked rule: reject only distance>2R AND max angle<1.5deg; "
        f"kept={initialization_report['stable_count']}/{initialization_report['source_count']}, "
        f"rejected={initialization_report['rejected_count']}, mask={mask_path}, report={report_path}"
    )
    initialization_mask = torch.from_numpy(initialization_mask_cpu).to(
        device=source_gaussians.get_xyz.device, dtype=torch.bool
    )
    retained_xyz_before = source_gaussians._xyz.detach()[initialization_mask].clone()
    retained_raw_scaling_before = source_gaussians._scaling.detach()[initialization_mask].clone()

    # Apply the mask to source_gaussians before any source-GS one-shot render.
    # GaussianModel.prune_points slices the existing parameter tensors and
    # optimizer slots; it does not recompute scale or any other parameter.
    source_gaussians.prune_points(~initialization_mask)
    if source_gaussians.get_xyz.shape[0] != expected_filtered_count:
        raise RuntimeError(
            f"Post-prune source count mismatch: {source_gaussians.get_xyz.shape[0]} "
            f"!= {expected_filtered_count}"
        )
    if not torch.equal(source_gaussians._xyz.detach(), retained_xyz_before):
        raise RuntimeError("InitFilter changed retained XYZ values instead of pure row slicing")
    if not torch.equal(source_gaussians._scaling.detach(), retained_raw_scaling_before):
        raise RuntimeError("InitFilter changed retained raw scales instead of pure row slicing")
    source_gaussians.save_ply(filtered_ply_path)
    print(f"[InitFilter] filtered source model (logging only): {filtered_ply_path}")

    # A pretrained ordinary main field is deliberately independent from the
    # filtered COLMAP source model above.  Keep ``source_gaussians`` untouched
    # after its audit: source-points NDF initialization and periodic seeds use
    # that filtered population, not this trainable replacement.
    pretrained_main_active = bool(hybrid_args.pretrained_main_ply_path)
    new_gaussians = None
    if pretrained_main_active:
        pretrained_main_gaussians = GaussianModel(dataset.sh_degree, opt.optimizer_type)
        pretrained_main_gaussians.load_ply(
            hybrid_args.pretrained_main_ply_path, dataset.train_test_exp
        )
        pretrained_count = int(pretrained_main_gaussians.get_xyz.shape[0])
        if pretrained_count != hybrid_args.pretrained_main_expected_count:
            raise ValueError(
                "pretrained main Gaussian count mismatch: "
                f"expected {hybrid_args.pretrained_main_expected_count}, got {pretrained_count}"
            )
        if main_mcmc and mcmc_cap_max < pretrained_count:
            raise ValueError(
                "main_mcmc cap must be >= the pretrained main-GS handoff count: "
                f"cap={mcmc_cap_max}, initial={pretrained_count}"
            )
        if dataset.sh_degree != 3 or pretrained_main_gaussians.max_sh_degree != 3:
            raise ValueError("pretrained main initialization requires full SH degree 3")
        pretrained_ply = PlyData.read(hybrid_args.pretrained_main_ply_path)
        pretrained_property_names = {
            property_.name for property_ in pretrained_ply.elements[0].properties
        }
        expected_sh3_rest_names = {f"f_rest_{index}" for index in range(45)}
        actual_rest_names = {
            property_name
            for property_name in pretrained_property_names
            if property_name.startswith("f_rest_")
        }
        if actual_rest_names != expected_sh3_rest_names:
            raise ValueError(
                "pretrained main Gaussian PLY must contain exactly full SH3 "
                "f_rest_0 through f_rest_44 coefficients"
            )
        del pretrained_ply
        pretrained_main_gaussians.spatial_lr_scale = source_gaussians.spatial_lr_scale
        pretrained_main_contract = _apply_pretrained_main_reset_factorial(
            pretrained_main_gaussians,
            expected_count=pretrained_count,
            color_mode=getattr(hybrid_args, "pretrained_main_color_mode", None),
            opacity_mode=getattr(hybrid_args, "pretrained_main_opacity_mode", None),
            opacity_reset_value=getattr(
                hybrid_args, "pretrained_main_opacity_reset_value", None
            ),
        )
        pretrained_main_gaussians.densify_split_mode = densify_split_mode
        pretrained_main_gaussians.densify_split_max_offset_fraction = (
            densify_split_max_offset_fraction
        )
        pretrained_main_gaussians.training_setup(opt)
        gaussians = pretrained_main_gaussians
        if main_mcmc and not hasattr(gaussians, "tmp_radii"):
            # MCMC can hold the main field at its cap, so no main
            # densify_and_prune call necessarily initializes this transient
            # bookkeeping before the unchanged external seed schedule runs.
            gaussians.tmp_radii = None
        scene.gaussians = gaussians
        print(
            "[PretrainedMain] loaded independent ordinary main GS "
            f"({pretrained_count} points); modes="
            f"{pretrained_main_contract['modes']} value="
            f"{pretrained_main_contract['value']}"
        )
        print(
            "[PretrainedMainContract] "
            + json.dumps(pretrained_main_contract, sort_keys=True, separators=(",", ":"))
        )
        # This is a fresh optimizer over the full-state PLY; no optimizer
        # slots are restored from the ordinary 3DGS control run.  The selected
        # start boundary therefore makes the first loop iteration start+1 while preserving the global LR
        # schedule and the existing PLY/opacity initialization path.
        first_iter = pretrained_main_start_iteration
        print(
            "[PretrainedMain] fresh_optimizer_from_ply=true "
            f"start_iteration={pretrained_main_start_iteration}; "
            "optimizer_resume=false"
        )
    else:
        # One-shot init from repository gaussians: multi-view union mask ->
        # add once to a fresh train_gaussians.  Pretrained-main mode skips this
        # entire construction/payload/replacement path.
        train_gaussians = initialize_empty_train_gaussians_from_source(source_gaussians, opt)

    # Allow CLI override for mesh_depth_tolerance (front/back classification threshold).
    # Smaller -> stricter front/back (more aggressive dissolve); larger -> looser.
    _mdt = float(getattr(opt, "mesh_depth_tolerance", 0.05))
    mesh_depth_tolerance_fill = _mdt
    mesh_depth_tolerance_dissolve = _mdt
    mesh_depth_tolerance = _mdt
    # Model-specific NDF ablation.  The main Mesh/GS renderer and the global
    # depth loss continue to use ``mesh_depth``; only the scalar NDF renderer
    # may receive no depth image when this explicit flag is enabled.
    ndf_no_mesh_depth_cutoff = bool(
        getattr(opt, "ndf_no_mesh_depth_cutoff", False)
    )
    surface_thickness = 0.2
    error_threshold = 0.01
    error_threshold_densification = 0.02
    # gaussolve dissolve regularization
    gaussolve_reg_sparse = float(getattr(opt, "gaussolve_reg_sparse", 1e-3))
    gaussolve_reg_tv = float(getattr(opt, "gaussolve_reg_tv", 2e-4))
    gaussolve_reg_bin = float(getattr(opt, "gaussolve_reg_bin", 5e-4))
    gaussolve_reg_warmup_iters = max(1, int(getattr(opt, "gaussolve_reg_warmup_iters", 2000)))
    # Directional tolerance scheduler:
    # - delayed start
    # - cycle-avg photometric score only (depth is excluded from scheduling)
    # - one-step directional probe and reverse on degradation
    dynamic_mesh_depth_tolerance = mesh_depth_tolerance

    def _seed_ndf_render(*args, **kwargs):
        """Keep periodic-seeding NDF semantics identical to training."""
        if ndf_no_mesh_depth_cutoff:
            kwargs["mesh_depth"] = None
        return gaussolve_render(*args, **kwargs)

    def _activate_delayed_hybrid():
        """Enable mesh/NDF after the pure-3DGS warm-up boundary."""
        nonlocal hybrid_active, gaussians
        if hybrid_active:
            return
        if mesh_renderer is None:
            raise RuntimeError(
                "delayed hybrid activation requires a valid --mesh_path"
            )
        if not pretrained_main_active and gaussians.get_xyz.shape[0] > 0:
            delayed_clone = prune_gaussians_on_good_mesh_multiview(
                source_gaussians=gaussians,
                cameras=train_cameras,
                mesh_renderer=mesh_renderer,
                gs_render_fn=regular_render,
                pipe=pipe,
                background=background,
                random_background=opt.random_background,
                use_trained_exp=dataset.train_test_exp,
                separate_sh=SPARSE_ADAM_AVAILABLE,
                error_threshold=error_threshold,
                depth_epsilon=mesh_depth_tolerance_fill,
                surface_thickness=surface_thickness,
                min_views=1,
            )
            if delayed_clone["new_xyz"].shape[0] > 0:
                gaussians.densification_postfix(
                    delayed_clone["new_xyz"],
                    delayed_clone["new_features_dc"],
                    delayed_clone["new_features_rest"],
                    delayed_clone["new_opacities"],
                    delayed_clone["new_scaling"],
                    delayed_clone["new_rotation"],
                    delayed_clone["new_tmp_radii"],
                )
                print(
                    "[HybridSchedule] delayed one-shot clone complete: "
                    f"{delayed_clone['new_xyz'].shape[0]} points"
                )
            else:
                print("[HybridSchedule] delayed one-shot clone selected no points")
        hybrid_active = True
        print(
            "[HybridSchedule] hybrid components active at iteration "
            f"{hybrid_start_iteration + 1}"
        )

    if not pretrained_main_active and hybrid_start_iteration == 0:
        if mesh_renderer is not None and source_gaussians.get_xyz.shape[0] > 0:
            new_gaussians = prune_gaussians_on_good_mesh_multiview(
                source_gaussians=source_gaussians,
                cameras=train_cameras,
                mesh_renderer=mesh_renderer,
                gs_render_fn=regular_render,
                pipe=pipe,
                background=background,
                random_background=opt.random_background,
                use_trained_exp=dataset.train_test_exp,
                separate_sh=SPARSE_ADAM_AVAILABLE,
                error_threshold=error_threshold,
                depth_epsilon=mesh_depth_tolerance_fill,
                surface_thickness=surface_thickness,
                # clone_mode="both",
                min_views=1,
            )


            if new_gaussians["new_xyz"].shape[0] > 0:
                train_gaussians.densification_postfix(
                    new_gaussians["new_xyz"],
                    new_gaussians["new_features_dc"],
                    new_gaussians["new_features_rest"],
                    new_gaussians["new_opacities"],
                    new_gaussians["new_scaling"],
                    new_gaussians["new_rotation"],
                    new_gaussians["new_tmp_radii"],
                )
                gaussians = train_gaussians
                scene.gaussians = gaussians
                print(f"\nOne-shot clone complete. Unique cloned gaussians: {new_gaussians['new_xyz'].shape[0]}")
            else:
                print("\nOne-shot clone complete. No gaussians selected; keep source model.")

    if gaussolve_mode is True and gaussolves is not None:
        if hybrid_args.external_ndf_init_mode == "source_points":
            # Reuse the already-filtered COLMAP/input.ply XYZ distribution as
            # an independent NDF field.  Clone before handing the tensor to
            # the initializer so the main GS parameter storage is never
            # aliased or consumed by NDF initialization.
            source_points = source_gaussians.get_xyz.detach().clone()
            if (
                source_points.ndim != 2
                or source_points.shape[1] != 3
                or not torch.isfinite(source_points).all()
            ):
                raise ValueError("source_points NDF initialization has invalid source XYZ")
            gaussolves = NativeScalarSHGaussianField.from_gaussolve_vertices(
                source_points.to(device="cuda", dtype=torch.float32).contiguous(),
                sh_degree=dataset.gaussolve_sh_degree,
                initial_opacity=0.01,
                initial_value=0.05,
                metadata={"downstream_arm": "filtered_colmap_source_points"},
            )
            gaussolves.metadata["geometry_source"] = "filtered_colmap_input_ply"
            gaussolves.metadata["source_point_count"] = int(source_points.shape[0])
            print(
                "[NDF init] filtered COLMAP/input.ply source distribution: "
                f"{gaussolves.get_xyz.shape[0]} points (independent clone)"
            )
        elif hybrid_args.external_ndf_init_mode == "direct_points":
            direct_payload = torch.load(
                hybrid_args.external_ndf_init_path,
                map_location="cpu",
                weights_only=False,
            )
            if direct_payload.get("format") != "uv_loss_surface_samples":
                raise ValueError("direct_points expects a uv_loss_surface_samples PT artifact")
            direct_xyz = direct_payload.get("xyz")
            if (
                not torch.is_tensor(direct_xyz)
                or direct_xyz.ndim != 2
                or direct_xyz.shape[1] != 3
                or not torch.isfinite(direct_xyz).all()
            ):
                raise ValueError("direct UV-loss sample XYZ is invalid")
            gaussolves = NativeScalarSHGaussianField.from_gaussolve_vertices(
                direct_xyz.to(device="cuda", dtype=torch.float32).contiguous(),
                sh_degree=dataset.gaussolve_sh_degree,
                initial_opacity=0.01,
                initial_value=0.05,
                metadata={"downstream_arm": "direct_uv_loss_surface"},
            )
            gaussolves.metadata["geometry_source"] = "uv_loss_texel_surface_samples"
            print(
                "[NDF init] direct UV-loss surface distribution: "
                f"{gaussolves.get_xyz.shape[0]} points"
            )
        elif hybrid_args.external_ndf_init_mode == "guided_pt":
            gaussolves = NativeScalarSHGaussianField.load(
                hybrid_args.external_ndf_init_path, device="cuda"
            )
            if gaussolves.max_sh_degree != dataset.gaussolve_sh_degree:
                raise ValueError(
                    "guided NDF SH degree does not match --gaussolve_sh_degree: "
                    f"{gaussolves.max_sh_degree} != {dataset.gaussolve_sh_degree}"
                )
            print(
                "[NDF init] pretrained/reinitialized error-guided distribution: "
                f"{gaussolves.get_xyz.shape[0]} points"
            )
        else:
            bad_vertex_mask = collect_bad_mesh_vertices_multiview(
                mesh_renderer=mesh_renderer,
                cameras=train_cameras,
                source_gaussians=source_gaussians if load_iteration is not None else None,
                gs_render_fn=regular_render,
                pipe=pipe,
                background=background,
                random_background=opt.random_background,
                use_trained_exp=dataset.train_test_exp,
                separate_sh=SPARSE_ADAM_AVAILABLE,
                mesh_depth_tolerance_dissolve=mesh_depth_tolerance_dissolve,
                loss_threshold_dissolve=error_threshold,
                min_views=1,
            )
            gaussolves.gaussolve_init(mesh_renderer, vertex_mask=bad_vertex_mask)
        # Inherit spatial_lr_scale from main gaussians so gaussolves' xyz lr is non-zero.
        # (gaussolve_init never goes through create_from_pcd, default would leave it at 0.)
        gaussolves.spatial_lr_scale = gaussians.spatial_lr_scale
        gaussolves.train_test_exp = dataset.train_test_exp
        gaussolves.training_setup(opt)
        if abs(ndf_lr_scale - 1.0) > 1e-12:
            for group in gaussolves.optimizer.param_groups:
                group["lr"] *= ndf_lr_scale
            base_xyz_scheduler = gaussolves.xyz_scheduler_args
            gaussolves.xyz_scheduler_args = (
                lambda iteration, fn=base_xyz_scheduler, scale=ndf_lr_scale:
                scale * fn(iteration)
            )
            print(f"[NDF] optimizer learning rates scaled by {ndf_lr_scale:g}")
        if ndf_freeze_until_iter > 0:
            for parameter in (
                gaussolves._xyz,
                gaussolves._features_dc,
                gaussolves._features_rest,
                gaussolves._opacity,
                gaussolves._scaling,
                gaussolves._rotation,
            ):
                parameter.requires_grad_(False)
            print(
                f"[NDF] updates and densification frozen through iteration "
                f"{ndf_freeze_until_iter}"
            )
        if freeze_ndf_geometry and not freeze_ndf_training:
            for parameter in (
                gaussolves._xyz,
                gaussolves._scaling,
                gaussolves._rotation,
            ):
                parameter.requires_grad_(False)
            print("[NDF] geometry updates disabled; appearance/opacity remain trainable")
        if freeze_ndf_training:
            for parameter in (
                gaussolves._xyz,
                gaussolves._features_dc,
                gaussolves._features_rest,
                gaussolves._opacity,
                gaussolves._scaling,
                gaussolves._rotation,
            ):
                parameter.requires_grad_(False)
            gaussolves.optimizer = None
            print("[NDF] parameter updates disabled; static initialized field is rendered")
        gaussolves.densify_split_mode = densify_split_mode
        gaussolves.densify_split_max_offset_fraction = densify_split_max_offset_fraction

    # The optional one-shot source replacement above may construct a fresh
    # GaussianModel with the historical default split mode.  Re-apply the
    # requested mode to the final main field after NDF initialization and
    # immediately before entering the training loop.
    gaussians.densify_split_mode = densify_split_mode
    gaussians.densify_split_max_offset_fraction = densify_split_max_offset_fraction
    
    viewpoint_stack = scene.getTrainCameras().copy()
    viewpoint_indices = list(range(len(viewpoint_stack)))
    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress", dynamic_ncols=True)
    first_iter += 1
    for iteration in range(first_iter, opt.iterations + 1):
        if (
            not hybrid_active
            and iteration == hybrid_start_iteration + 1
        ):
            _activate_delayed_hybrid()
        if network_gui.conn == None:
            network_gui.try_connect()
        while network_gui.conn != None:
            try:
                net_image_bytes = None
                custom_cam, do_training, pipe.convert_SHs_python, pipe.compute_cov3D_python, keep_alive, scaling_modifer = network_gui.receive()
                if custom_cam != None:
                    if hybrid_active:
                        net_image = meshgs_render.render_wrapper(
                            custom_cam,
                            gaussians,
                            pipe,
                            background,
                            scaling_modifier=scaling_modifer,
                            use_trained_exp=dataset.train_test_exp,
                            separate_sh=SPARSE_ADAM_AVAILABLE,
                            mesh_depth_tolerance=dynamic_mesh_depth_tolerance,
                            gaussolves=gaussolves,
                            gaussolve_mode=gaussolve_mode,
                            ndf_no_mesh_depth_cutoff=ndf_no_mesh_depth_cutoff,
                        )["render"]
                    else:
                        net_image = regular_render(
                            custom_cam,
                            gaussians,
                            pipe,
                            background,
                            scaling_modifier=scaling_modifer,
                            separate_sh=SPARSE_ADAM_AVAILABLE,
                            use_trained_exp=dataset.train_test_exp,
                        )["render"]
                    net_image_bytes = memoryview((torch.clamp(net_image, min=0, max=1.0) * 255).byte().permute(1, 2, 0).contiguous().cpu().numpy())
                network_gui.send(net_image_bytes, dataset.source_path)
                if do_training and ((iteration < int(opt.iterations)) or not keep_alive):
                    break
            except Exception as e:
                network_gui.conn = None

        iter_start.record()

        if (
            hybrid_active
            and gaussolve_mode is True
            and ndf_freeze_until_iter > 0
            and iteration == ndf_freeze_until_iter + 1
        ):
            for parameter in (
                gaussolves._xyz,
                gaussolves._features_dc,
                gaussolves._features_rest,
                gaussolves._opacity,
                gaussolves._scaling,
                gaussolves._rotation,
            ):
                parameter.requires_grad_(True)
            print(f"[NDF] updates resumed at iteration {iteration}")

        main_xyz_lr = gaussians.update_learning_rate(iteration)
        ndf_updates_enabled = (
            hybrid_active
            and gaussolve_mode is True
            and not freeze_ndf_training
            and iteration > ndf_freeze_until_iter
        )
        if ndf_updates_enabled:
            gaussolves.update_learning_rate(iteration)
        if hybrid_active and mesh_renderer is not None and opt.mesh_train:
            if (
                mesh_appearance is not None
                and iteration >= int(opt.mesh_start_finetune_iteration)
            ):
                # A loaded Stage-1 sidecar starts a fresh joint optimization
                # phase.  Its texture schedule must use the number of real
                # joint steps, rather than the global training iteration.
                mesh_joint_step = max(
                    0, iteration - int(opt.mesh_start_finetune_iteration)
                )
                mesh_renderer.update_learning_rate(mesh_joint_step)
            else:
                # Keep the legacy (no-sidecar) mesh scheduler semantics.
                mesh_renderer.update_learning_rate(mesh_pretrain_iterations + iteration)
        if appearance_optimizer is not None and appearance_lr_scheduler is not None:
            appearance_factor = appearance_lr_scheduler(
                max(0, iteration - int(opt.mesh_start_finetune_iteration))
            )
            for group, base_lr in zip(
                appearance_optimizer.param_groups, appearance_base_lrs
            ):
                group["lr"] = base_lr * appearance_factor

        # Every 1000 its we increase the levels of SH up to a maximum degree
        if iteration % 1000 == 0:
            gaussians.oneupSHdegree()
            if gaussians.active_sh_degree < gaussians.max_sh_degree:
                print("--------------------------------")
                print("active_sh_degree: ", gaussians.active_sh_degree)
                print("max_sh_degree: ", gaussians.max_sh_degree)
                print("--------------------------------")
            if hybrid_active and gaussolve_mode is True:
                gaussolves.oneupSHdegree()
                if gaussolves.active_sh_degree < gaussolves.max_sh_degree:
                    print("gaussolve_active_sh_degree: ", gaussolves.active_sh_degree)
                    print("gaussolve_max_sh_degree: ", gaussolves.max_sh_degree)
                    print("--------------------------------")

        # Pick a random Camera
        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()
            viewpoint_indices = list(range(len(viewpoint_stack)))
            new_collection = True
        rand_idx = 0 if deterministic_camera_schedule else randint(
            0, len(viewpoint_indices) - 1
        )
        viewpoint_cam = viewpoint_stack.pop(rand_idx)
        vind = viewpoint_indices.pop(rand_idx)

        # Render
        if (iteration - 1) == debug_from:
            pipe.debug = True

        bg = torch.rand((3), device="cuda") if opt.random_background else background

        # Step 1: Mesh rasterization and depth rendering (if mesh_renderer is available)
        if hybrid_active and mesh_renderer is not None:
            # Rasterize mesh once
            pos, verts_view, rast, mask = mesh_renderer.rasterize(viewpoint_cam)
            depth_linear, depth_normalized = mesh_renderer.depth_from_rast(viewpoint_cam, verts_view, rast, mask, normalize=False)
            
            # Compute mesh depth
            mesh_depth = depth_linear # [1, H, W]

            # Step 2.1: Gaussian rendering with mesh depth (for CUDA-level fusion).
            # hardz mode here means "soft kernel but no NDF": gaussolve_mode is forced
            # off in main(), so the alpha_eff=1 path makes the fusion image = Cf + Tf*Cm.
            # We deliberately keep the soft rasterizer so back-of-mesh Gaussians still
            # attenuate T_front and stay alive (true mesh_gs_hard collapses them to 0
            # opacity and prunes everything within ~3000 iters).
            render_pkg = meshgs_render.render(
                viewpoint_cam, gaussians, pipe, bg,
                use_trained_exp=dataset.train_test_exp,
                separate_sh=SPARSE_ADAM_AVAILABLE,
                mesh_depth=mesh_depth,
                output_inverse_depth=True,
                mesh_depth_tolerance=dynamic_mesh_depth_tolerance,
                )

            if synchronous_render_streams:
                mesh_rendered = meshgs_render.mesh_color_from_rast(
                    viewpoint_cam,
                    pos,
                    rast,
                    mask,
                    bg_color=bg,
                    enable_antialiasing=True,
                )
                if hybrid_active and gaussolve_mode is True:
                    gaussolve_pkg = gaussolve_render(
                        viewpoint_cam, gaussolves, pipe, torch.zeros_like(bg),
                        use_trained_exp=dataset.train_test_exp,
                        separate_sh=SPARSE_ADAM_AVAILABLE,
                        mesh_depth=(
                            None if ndf_no_mesh_depth_cutoff else mesh_depth
                        ),
                        mesh_depth_tolerance=dynamic_mesh_depth_tolerance,
                    )
            else:
                with torch.cuda.stream(stream1):
                    mesh_rendered = meshgs_render.mesh_color_from_rast(
                        viewpoint_cam,
                        pos,
                        rast,
                        mask,
                        bg_color=bg,
                        enable_antialiasing=True,
                    )
                if hybrid_active and gaussolve_mode is True:
                    with torch.cuda.stream(stream2):
                        gaussolve_pkg = gaussolve_render(
                            viewpoint_cam, gaussolves, pipe, torch.zeros_like(bg),
                            use_trained_exp=dataset.train_test_exp,
                            separate_sh=SPARSE_ADAM_AVAILABLE,
                            mesh_depth=(
                                None if ndf_no_mesh_depth_cutoff else mesh_depth
                            ),
                            mesh_depth_tolerance=dynamic_mesh_depth_tolerance,
                        )

            torch.cuda.synchronize()

            # Three-layer blending formula: C_final = alpha_eff * (C_f + T_f * C_m) + (1.0 - alpha_eff) * C_g
            # Get components
            Cf = render_pkg["render"]  # C_f: front-facing GS [3, H, W]
            Cg = render_pkg["render_all"]  # C_g: total GS [3, H, W]
            Cm = mesh_rendered  # C_m: mesh color [3, H, W]
            Tf = render_pkg["alpha"]  # T_f: transmittance of front-facing GS [1, H, W]
            
            # Get alpha_eff from dissolve field:
            # 1) gaussolve 模式：使用 gaussolve_pkg 渲染出的单通道图作为溶解场
            # 2) mesh alpha 模式：使用 mesh 的 alpha map 作为溶解场
            # 3) 默认：不使用溶解场，相当于 alpha_eff=1
            if hybrid_active and gaussolve_mode is True:
                dissolve_alpha = gaussolve_pkg["render"][:1].clamp(0, 1)  # [1, H, W]
                alpha_for_main = (
                    dissolve_alpha.detach()
                    if detach_ndf_alpha_from_main
                    else dissolve_alpha
                )
                alpha_eff = 1.0 - alpha_for_main  # alpha_eff: 1 = keep mesh, 0 = dissolve
            elif mesh_alpha_mode:
                # 使用 mesh 的 alpha map 作为溶解场
                alpha_eff = mesh_renderer.get_alpha_map(viewpoint_cam, pos, rast, mask)  # [1, H, W]
            else:
                # 默认：没有溶解场，始终保留 mesh
                alpha_eff = torch.ones_like(Tf)
            image = alpha_eff * (Cf + Tf * Cm) + (1.0 - alpha_eff) * Cg
        else:
            render_pkg = regular_render(
                viewpoint_cam, gaussians, pipe, bg, 
                use_trained_exp=dataset.train_test_exp, 
                separate_sh=SPARSE_ADAM_AVAILABLE
            )
            image = render_pkg["render"]
        # import torchvision
            depth = render_pkg["depth"]
        radii = render_pkg["radii"]
        visibility_filter = render_pkg["visibility_filter"]
        viewspace_point_tensor = render_pkg["viewspace_points"]
        gaussolve_radii = None
        gaussolve_visibility_filter = None
        gaussolve_viewspace_points = None
        if hybrid_active and gaussolve_mode is True and mesh_renderer is not None:
            gaussolve_radii = gaussolve_pkg["radii"]
            gaussolve_visibility_filter = gaussolve_pkg["visibility_filter"]
            gaussolve_viewspace_points = gaussolve_pkg["viewspace_points"]

        if viewpoint_cam.mask is not None:
            alpha_mask = viewpoint_cam.mask.cuda()
            image *= alpha_mask

        # Loss


        gt_image = viewpoint_cam.original_image.cuda()
        Ll1 = l1_loss(image, gt_image)
        if FUSED_SSIM_AVAILABLE:
            ssim_value = fused_ssim(image.unsqueeze(0), gt_image.unsqueeze(0))
        else:
            ssim_value = ssim(image, gt_image)

        loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim_value)
        gaussolve_reg_value = 0.0
        gaussolve_sup_value = 0.0
        if hybrid_active and gaussolve_mode is True and gaussolve_pkg is not None:
            dissolve_alpha_reg = gaussolve_pkg["render"][:1].clamp(0, 1)
            reg_sparse = dissolve_alpha_reg.mean()
            reg_tv = (dissolve_alpha_reg[:, :, 1:] - dissolve_alpha_reg[:, :, :-1]).abs().mean() + \
                     (dissolve_alpha_reg[:, 1:, :] - dissolve_alpha_reg[:, :-1, :]).abs().mean()
            reg_bin = (dissolve_alpha_reg * (1.0 - dissolve_alpha_reg)).mean()
            warmup_scale = min(1.0, float(iteration) / float(gaussolve_reg_warmup_iters))
            gaussolve_reg = warmup_scale * (
                gaussolve_reg_sparse * reg_sparse +
                gaussolve_reg_tv * reg_tv +
                gaussolve_reg_bin * reg_bin
            )
            loss = loss + gaussolve_reg
            gaussolve_reg_value = float(gaussolve_reg.item())
        
        # Regularization: prevent mesh from being fully dissolved
        # if mesh_renderer is not None and hasattr(mesh_renderer, 'alpha'):
        #     lambda_reg = getattr(opt, 'dissolve_reg_weight', 0.01)
        #     # Get current alpha_eff for regularization
        #     if 'alpha_eff' in locals():
        #         loss_reg = lambda_reg * torch.mean(1.0 - alpha_eff)  # Encourage alpha_eff to be high (keep mesh)
        #         loss = loss + loss_reg

        # if mesh_renderer is not None:
        #     diff_map = torch.abs(mesh_rendered - gt_image).mean(dim=0, keepdim=True).detach()
        #     threshold = 0.05
        #     sparsity_weight = torch.clamp(1.0 - (diff_map / threshold), min=0.0, max=1.0)
        #     lambda_sparsity = 100
        #     loss_adaptive_sparsity = (alpha * sparsity_weight).mean()
        #     loss += lambda_sparsity * loss_adaptive_sparsity

        # Depth regularization (inverse depth): compare fused depth with monocular depth.
        Ll1depth_pure = 0.0
        Ll1depth = 0.0
        if hybrid_active and depth_l1_weight(iteration) > 0 and viewpoint_cam.depth_reliable:
            mono_invdepth = viewpoint_cam.invdepthmap.cuda()
            depth_mask = viewpoint_cam.depth_mask.cuda()

            invDepth = render_pkg["depth"]  # GS inverse depth
            
            if mesh_renderer is not None:
                mesh_mask = mask.float()
                if mesh_mask.dim() == 4:
                    mesh_mask = mesh_mask.squeeze(-1)
                mesh_invdepth = (1.0 / (mesh_depth + 1e-6)) * mesh_mask
                # Use alpha_eff if available, otherwise use default blending
                if 'alpha_eff' in locals():
                    invDepth_total = render_pkg["depth_total"]
                    fused_invdepth = alpha_eff * (invDepth + Tf * mesh_invdepth) + (1.0 - alpha_eff) * invDepth_total
                else:
                    fused_invdepth = invDepth + Tf * mesh_invdepth
            else:
                fused_invdepth = invDepth
            Ll1depth_pure = torch.abs((fused_invdepth - mono_invdepth) * depth_mask).mean()
            Ll1depth = depth_l1_weight(iteration) * Ll1depth_pure
            loss += Ll1depth
            Ll1depth = Ll1depth.item()
            if gaussolve_mode is True and gaussolve_pkg is not None:
                if gaussolve_sup_enable and gaussolve_sup_weight > 0.0 and mesh_renderer is not None and viewpoint_cam.depth_reliable:
                    depth_mask_sup = viewpoint_cam.depth_mask.cuda()
                    mono_invdepth_sup = viewpoint_cam.invdepthmap.cuda()
                    mesh_mask_sup = mask.float()
                    if mesh_mask_sup.dim() == 4:
                        mesh_mask_sup = mesh_mask_sup.squeeze(-1)

                    # Use mesh linear depth directly; convert mono inverse depth to linear depth.
                    mono_depth_sup = 1.0 / torch.clamp(mono_invdepth_sup, min=1e-6)
                    valid_sup = (depth_mask_sup * mesh_mask_sup * (mono_invdepth_sup > 1e-6).float()).clamp(0.0, 1.0)

                    # Per-view median depth from predicted depth to normalize scale across scenes.
                    valid_mask_bool = valid_sup > 0.5
                    if valid_mask_bool.any():
                        depth_median = torch.median(mono_depth_sup[valid_mask_bool]).detach().clamp_min(1e-6)
                    else:
                        depth_median = torch.tensor(1.0, device=mono_depth_sup.device, dtype=mono_depth_sup.dtype)

                    depth_ratio_err = torch.abs(mesh_depth - mono_depth_sup) / depth_median
                    # Sigmoid maps continuous ratio error to smooth dissolve target; hard threshold enforces full dissolve.
                    target_soft = torch.sigmoid(
                        (depth_ratio_err - gaussolve_sup_relerr_thr) /
                        max(gaussolve_sup_relerr_temp, 1e-6)
                    )
                    target_hard = (depth_ratio_err >= gaussolve_sup_hard_relerr_thr).float()
                    target_dissolve = torch.maximum(target_soft, target_hard)
                    denom = valid_sup.sum().clamp_min(1.0)
                    # One-sided ReLU supervision: only encourage higher dissolve alpha in bad-depth regions.
                    # Restored C-state: residual scaled by current loss as natural early-strong/late-weak curriculum.
                    sup_residual = torch.relu(target_dissolve - dissolve_alpha_reg)
                    if gaussolve_sup_loss_scaling:
                        sup_residual = sup_residual * loss.detach().clone()
                    gaussolve_sup_loss = (sup_residual * valid_sup).sum() / denom
                    loss = loss + gaussolve_sup_weight * gaussolve_sup_loss
                    gaussolve_sup_value = float(gaussolve_sup_loss.item())

        # Keep this main-only term after Gaussolve supervision has consumed the
        # detached photometric loss.  Adding it earlier would change the NDF
        # supervision scaling and therefore invalidate the frozen hybrid loss
        # contract.
        mcmc_reg_value = 0.0
        mcmc_reg_active = (
            main_mcmc
            and hybrid_active
            and iteration >= hybrid_start_iteration + 1
            and iteration <= int(opt.iterations)
        )
        if mcmc_reg_active:
            mcmc_reg = main_mcmc_regularizer(
                gaussians,
                opacity_reg=mcmc_opacity_reg,
                scale_reg=mcmc_scale_reg,
            )
            loss = loss + mcmc_reg
            mcmc_reg_value = float(mcmc_reg.detach().item())
        loss.backward()
        iter_end.record()

        with torch.no_grad():
            # Progress bar
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            ema_Ll1depth_for_log = 0.4 * Ll1depth + 0.6 * ema_Ll1depth_for_log

            if iteration % 10 == 0:
                progress_bar.set_postfix({
                    "Loss": f"{ema_loss_for_log:.{7}f}",
                    "Depth Loss": f"{ema_Ll1depth_for_log:.{7}f}",
                    "GaussolveReg": f"{gaussolve_reg_value:.{7}f}"
                })
                progress_bar.update(10)
            if iteration == opt.iterations:
                progress_bar.close()
            if tb_writer:
                tb_writer.add_scalar("train_loss_patches/gaussolve_reg", gaussolve_reg_value, iteration)
                tb_writer.add_scalar("train_loss_patches/main_mcmc_reg", mcmc_reg_value, iteration)

            # Log and save.  The warm-up report must use the ordinary renderer;
            # otherwise a hidden mesh/NDF render would contaminate the
            # pure-3DGS warm-up phase.
            if hybrid_active:
                report_render_func = meshgs_render.render_wrapper
                report_render_args = (
                    pipe,
                    background,
                    1.0,
                    SPARSE_ADAM_AVAILABLE,
                    None,
                    dataset.train_test_exp,
                    dynamic_mesh_depth_tolerance,
                    gaussolves,
                    gaussolve_mode,
                    ndf_no_mesh_depth_cutoff,
                )
            else:
                report_render_func = regular_render
                report_render_args = (
                    pipe,
                    background,
                    1.0,
                    SPARSE_ADAM_AVAILABLE,
                    None,
                    dataset.train_test_exp,
                )
            final_asset_dir = task_output_dir if task_output_dir is not None else dataset.model_path
            exact_state_dir = (
                os.path.join(final_asset_dir, "exact_render_state")
                if iteration == int(opt.iterations)
                else ""
            )
            exact_test_view_records = training_report(
                tb_writer,
                iteration,
                Ll1,
                loss,
                l1_loss,
                iter_start.elapsed_time(iter_end),
                testing_iterations,
                scene,
                report_render_func,
                report_render_args,
                dataset.train_test_exp,
                metrics_output_path,
                full_train_metrics=full_train_metrics,
                exact_report_dir=exact_state_dir,
                exact_report_iteration=int(opt.iterations),
            )
            if iteration == int(opt.iterations):
                save_exact_bundle(
                    exact_state_dir,
                    iteration=iteration,
                    dataset=dataset,
                    opt=opt,
                    pipe=pipe,
                    scene=scene,
                    report_render_func=report_render_func,
                    report_render_args=report_render_args,
                    gaussians=gaussians,
                    mesh_renderer=mesh_renderer,
                    mesh_appearance=mesh_appearance,
                    gaussolves=gaussolves,
                    test_view_records=exact_test_view_records,
                )
            if (iteration in saving_iterations):
                print("\n[ITER {}] Saving Gaussians".format(iteration))
                scene.save(iteration, task_path)
                if mesh_appearance is not None:
                    appearance_dir = (
                        task_output_dir
                        if task_output_dir is not None
                        else dataset.model_path
                    )
                    mesh_appearance.save(
                        os.path.join(
                            appearance_dir,
                            f"mesh_appearance_iteration_{iteration}.pt",
                        ),
                        extra={
                            "iteration": int(iteration),
                            "joint_finetune": bool(
                                iteration >= int(opt.mesh_start_finetune_iteration)
                            ),
                        },
                    )
    
            # Main and NDF structure schedules are deliberately independent.
            # ``main_structure_active`` controls only main clone/split and its
            # statistics; ``ndf_structure_active`` may remain true through
            # global 25.9k for the extended NDF arm.  Both populations keep
            # training and the common cycle cleanup after their cutoff.
            main_structure_active = (
                hybrid_active and iteration < main_densify_until_iteration
            )
            ndf_structure_active = (
                ndf_updates_enabled
                and dataset.gaussolve_densification is True
                and gaussolve_radii is not None
                and iteration < ndf_densify_until_iteration
            )
            if main_structure_active:
                gaussians.max_radii2D[visibility_filter] = torch.max(
                    gaussians.max_radii2D[visibility_filter], radii[visibility_filter]
                )
                gaussians.add_densification_stats(
                    viewspace_point_tensor, visibility_filter
                )
            if ndf_structure_active:
                gaussolves.max_radii2D[gaussolve_visibility_filter] = torch.max(
                    gaussolves.max_radii2D[gaussolve_visibility_filter],
                    gaussolve_radii[gaussolve_visibility_filter],
                )
                gaussolves.add_densification_stats(
                    gaussolve_viewspace_points, gaussolve_visibility_filter
                )

            # Shared camera-cycle cleanup must run after this iteration's
            # statistics have been collected but before any population changes
            # its tensor shapes through densify/seed/reset.  The active/cutoff
            # split preserves the historical semantics: active main/NDF fields
            # do not receive the cycle opacity prune, while each inactive field
            # does.  ``new_collection`` is consumed exactly once for both.
            if hybrid_active and mesh_renderer is not None:
                visibility_filter = render_pkg["gs_visibility_under_mesh"] == 1.0
                gaussians.visibility[visibility_filter] = 1.0
                if new_collection:
                    if not main_structure_active and not main_mcmc:
                        main_opacity_prune = (gaussians.get_opacity < 0.005).squeeze()
                        if not standard_main_prune_only:
                            main_visibility_prune = gaussians.visibility == 0
                            main_prune = torch.logical_or(
                                main_visibility_prune, main_opacity_prune
                            )
                        else:
                            main_prune = main_opacity_prune
                        gaussians.prune_points(main_prune)
                    elif not standard_main_prune_only:
                        # During the active main structure phase, retain only
                        # the legacy visibility component; standard_main_prune_only
                        # disables that component without adding opacity pruning.
                        gaussians.prune_points(gaussians.visibility == 0)

                    if (
                        not ndf_structure_active
                        and gaussolve_mode is True
                        and gaussolves is not None
                    ):
                        ndf_opacity_prune = (gaussolves.get_opacity < 0.005).squeeze()
                        gaussolves.prune_points(ndf_opacity_prune)

                    gaussians.visibility = torch.zeros_like(gaussians.visibility)
                    new_collection = False

            if (
                (main_structure_active or ndf_structure_active)
                and iteration > opt.densify_from_iter
                and iteration % opt.densification_interval == 0
            ):
                size_threshold = (
                    None
                    if disable_screen_size_prune
                    else (20 if iteration > opt.opacity_reset_interval else None)
                )
                if main_structure_active and not main_mcmc:
                    main_densification_stats = gaussians.densify_and_prune(
                        opt.densify_grad_threshold,
                        densify_min_opacity,
                        scene.cameras_extent,
                        size_threshold,
                        radii,
                        surface_split_query=surface_split_query,
                    )
                    log_densification_event(
                        "main", iteration, main_densification_stats
                    )
                if ndf_structure_active:
                    ndf_densification_stats = gaussolves.densify_and_prune(
                        opt.densify_grad_threshold,
                        densify_min_opacity,
                        scene.cameras_extent,
                        size_threshold,
                        gaussolve_radii,
                    )
                    log_densification_event(
                        "ndf", iteration, ndf_densification_stats
                    )
                
            if main_structure_active or ndf_structure_active:
                _seed_interval = int(getattr(opt, "seed_gaussians_interval", 1500))
                _seed_start = int(getattr(opt, "seed_gaussians_start_iter", 6000))
                if (
                    hybrid_active
                    and main_structure_active
                    and mesh_renderer is not None
                    and opt.seed_gaussians
                    and iteration > hybrid_start_iteration
                    and iteration < main_densify_until_iteration
                    and iteration % _seed_interval == 0
                    and iteration >= _seed_start
                ):
                    # The checkpoint boundary itself is never a seed event:
                    # the first event is the next interval-aligned global step.
                    record_seed_event(iteration)
                    seed_gaussians_cameras = scene.getTrainCameras().copy()
                    new_gaussians = seed_gaussians_for_patching_multiview(
                        gaussians,
                        seed_gaussians_cameras,
                        mesh_renderer,
                        background,
                        random_background=opt.random_background,
                        error_threshold=error_threshold_densification,
                        voxel_size_gaussolves=surface_thickness,
                        voxel_size_gaussians=surface_thickness/2,
                        gen_num_per_view=opt.seed_gaussians_gen_num_per_view,
                        gs_render_fn=meshgs_render.render,
                        gaussolve_render_fn=_seed_ndf_render,
                        pipe=pipe,
                        use_trained_exp=dataset.train_test_exp,
                        separate_sh=SPARSE_ADAM_AVAILABLE,
                        mesh_depth_tolerance=dynamic_mesh_depth_tolerance,
                        gaussolves=gaussolves,
                        gaussolve_mode=gaussolve_mode,
                        mesh_alpha=False,
                        min_views=seedfix_args.min_views,
                        min_opacity_mass=seedfix_args.min_opacity_mass,
                        seed_depth_mode=seedfix_args.depth_mode,
                        seed_depth_render_fn=None,
                        ndf_no_mesh_depth_cutoff=ndf_no_mesh_depth_cutoff,
                        diagnostic_log_path=os.path.join(
                            seedfix_args.log_dir,
                            f"seedfix_iter_{iteration:06d}.json",
                        ),
                    )
                    if new_gaussians is None:
                        print(
                            f"[Seedfix] No seed package returned at iteration {iteration}; "
                            "training iteration continues."
                        )
                    # save new gaussians to ply
                    if new_gaussians is not None and new_gaussians["new_xyz"].shape[0] > 0:
                        saved_gaussians = initialize_empty_train_gaussians_from_source(gaussians, opt)
                        saved_gaussians.densification_postfix(
                            new_gaussians["new_xyz"],
                            new_gaussians["new_features_dc"],
                            new_gaussians["new_features_rest"],
                            new_gaussians["new_opacities"],
                            new_gaussians["new_scaling"],
                            new_gaussians["new_rotation"],
                            new_gaussians["new_tmp_radii"],
                        )
                        saved_gaussians.save_ply(os.path.join(
                            seedfix_args.log_dir,
                            f"seed_package_iter_{iteration:06d}.ply",
                        ))
                    if new_gaussians is not None:
                        # 1) Main Gaussian seeds (may be empty after min_views).
                        if new_gaussians["new_xyz"].shape[0] > 0:
                            gaussians.densification_postfix(
                                new_gaussians["new_xyz"],
                                new_gaussians["new_features_dc"],
                                new_gaussians["new_features_rest"],
                                new_gaussians["new_opacities"],
                                new_gaussians["new_scaling"],
                                new_gaussians["new_rotation"],
                                new_gaussians["new_tmp_radii"],
                            )
                            print(f"seed_gaussians complete. Unique cloned gaussians: {new_gaussians['new_xyz'].shape[0]}")

                        # 2) NDF mesh-depth seeds are intentionally unchanged
                        # and independent of main-GS camera-support filtering.
                        if (
                            gaussolve_mode and gaussolves is not None
                            and new_gaussians.get("mesh_new_xyz") is not None
                            and new_gaussians["mesh_new_xyz"].shape[0] > 0
                        ):
                            num_points = new_gaussians["mesh_new_xyz"].shape[0]
                            target_alpha = 0.05
                            features = torch.zeros(
                                (num_points, 1, (gaussolves.max_sh_degree + 1) ** 2),
                                device=gaussolves.get_xyz.device,
                                dtype=gaussolves.get_xyz.dtype,
                            )
                            features[:, 0, 0] = (target_alpha - 0.5) / C0
                            gaussolves.densification_postfix(
                                new_gaussians["mesh_new_xyz"],
                                features[:, :, 0:1].transpose(1, 2).contiguous(),
                                features[:, :, 1:].transpose(1, 2).contiguous(),
                                new_gaussians["mesh_new_opacities"],
                                new_gaussians["mesh_new_scaling"],
                                new_gaussians["mesh_new_rotation"],
                                new_gaussians["mesh_new_tmp_radii"]
                            )
                            print(f"seed_gaussians complete. Also seeded {num_points} gaussolve points.")
                if main_structure_active and not main_mcmc and (
                    iteration % opt.opacity_reset_interval == 0
                    or (dataset.white_background and iteration == opt.densify_from_iter)
                ):
                    gaussians.reset_opacity()

                if (
                    ndf_structure_active
                    and dataset.gaussolve_reset_opacity is True
                    and iteration % opt.opacity_reset_interval == 0
                    and gaussolve_mode is True
                ):
                    gaussolves.reset_opacity()

                if (
                    main_mcmc
                    and main_structure_active
                    and iteration >= mcmc_relocate_from_global
                    and iteration < mcmc_relocate_until_global
                    and iteration % mcmc_interval == 0
                ):
                    mcmc_stats = relocate_and_grow_main(
                        gaussians,
                        cap_max=mcmc_cap_max,
                    )
                    mcmc_stats["seed_over_cap"] = max(
                        0,
                        int(gaussians.get_xyz.shape[0]) - mcmc_cap_max,
                    )
                    log_main_mcmc_event(iteration, mcmc_stats)
                    print(
                        "[MainMCMC] iter=%d before=%d dead=%d relocated=%d "
                        "added=%d after=%d cap=%d seed_over_cap=%d"
                        % (
                            int(iteration),
                            int(mcmc_stats["before"]),
                            int(mcmc_stats["dead"]),
                            int(mcmc_stats["relocated"]),
                            int(mcmc_stats["added"]),
                            int(mcmc_stats["after"]),
                            int(mcmc_stats["cap"]),
                            int(mcmc_stats["seed_over_cap"]),
                        )
                    )


            # 1000-iter self-prune: only run BEFORE mesh finetune begins. Once mesh starts
            # moving (iter >= mesh_start_finetune_iteration), gaussians can be transiently
            # occluded by the shifting mesh and would be permanently pruned, causing
            # catastrophic forgetting during finetune. Cap the self-prune window.
            if (not standard_main_prune_only and iteration > 0 and
                    iteration < opt.mesh_start_finetune_iteration and
                    (iteration % 1000 == 0) and hybrid_active and mesh_renderer is not None):
                with torch.no_grad():
                    P = gaussians.get_xyz.shape[0]
                    if not hasattr(gaussians, "contribution") or gaussians.contribution.shape[0] != P:
                        gaussians.contribution = torch.zeros(P, device=gaussians.get_xyz.device)
                    else:
                        gaussians.contribution.zero_()
                    visibility_filter_prune = torch.zeros(gaussians.get_xyz.shape[0], device=gaussians.get_xyz.device)
                    train_cams = scene.getTrainCameras()
                    
                    # 初始化 gaussolve 颜色收集（如果启用）
                    gaussolve_colors_list = []
                    
                    for cam in tqdm(train_cams, desc="Pruning views", leave=False):
                        render_pkg_prune = meshgs_render.prune_render(
                            cam, gaussians, pipe, bg,
                            use_trained_exp=dataset.train_test_exp,
                            separate_sh=SPARSE_ADAM_AVAILABLE,
                            mesh_depth_tolerance=dynamic_mesh_depth_tolerance,
                        )
                        visibility_filter = render_pkg_prune["visibility_filter"] == 1.0
                        visibility_filter_prune[visibility_filter] = 1.0
                        
                        # 同时计算 gaussolve 在当前视角的颜色值
                        # if gaussolve_mode is True and gaussolves is not None and gaussolves.get_xyz.shape[0] > 0:
                        #     dir_pp = (gaussolves.get_xyz - cam.camera_center.repeat(gaussolves.get_features.shape[0], 1))  # [N, 3]
                        #     dir_pp_normalized = dir_pp / dir_pp.norm(dim=1, keepdim=True)  # [N, 3]
                            
                        #     # 获取 SH 特征并计算颜色（与 gaussolve_render.py 中的逻辑一致）
                        #     feature_channels = gaussolves.get_features.shape[-1]  # 应该是 1
                        #     shs_view = gaussolves.get_features.transpose(1, 2).view(-1, feature_channels, (gaussolves.max_sh_degree + 1) ** 2)  # [N, 1, (deg+1)^2]
                            
                        #     # 使用 eval_sh 计算 SH 值
                        #     sh_values = eval_sh(gaussolves.active_sh_degree, shs_view, dir_pp_normalized)  # [N, 1]
                            
                        #     # 转换为颜色值（与 gaussolve_render.py 中的逻辑一致）
                        #     colors = torch.clamp_min(sh_values + 0.5, 0.0)  # [N, 1]
                        #     colors = colors.squeeze(-1)  # [N] 单通道值（alpha）
                        #     gaussolve_colors_list.append(colors)

                    prune_mask = visibility_filter_prune == 0
                    gaussians.prune_points(prune_mask)
                    print("self pruning points: ", prune_mask.sum())
                    
                    # Disabled: color-based gaussolve prune (per user request).
                    if gaussolve_mode is True and gaussolves is not None and len(gaussolve_colors_list) > 0:
                        all_colors_tensor = torch.stack(gaussolve_colors_list, dim=0)  # [num_cameras, N]
                        avg_colors = all_colors_tensor.mean(dim=0)  # [N]
                        color_threshold = getattr(opt, 'gaussolve_color_threshold', 0.005)
                        prune_mask_gaussolve = avg_colors < color_threshold
                        if prune_mask_gaussolve.any():
                            num_pruned = prune_mask_gaussolve.sum().item()
                            print(f"[ITER {iteration}] Pruning {num_pruned} gaussolve points with avg color < {color_threshold:.4f}")
                            gaussolves.prune_points(prune_mask_gaussolve)

            # Optimizer step
            if iteration < opt.iterations:
                if dataset.train_test_exp:
                    gaussians.exposure_optimizer.step()
                    gaussians.exposure_optimizer.zero_grad(set_to_none = True)

                # 三阶段 mesh 训练：
                # 1) 单独 mesh 预训练（上面已经完成）
                # 2) 中间阶段只训 GS / gaussolve，mesh 冻结不反向
                # 3) 到指定迭代（默认 20000）重新开启 mesh finetune
                mesh_finetune_start = opt.mesh_start_finetune_iteration
                if hybrid_active and mesh_renderer is not None and opt.mesh_train and iteration >= mesh_finetune_start:
                    if iteration == mesh_finetune_start:
                        # 预训练后曾经 freeze + 删除 optimizer，这里重新解冻并建立新的优化器
                        mesh_renderer.unfreeze_all()
                        mesh_renderer.training_setup(opt)
                        if mesh_appearance is not None:
                            # Stage-1 sidecars were frozen during the middle
                            # phase, so both texture and UV-SH begin a fresh
                            # schedule at this joint boundary.  Do not alter
                            # the historical scheduler used without a sidecar.
                            joint_max_steps = max(
                                1,
                                int(opt.iterations)
                                - int(mesh_finetune_start),
                            )
                            mesh_renderer.lr_scheduler = get_expon_lr_func(
                                lr_init=float(opt.mesh_lr_init),
                                lr_final=float(opt.mesh_lr_final),
                                lr_delay_steps=int(opt.mesh_lr_delay_steps),
                                lr_delay_mult=float(opt.mesh_lr_delay_mult),
                                max_steps=joint_max_steps,
                            )
                            mesh_renderer.update_learning_rate(0)
                            for parameter in mesh_appearance.parameters():
                                parameter.requires_grad_(True)
                            appearance_groups = mesh_appearance.optimizer_groups()
                            if appearance_groups:
                                appearance_base_lrs = []
                                for group in appearance_groups:
                                    scaled_lr = float(group["lr"]) * float(
                                        hybrid_args.appearance_joint_lr_scale
                                    )
                                    group["lr"] = scaled_lr
                                    appearance_base_lrs.append(scaled_lr)
                                appearance_optimizer = torch.optim.Adam(
                                    appearance_groups, lr=0.0, eps=1e-8
                                )
                                appearance_lr_scheduler = get_expon_lr_func(
                                    lr_init=1.0,
                                    lr_final=float(
                                        hybrid_args.appearance_lr_final_factor
                                    ),
                                    max_steps=joint_max_steps,
                                )
                                print(
                                    f"[Stage1] Joint {mesh_appearance.variant} auxiliary "
                                    "appearance finetuning enabled; "
                                    f"appearance_lr_scale={hybrid_args.appearance_joint_lr_scale:g}."
                                )
                            else:
                                # ``texture`` uses MeshRenderer's freshly
                                # restarted texture LR schedule only; do not
                                # create an empty auxiliary Adam optimizer.
                                appearance_base_lrs = []
                                appearance_optimizer = None
                                appearance_lr_scheduler = None
                                print(
                                    "[Stage1] Joint texture finetuning enabled; "
                                    "no auxiliary appearance optimizer groups."
                                )
                        else:
                            # Preserve the legacy no-sidecar resume behavior.
                            mesh_renderer.update_learning_rate(mesh_pretrain_iterations)
                    if mesh_renderer.optimizer is not None:
                        mesh_renderer.optimizer.step()
                        mesh_renderer.optimizer.zero_grad(set_to_none = True)
                    if appearance_optimizer is not None:
                        appearance_optimizer.step()
                        appearance_optimizer.zero_grad(set_to_none=True)
                if use_sparse_adam:
                    visible = radii > 0
                    gaussians.optimizer.step(visible, radii.shape[0])
                    gaussians.optimizer.zero_grad(set_to_none = True)
                else:
                    gaussians.optimizer.step()
                    gaussians.optimizer.zero_grad(set_to_none = True)
                if (
                    main_mcmc
                    and hybrid_active
                    and iteration >= hybrid_start_iteration + 1
                    and iteration <= int(opt.iterations)
                ):
                    apply_main_mcmc_noise(
                        gaussians,
                        xyz_lr=float(main_xyz_lr),
                        noise_lr=mcmc_noise_lr,
                    )
                if ndf_updates_enabled:
                    gaussolves.optimizer.step()
                    gaussolves.optimizer.zero_grad(set_to_none=True)

            if (iteration in checkpoint_iterations):
                print("\n[ITER {}] Saving Checkpoint".format(iteration))
                if task_path is not None:
                    torch.save((gaussians.capture(), iteration), os.path.join(scene.model_path, task_path) + "/chkpnt" + str(iteration) + ".pth")
                else:   
                    torch.save((gaussians.capture(), iteration), scene.model_path + "/chkpnt" + str(iteration) + ".pth")

    # 训练结束时保存最终 mesh
    if 'mesh_renderer' in locals() and mesh_renderer is not None:
        final_mesh_dir = task_output_dir if 'task_output_dir' in locals() and task_output_dir is not None else dataset.model_path
        os.makedirs(final_mesh_dir, exist_ok=True)
        final_mesh_path = os.path.join(final_mesh_dir, "mesh_final.glb")
        mesh_renderer.save_model(final_mesh_path)
        if mesh_appearance is not None:
            mesh_appearance.save(
                os.path.join(final_mesh_dir, "mesh_final.environment.pt"),
                extra={"iteration": int(opt.iterations), "stage": "hybrid_final"},
            )

    # 同时把 gaussolve 的高斯球保存为 ply，特征通道复制为 3 通道，方便统一用 RGB 方式可视化
    if gaussolve_mode and 'gaussolves' in locals() and gaussolves is not None:
        gaussolve_dir = task_output_dir if 'task_output_dir' in locals() and task_output_dir is not None else dataset.model_path
        os.makedirs(gaussolve_dir, exist_ok=True)
        gaussolve_ply_path = os.path.join(gaussolve_dir, "gaussolve_rgb.ply")
        with torch.no_grad():
            # 将单通道 SH 扩展为 3 通道，保证与普通 RGB GS 的 PLY 布局一致，方便查看器解析所有阶数。
            orig_dc = getattr(gaussolves, "_features_dc", None)      # [P, 1, C]，C=1
            orig_rest = getattr(gaussolves, "_features_rest", None)  # [P, K-1, C]，C=1

            if orig_dc is not None:
                dc = orig_dc.detach()
                if dc.shape[2] == 1:
                    # 在最后一维（通道维）上复制为 3 通道： [P, 1, 1] -> [P, 1, 3]
                    dc_rgb = dc.repeat(1, 1, 3)
                    gaussolves._features_dc = torch.nn.Parameter(dc_rgb)

            if orig_rest is not None:
                rest = orig_rest.detach()
                if rest.shape[2] == 1:
                    # 同样在通道维复制： [P, K-1, 1] -> [P, K-1, 3]
                    rest_rgb = rest.repeat(1, 1, 3)
                    gaussolves._features_rest = torch.nn.Parameter(rest_rgb)
            print(f"exporting gaussolve to {gaussolve_ply_path}")
            gaussolves.save_ply(gaussolve_ply_path)


def prepare_output_and_logger(args):    
    if not args.model_path:
        cur_time = datetime.now().strftime("%m%d_%H%M%S")
        
        if os.getenv('OAR_JOB_ID'):
            unique_str = os.getenv('OAR_JOB_ID')
        else:
            unique_str = str(uuid.uuid4())[:5] # 缩短 UUID，反正有时间戳
            
        args.model_path = os.path.join("./output/", f"{cur_time}_{unique_str}")
        
    print("\nOutput folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok = True)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    # Create Tensorboard writer
    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(args.model_path)
    else:
        print("\nTensorboard not available: not logging progress")
    return tb_writer

def training_report(tb_writer, iteration, Ll1, loss, l1_loss, elapsed, testing_iterations, scene : Scene, renderFunc, renderArgs, train_test_exp, metrics_output_path=None, full_train_metrics=False, exact_report_dir="", exact_report_iteration=None):
    exact_test_view_records = []
    if tb_writer:
        tb_writer.add_scalar('train_loss_patches/l1_loss', Ll1.item(), iteration)
        tb_writer.add_scalar('train_loss_patches/total_loss', loss.item(), iteration)
        tb_writer.add_scalar('iter_time', elapsed, iteration)

    # Report test and samples of training set
    if iteration in testing_iterations:
        torch.cuda.empty_cache()
        train_cameras = scene.getTrainCameras()
        validation_configs = ({'name': 'test', 'cameras' : scene.getTestCameras()},
                              {'name': 'train', 'cameras' : train_cameras if full_train_metrics else [train_cameras[idx % len(train_cameras)] for idx in range(5, 30, 5)]})

        for config in validation_configs:
            if config['cameras'] and len(config['cameras']) > 0:
                l1_test = 0.0
                psnr_test = 0.0
                ssim_test = 0.0
                lpips_test = 0.0
                for idx, viewpoint in enumerate(config['cameras']):
                    image = torch.clamp(renderFunc(viewpoint, scene.gaussians, *renderArgs)["render"], 0.0, 1.0)
                    gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
                    if train_test_exp:
                        image = image[..., image.shape[-1] // 2:]
                        gt_image = gt_image[..., gt_image.shape[-1] // 2:]
                    if tb_writer and (idx < 5):
                        tb_writer.add_images(config['name'] + "_view_{}/render".format(viewpoint.image_name), image[None], global_step=iteration)
                        if iteration == testing_iterations[0]:
                            tb_writer.add_images(config['name'] + "_view_{}/ground_truth".format(viewpoint.image_name), gt_image[None], global_step=iteration)
                    # 计算多种指标：L1 / PSNR / SSIM / LPIPS
                    l1_val = l1_loss(image, gt_image).mean().double()
                    psnr_val = psnr(image, gt_image).mean().double()
                    if (
                        exact_report_dir
                        and exact_report_iteration is not None
                        and int(iteration) == int(exact_report_iteration)
                        and config['name'] == 'test'
                    ):
                        exact_test_view_records.append(
                            save_test_view(
                                exact_report_dir,
                                iteration,
                                viewpoint,
                                image,
                                gt_image,
                                psnr_val.item(),
                            )
                        )
                    if full_train_metrics and config['name'] == 'train':
                        ssim_val = torch.zeros((), device=image.device, dtype=torch.float64)
                        lpips_val = torch.zeros((), device=image.device, dtype=torch.float64)
                    else:
                        ssim_val = ssim(image.unsqueeze(0), gt_image.unsqueeze(0))
                        lpips_val = lpips(image.unsqueeze(0), gt_image.unsqueeze(0), net_type='vgg')

                    l1_test += l1_val
                    psnr_test += psnr_val
                    ssim_test += ssim_val.double()
                    lpips_test += lpips_val.double()
                psnr_test /= len(config['cameras'])
                l1_test /= len(config['cameras'])
                ssim_test /= len(config['cameras'])
                lpips_test /= len(config['cameras'])
                # 打印时转成标量，避免出现 tensor([[[[...]]]]) 的形式
                print("[ITER {}] Evaluating {}: L1 {:.7f} PSNR {:.7f} SSIM {:.7f} LPIPS {:.7f}".format(
                    iteration,
                    config['name'],
                    l1_test.item() if torch.is_tensor(l1_test) else float(l1_test),
                    psnr_test.item() if torch.is_tensor(psnr_test) else float(psnr_test),
                    ssim_test.item() if torch.is_tensor(ssim_test) else float(ssim_test),
                    lpips_test.item() if torch.is_tensor(lpips_test) else float(lpips_test),
                ))
                if metrics_output_path:
                    try:
                        with open(metrics_output_path, "a", encoding="utf-8") as f_metrics:
                            f_metrics.write(
                                "{},{},{:.7f},{:.7f},{:.7f},{:.7f}\n".format(
                                    iteration,
                                    config['name'],
                                    l1_test.item() if torch.is_tensor(l1_test) else float(l1_test),
                                    psnr_test.item() if torch.is_tensor(psnr_test) else float(psnr_test),
                                    ssim_test.item() if torch.is_tensor(ssim_test) else float(ssim_test),
                                    lpips_test.item() if torch.is_tensor(lpips_test) else float(lpips_test),
                                )
                            )
                    except Exception as e:
                        print(f"[WARN] Failed to write metrics csv: {e}")
                if tb_writer:
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - l1_loss', l1_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - psnr', psnr_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - ssim', ssim_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - lpips', lpips_test, iteration)

        if tb_writer:
            tb_writer.add_histogram("scene/opacity_histogram", scene.gaussians.get_opacity, iteration)
            tb_writer.add_scalar('total_points', scene.gaussians.get_xyz.shape[0], iteration)
        torch.cuda.empty_cache()
    return exact_test_view_records

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    # The isolated schedule uses a global joint boundary of 20k.  Keep the
    # legacy OptimizationParams option visible for parity, but fail closed if
    # a caller explicitly requests any other boundary below.
    parser.set_defaults(mesh_start_finetune_iteration=20_000)
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6009)
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[7_000, 30_000])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[7_000, 30_000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument('--disable_viewer', action='store_true', default=False)
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--full_train_metrics", action="store_true", default=False,
                        help="evaluate all train cameras with L1/PSNR at each testing iteration")
    parser.add_argument("--start_checkpoint", type=str, default = None)
    parser.add_argument("--load_iteration", type=int, default = None)
    parser.add_argument("--task_path", type=str, default = None)
    parser.add_argument(
        "--seedfix_depth_mode", choices=["raw"], default="raw",
        help="frozen rebuttal contract: historical raw hybrid depth moment",
    )
    parser.add_argument(
        "--seedfix_min_views", type=int, default=2,
        help="distinct camera support required per main-Gaussian seed voxel",
    )
    parser.add_argument(
        "--seedfix_log_dir", type=str,
        default=os.path.join("..", "depth_fix_experiments", "seedfix_logs"),
        help="isolated per-seeding-round JSON diagnostics",
    )
    parser.add_argument(
        "--stage1_appearance_path",
        type=str,
        default="",
        help="UV-SH environment.pt paired with --mesh_path",
    )
    parser.add_argument(
        "--external_ndf_init_mode",
        choices=["none", "source_points", "direct_points", "guided_pt"],
        default="none",
        help=(
            "replace legacy bad-vertex NDF initialization; source_points reuses "
            "the filtered COLMAP/input.ply XYZ distribution"
        ),
    )
    parser.add_argument("--external_ndf_init_path", type=str, default="")
    parser.add_argument(
        "--pretrained_main_ply_path",
        type=str,
        default="",
        help=(
            "optional fresh-run ordinary-main Gaussian PLY; when set, skip the "
            "filtered-source one-shot main-GS construction"
        ),
    )
    parser.add_argument(
        "--pretrained_main_opacity_mode",
        choices=["keep", "reset"],
        default=None,
        help=(
            "required explicit opacity arm for --pretrained_main_ply_path; "
            "keep preserves PLY opacity, reset uses the required target value"
        ),
    )
    parser.add_argument(
        "--pretrained_main_opacity_reset_value",
        type=float,
        default=None,
        help="required opacity target in (0, 1) when opacity mode is reset",
    )
    parser.add_argument(
        "--pretrained_main_color_mode",
        choices=["keep", "reset"],
        default=None,
        help=(
            "required explicit color arm for --pretrained_main_ply_path; "
            "keep preserves full SH3 color, reset zeros DC and SH-rest"
        ),
    )
    parser.add_argument(
        "--pretrained_main_expected_count",
        type=int,
        default=-1,
        help="required positive PLY vertex count for --pretrained_main_ply_path",
    )
    parser.add_argument(
        "--pretrained_main_start_iteration",
        type=int,
        default=6_000,
        help=(
            "fresh PLY initialization start boundary; only applies with "
            "--pretrained_main_ply_path (the first loop iteration is start+1)"
        ),
    )
    parser.add_argument(
        "--forbid_periodic_seeding",
        action="store_true",
        default=False,
        help="fail closed if --seed_gaussians is present",
    )
    parser.add_argument(
        "--standard_main_prune_only",
        action="store_true",
        default=True,
        help=(
            "disable main-Gaussian visibility pruning (cycle and all-view self-prune); "
            "keep cycle opacity<0.005 cleanup, standard densify_and_prune, "
            "NDF opacity prune, and other non-visibility pruning"
        ),
    )
    parser.add_argument(
        "--main_mcmc",
        action="store_true",
        default=False,
        help=(
            "enable main-GS-only MCMC relocation/growth and SGLD; NDF/mesh "
            "densification and losses remain on the frozen path"
        ),
    )
    parser.add_argument(
        "--mcmc_cap_max",
        "--cap_max",
        dest="mcmc_cap_max",
        type=int,
        default=-1,
        help="hard cap for MCMC growth (seeding may be reported separately as seed_over_cap)",
    )
    parser.add_argument(
        "--mcmc_opacity_reg",
        "--opacity_reg",
        dest="mcmc_opacity_reg",
        type=float,
        default=0.0,
        help="main-GS opacity regularizer used by MCMC",
    )
    parser.add_argument(
        "--mcmc_scale_reg",
        "--scale_reg",
        dest="mcmc_scale_reg",
        type=float,
        default=0.0,
        help="main-GS activated-scale regularizer used by MCMC",
    )
    parser.add_argument(
        "--mcmc_noise_lr",
        "--noise_lr",
        dest="mcmc_noise_lr",
        type=float,
        default=0.0,
        help="main-GS covariance-shaped SGLD noise coefficient",
    )
    parser.add_argument(
        "--mcmc_relocate_from_global",
        "--relocate_from_global",
        dest="mcmc_relocate_from_global",
        type=int,
        default=-1,
        help="inclusive global iteration at which MCMC relocation begins",
    )
    parser.add_argument(
        "--mcmc_relocate_until_global",
        "--relocate_until_global",
        dest="mcmc_relocate_until_global",
        type=int,
        default=-1,
        help="exclusive global iteration at which MCMC relocation ends",
    )
    parser.add_argument(
        "--mcmc_interval",
        "--interval",
        dest="mcmc_interval",
        type=int,
        default=-1,
        help="global iteration interval for MCMC relocation/add-new",
    )
    parser.add_argument(
        "--densify_min_opacity",
        type=float,
        default=0.005,
        help="minimum opacity used by standard main/NDF densify-and-prune (legacy cycle cleanup stays at 0.005)",
    )
    parser.add_argument(
        "--log_densification_events",
        action="store_true",
        default=False,
        help="write a fresh densification_events.csv with main/NDF standard densify stats",
    )
    parser.add_argument(
        "--densify_split_mode",
        choices=["random", "deterministic_principal"],
        default="random",
        help="split-child placement mode; random is the historical default",
    )
    parser.add_argument(
        "--densify_split_max_offset_fraction",
        type=float,
        default=0.05,
        help="deterministic split offset cap as a fraction of scene extent",
    )
    parser.add_argument(
        "--mesh_aware_split",
        action="store_true",
        default=False,
        help=(
            "main-GS only: clamp split children that cross checked mesh "
            "triangle planes; NDF/gaussolve densification is unchanged"
        ),
    )
    parser.add_argument(
        "--mesh_aware_split_nearest_faces",
        type=int,
        choices=[1, 2, 3],
        default=1,
        help=(
            "mesh-aware main-GS split: check this many nearest triangle "
            "planes; default 1, conservative options 2 or 3"
        ),
    )
    parser.add_argument(
        "--gaussolve_sup_loss_scaling",
        action="store_true",
        default=False,
        help="scale Gaussolve supervision residuals by the detached main loss",
    )
    parser.add_argument(
        "--appearance_joint_lr_scale",
        type=float,
        default=0.1,
        help="scale Stage-1 UV-SH learning rates during mesh+GS joint finetuning",
    )
    parser.add_argument(
        "--appearance_lr_final_factor",
        type=float,
        default=0.1,
        help="exponential UV-SH LR multiplier at the end of joint finetuning",
    )
    parser.add_argument(
        "--mesh_direct_rgb",
        action="store_true",
        default=False,
        help="train the pre-baked texture directly in [0,1] (fast one-sample render path)",
    )
    parser.add_argument(
        "--hybrid_start_iteration",
        type=int,
        default=6_000,
        help=(
            "global delayed-hybrid boundary; supported multistart values are 6000, 9000, 12000"
        ),
    )
    parser.add_argument(
        "--main_densify_until_global",
        type=int,
        default=15_000,
        help="exclusive global cutoff for main-GS clone/split",
    )
    parser.add_argument(
        "--ndf_densify_until_global",
        type=int,
        default=15_000,
        help="exclusive global cutoff for NDF clone/split",
    )
    parser.add_argument(
        "--mesh_joint_start_global",
        type=int,
        default=20_000,
        help="global boundary for mesh/appearance joint training",
    )
    args = parser.parse_args(sys.argv[1:])
    if args.mesh_direct_rgb:
        args.use_sigmoid_rgb = False
    if args.densify_min_opacity < 0.0:
        parser.error("--densify_min_opacity must be >= 0")
    if args.densify_split_max_offset_fraction <= 0.0:
        parser.error("--densify_split_max_offset_fraction must be > 0")
    # The Hard-Z ablation is intentionally not exposed by the official
    # rebuttal entry.  It remains available through the archived experiment
    # entry points.
    args.hardz = False
    # Kept internally for the shared utility signature; raw mode never uses
    # this threshold to accept or reject a seed.
    args.seedfix_min_opacity_mass = 1e-4
    if args.seedfix_min_views != 2:
        parser.error("frozen rebuttal contract requires --seedfix_min_views 2")
    if args.seedfix_depth_mode != "raw":
        parser.error("frozen rebuttal contract requires --seedfix_depth_mode raw")
    if args.start_checkpoint is not None or args.load_iteration is not None:
        parser.error("the frozen initialization filter requires a fresh run; resume is unsupported")
    if args.stage1_appearance_path and not os.path.isfile(args.stage1_appearance_path):
        parser.error(f"missing --stage1_appearance_path: {args.stage1_appearance_path}")
    if args.external_ndf_init_mode != "none":
        if not args.gaussolve_mode:
            parser.error("external NDF initialization requires --gaussolve_mode")
        if args.external_ndf_init_mode == "source_points":
            if args.external_ndf_init_path:
                parser.error("source_points uses the filtered source_gaussians; do not pass --external_ndf_init_path")
        elif not args.external_ndf_init_path or not os.path.isfile(args.external_ndf_init_path):
            parser.error(
                f"missing --external_ndf_init_path: {args.external_ndf_init_path}"
            )
    elif args.external_ndf_init_path:
        parser.error("--external_ndf_init_path requires a non-'none' mode")
    pretrained_main_active = bool(args.pretrained_main_ply_path)
    if pretrained_main_active:
        if not os.path.isfile(args.pretrained_main_ply_path):
            parser.error(
                f"missing --pretrained_main_ply_path: {args.pretrained_main_ply_path}"
            )
        if args.pretrained_main_expected_count <= 0:
            parser.error(
                "pretrained main initialization requires "
                "--pretrained_main_expected_count > 0"
            )
        if args.pretrained_main_color_mode is None:
            parser.error(
                "pretrained main initialization requires explicit "
                "--pretrained_main_color_mode {keep,reset}"
            )
        if args.pretrained_main_opacity_mode is None:
            parser.error(
                "pretrained main initialization requires "
                "explicit --pretrained_main_opacity_mode {keep,reset}"
            )
        if args.pretrained_main_opacity_mode == "reset":
            if (
                args.pretrained_main_opacity_reset_value is None
                or not (0.0 < args.pretrained_main_opacity_reset_value < 1.0)
            ):
                parser.error(
                    "opacity reset mode requires "
                    "--pretrained_main_opacity_reset_value in (0, 1)"
                )
        elif args.pretrained_main_opacity_reset_value is not None:
            parser.error(
                "--pretrained_main_opacity_reset_value must be omitted "
                "when --pretrained_main_opacity_mode keep is selected"
            )
        if args.pretrained_main_start_iteration < 0 or args.pretrained_main_start_iteration >= args.iterations:
            parser.error(
                "--pretrained_main_start_iteration must be >= 0 and < --iterations"
            )
    elif (
        args.pretrained_main_color_mode is not None
        or args.pretrained_main_opacity_mode is not None
        or args.pretrained_main_opacity_reset_value is not None
        or args.pretrained_main_expected_count != -1
        or args.pretrained_main_start_iteration != 0
    ):
        parser.error(
            "pretrained main mode/value/count/start arguments require "
            "--pretrained_main_ply_path"
        )
    mcmc_values_supplied = any(
        (
            args.mcmc_cap_max != -1,
            args.mcmc_opacity_reg != 0.0,
            args.mcmc_scale_reg != 0.0,
            args.mcmc_noise_lr != 0.0,
            args.mcmc_relocate_from_global != -1,
            args.mcmc_relocate_until_global != -1,
            args.mcmc_interval != -1,
        )
    )
    if args.main_mcmc:
        if not args.standard_main_prune_only:
            parser.error("--main_mcmc requires --standard_main_prune_only")
        if not pretrained_main_active:
            parser.error("--main_mcmc requires --pretrained_main_ply_path")
        if args.mcmc_cap_max <= 0:
            parser.error("--main_mcmc requires --cap_max/--mcmc_cap_max > 0")
        if args.mcmc_opacity_reg < 0.0 or args.mcmc_scale_reg < 0.0 or args.mcmc_noise_lr < 0.0:
            parser.error("MCMC opacity_reg, scale_reg, and noise_lr must be non-negative")
        if args.mcmc_relocate_from_global < 0 or args.mcmc_relocate_until_global <= args.mcmc_relocate_from_global:
            parser.error("MCMC relocation global interval is invalid")
        if args.mcmc_interval <= 0:
            parser.error("--main_mcmc requires --interval/--mcmc_interval > 0")
        if args.mcmc_relocate_from_global % args.mcmc_interval != 0:
            parser.error("MCMC relocation start must align with --interval")
        if args.mcmc_relocate_from_global < args.hybrid_start_iteration + 1:
            parser.error("MCMC relocation must begin after the hybrid handoff")
        if args.mcmc_relocate_until_global > args.main_densify_until_global:
            parser.error("MCMC relocation must end no later than main densification cutoff")
    elif mcmc_values_supplied:
        parser.error("MCMC parameter flags require --main_mcmc")
    if not args.seed_gaussians:
        parser.error("multistart contract requires --seed_gaussians")
    if args.forbid_periodic_seeding:
        parser.error("multistart contract forbids --forbid_periodic_seeding")
    if args.seed_gaussians_interval != 1_500:
        parser.error("multistart contract requires --seed_gaussians_interval 1500")
    if args.seed_gaussians_gen_num_per_view != 300:
        parser.error("multistart contract requires --seed_gaussians_gen_num_per_view 300")
    if not (0.0 < args.appearance_joint_lr_scale <= 1.0):
        parser.error("--appearance_joint_lr_scale must be in (0, 1]")
    if not (0.0 < args.appearance_lr_final_factor <= 1.0):
        parser.error("--appearance_lr_final_factor must be in (0, 1]")
    if args.hybrid_start_iteration < 0 or args.hybrid_start_iteration >= args.iterations:
        parser.error("--hybrid_start_iteration must be >= 0 and < --iterations")
    # This entry point accepts only the audited 6k/9k/12k starts and the exact
    # global schedule.  A launcher cannot silently reinterpret a local suffix
    # as a global densification or mesh-joint boundary.
    if not pretrained_main_active:
        parser.error("multistart contract requires --pretrained_main_ply_path")
    if args.iterations != 30_000:
        parser.error("multistart contract requires --iterations 30000")
    if args.hybrid_start_iteration not in {6_000, 9_000, 12_000}:
        parser.error("multistart contract requires --hybrid_start_iteration in {6000,9000,12000}")
    if args.pretrained_main_start_iteration != args.hybrid_start_iteration:
        parser.error(
            "multistart contract requires pretrained and hybrid starts to be equal"
        )
    if args.seed_gaussians_start_iter > args.hybrid_start_iteration:
        parser.error(
            "seed_gaussians_start_iter must be <= hybrid_start_iteration so the "
            "first event is the next aligned global interval"
        )
    try:
        _build_hybrid_structure_contract(
            iterations=args.iterations,
            hybrid_start_iteration=args.hybrid_start_iteration,
            pretrained_main_start_iteration=args.pretrained_main_start_iteration,
            hybrid_local_iterations=args.iterations - args.hybrid_start_iteration,
            main_densify_until_global=args.main_densify_until_global,
            ndf_densify_until_global=args.ndf_densify_until_global,
            mesh_joint_start_global=args.mesh_joint_start_global,
            seed_gaussians_interval=args.seed_gaussians_interval,
            seed_gaussians_gen_num_per_view=args.seed_gaussians_gen_num_per_view,
            seed_gaussians_start_iter=args.seed_gaussians_start_iter,
        )
    except ValueError as exc:
        parser.error(str(exc))
    if args.mesh_start_finetune_iteration != args.mesh_joint_start_global:
        parser.error(
            "--mesh_start_finetune_iteration must equal global hybrid joint boundary "
            f"{args.mesh_joint_start_global}"
        )
    if args.densification_interval != 100:
        parser.error("multistart contract requires --densification_interval 100")
    args.save_iterations.append(args.iterations)
    
    print("\nOptimizing " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    # Start GUI server, configure and run training
    if not args.disable_viewer:
        network_gui.init(args.ip, args.port)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    dataset_args = lp.extract(args)
    dataset_args.hardz = False
    seedfix_args = Namespace(
        depth_mode=args.seedfix_depth_mode,
        min_views=args.seedfix_min_views,
        min_opacity_mass=args.seedfix_min_opacity_mass,
        log_dir=os.path.abspath(args.seedfix_log_dir),
    )
    hybrid_args = Namespace(
        stage1_appearance_path=os.path.abspath(args.stage1_appearance_path)
        if args.stage1_appearance_path
        else "",
        external_ndf_init_mode=args.external_ndf_init_mode,
        external_ndf_init_path=os.path.abspath(args.external_ndf_init_path)
        if args.external_ndf_init_path
        else "",
        pretrained_main_ply_path=os.path.abspath(args.pretrained_main_ply_path)
        if args.pretrained_main_ply_path
        else "",
        pretrained_main_color_mode=args.pretrained_main_color_mode,
        pretrained_main_opacity_mode=args.pretrained_main_opacity_mode,
        pretrained_main_opacity_reset_value=args.pretrained_main_opacity_reset_value,
        pretrained_main_expected_count=int(args.pretrained_main_expected_count),
        pretrained_main_start_iteration=int(args.pretrained_main_start_iteration),
        forbid_periodic_seeding=bool(args.forbid_periodic_seeding),
        appearance_joint_lr_scale=float(args.appearance_joint_lr_scale),
        appearance_lr_final_factor=float(args.appearance_lr_final_factor),
        hybrid_local_iterations=int(args.iterations - args.hybrid_start_iteration),
        main_densify_until_global=int(args.main_densify_until_global),
        ndf_densify_until_global=int(args.ndf_densify_until_global),
        mesh_joint_start_global=int(args.mesh_joint_start_global),
    )
    print(
        "[Seedfix] isolated seeding configuration: "
        f"depth_mode={seedfix_args.depth_mode}, min_views={seedfix_args.min_views}, "
        f"min_opacity_mass={seedfix_args.min_opacity_mass:g}, log_dir={seedfix_args.log_dir}"
    )
    print(
        "[NDF] mesh-depth cutoff: "
        + ("disabled (depth_image=None for NDF only)" if args.ndf_no_mesh_depth_cutoff else "historical enabled")
    )
    optimization_args = op.extract(args)
    optimization_args.standard_main_prune_only = bool(args.standard_main_prune_only)
    optimization_args.main_mcmc = bool(args.main_mcmc)
    optimization_args.mcmc_cap_max = int(args.mcmc_cap_max)
    optimization_args.mcmc_opacity_reg = float(args.mcmc_opacity_reg)
    optimization_args.mcmc_scale_reg = float(args.mcmc_scale_reg)
    optimization_args.mcmc_noise_lr = float(args.mcmc_noise_lr)
    optimization_args.mcmc_relocate_from_global = int(args.mcmc_relocate_from_global)
    optimization_args.mcmc_relocate_until_global = int(args.mcmc_relocate_until_global)
    optimization_args.mcmc_interval = int(args.mcmc_interval)
    optimization_args.gaussolve_sup_loss_scaling = bool(args.gaussolve_sup_loss_scaling)
    optimization_args.densify_min_opacity = float(args.densify_min_opacity)
    optimization_args.log_densification_events = bool(args.log_densification_events)
    optimization_args.densify_split_mode = str(args.densify_split_mode)
    optimization_args.densify_split_max_offset_fraction = float(
        args.densify_split_max_offset_fraction
    )
    optimization_args.mesh_aware_split = bool(args.mesh_aware_split)
    optimization_args.mesh_aware_split_nearest_faces = int(
        args.mesh_aware_split_nearest_faces
    )
    optimization_args.hybrid_start_iteration = int(args.hybrid_start_iteration)
    training(dataset_args, optimization_args, pp.extract(args), args.test_iterations,
             args.save_iterations, args.checkpoint_iterations, args.start_checkpoint,
             args.debug_from, args.load_iteration, args.task_path, seedfix_args,
             hybrid_args, args.full_train_metrics)

    # All done
    print("\nTraining complete.")
