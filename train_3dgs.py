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
import torch
import torchvision
from datetime import datetime
from random import randint
from utils.gaussolve_utils import tv_loss
from trimesh.constants import ToleranceMesh
from utils.loss_utils import l1_loss, ssim
from gaussian_renderer import render as regular_render, network_gui
from gaussolve_package.meshgs_render import MeshGsRender
from gaussolve_package.gaussolve_render import gaussolve_render
from gaussolve_package.gaussolve_model import collect_bad_mesh_vertices_multiview
import sys
from scene import Scene, GaussianModel
from utils.general_utils import safe_state, get_expon_lr_func, inverse_sigmoid
import uuid
from tqdm import tqdm
from utils.image_utils import psnr
from utils.graphics_utils import get_opengl_matrices
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams
from mesh_renderer import MeshRenderer
from utils.gaussolve_utils import prune_gaussians_on_good_mesh_multiview, initialize_empty_train_gaussians_from_source
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

def training(dataset, opt, pipe, testing_iterations, saving_iterations, checkpoint_iterations, checkpoint, debug_from, load_iteration, task_path, init_ply_path="", init_reset_opacity=None, init_expected_count=-1):
    
    if not SPARSE_ADAM_AVAILABLE and opt.optimizer_type == "sparse_adam":
        sys.exit(f"Trying to use sparse adam but it is not installed, please install the correct rasterizer using pip install [3dgs_accel].")

    first_iter = 0
    tb_writer = prepare_output_and_logger(dataset)
    gaussians = GaussianModel(dataset.sh_degree, opt.optimizer_type)
    scene = Scene(dataset, gaussians, load_iteration=load_iteration)
    if init_ply_path:
        gaussians.load_ply(init_ply_path, dataset.train_test_exp)
        loaded_count = int(gaussians.get_xyz.shape[0])
        if int(init_expected_count) > 0 and loaded_count != int(init_expected_count):
            raise ValueError(
                f"external init PLY count mismatch: expected {int(init_expected_count)}, got {loaded_count}"
            )
        if dataset.sh_degree != 3 or gaussians.max_sh_degree != 3:
            raise ValueError("external init PLY requires full SH degree 3")
        if init_reset_opacity is not None:
            target = float(init_reset_opacity)
            if not 0.0 < target < 1.0:
                raise ValueError("external init opacity target must be in (0, 1)")
            reset_raw_opacity = inverse_sigmoid(
                torch.full_like(gaussians._opacity, target)
            )
            gaussians._opacity = torch.nn.Parameter(
                reset_raw_opacity.requires_grad_(True)
            )
        print(
            f"[ExternalInit] loaded pure-3DGS PLY ({loaded_count} points)"
            + (
                f" and reset opacity to {float(init_reset_opacity):g}"
                if init_reset_opacity is not None
                else ""
            )
        )
    gaussians.training_setup(opt)
    if checkpoint:
        (model_params, first_iter) = torch.load(checkpoint)
        gaussians.restore(model_params, opt)

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    # Initialize mesh renderer if mesh_path is provided
    mesh_renderer = None
    if hasattr(dataset, 'mesh_path') and dataset.mesh_path and os.path.exists(dataset.mesh_path):
        texture_path = getattr(dataset, 'mesh_texture_path', None)
        if texture_path and not os.path.exists(texture_path):
            texture_path = None
        mesh_renderer = MeshRenderer(dataset.mesh_path, texture_path, device="cuda")
        print(f"Mesh renderer initialized with mesh: {dataset.mesh_path}")
        if opt.mesh_train:
            mesh_renderer.training_setup(opt)
            mesh_renderer.unfreeze()
            print(f"Mesh parameter training enabled. Current LR: {mesh_renderer.get_learning_rate():.6e}")
        else:
            mesh_renderer.freeze()

        for view in scene.getTrainCameras():
            view.open_gl_view, view.open_gl_mvp = get_opengl_matrices(view)
        for view in scene.getTestCameras():
            view.open_gl_view, view.open_gl_mvp = get_opengl_matrices(view)
    
    meshgs_render = MeshGsRender(mesh_renderer)


    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)

    use_sparse_adam = opt.optimizer_type == "sparse_adam" and SPARSE_ADAM_AVAILABLE 
    depth_l1_weight = get_expon_lr_func(opt.depth_l1_weight_init, opt.depth_l1_weight_final, max_steps=opt.iterations)

    ema_loss_for_log = 0.0
    ema_Ll1depth_for_log = 0.0

    stream1 = torch.cuda.Stream()  # 用于 mesh color 渲染
    
    new_collection = False

    # loss 阈值
    train_cameras = scene.getTrainCameras()
    
    mesh_pretrain_iterations = getattr(opt, "mesh_pretrain_iterations", 0)
    if mesh_renderer is not None and opt.mesh_train and mesh_renderer.optimizer is not None and mesh_pretrain_iterations > 0:
        print(f"Starting mesh pretraining for {mesh_pretrain_iterations} iterations...")
        
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
            image = mesh_renderer.color_from_rast(
                viewpoint_cam, pos, rast, mask, bg_color=bg, enable_antialiasing=True
            )

            if viewpoint_cam.mask is not None:
                alpha_mask = viewpoint_cam.mask.cuda()
                image *= alpha_mask

            gt_image = viewpoint_cam.original_image.cuda()
            Ll1 = l1_loss(image, gt_image)
            if FUSED_SSIM_AVAILABLE:
                ssim_value = fused_ssim(image.unsqueeze(0), gt_image.unsqueeze(0))
            else:
                ssim_value = ssim(image, gt_image)

            loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim_value)
            tv_w = 1e-5  # 建议从 1e-6 ~ 1e-4 试，噪声重就调大
            loss_tv = 0.0
            if mesh_renderer.texture is not None:
                # 根据参数化模式决定是否使用 sigmoid
                if getattr(mesh_renderer, "use_sigmoid_rgb", True):
                    tex01 = torch.sigmoid(mesh_renderer.texture)  # texture 是 logits 参数
                else:
                    tex01 = mesh_renderer._clamp_color(mesh_renderer.texture)  # texture 已经是 [0,1]
                # tex01: [H,W,3] -> [3,H,W]
                loss_tv = tv_loss(tex01.permute(2, 0, 1))

            loss = loss + tv_w * loss_tv
            loss.backward()

            with torch.no_grad():
                # Record per-camera mesh pretrain loss for cycle-wise averaging.
                viewpoint_cam.mesh_pretrain_loss = float(loss.item())
                mesh_renderer.update_learning_rate(mesh_iteration)
                mesh_renderer.optimizer.step()
                mesh_renderer.optimizer.zero_grad(set_to_none=True)
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
                            f"[Mesh pretrain cycle {cycle_idx}] avg loss: {avg_cycle_loss:.7f}, "
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
        mesh_renderer.save_model(dataset.model_path + "/mesh_pretrained.glb")
        print("Mesh pretraining complete.")


    # One-shot init from repository gaussians:
    # multi-view union mask -> add once to a fresh train_gaussians.
    source_gaussians = gaussians
    train_gaussians = initialize_empty_train_gaussians_from_source(source_gaussians, opt)
    mesh_depth_tolerance_fill = 0.05
    mesh_depth_tolerance = 0.00
    new_gaussians = None
    error_threshold = 0.01
    surface_thickness = 0.2
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
            print(f"One-shot clone complete. Unique cloned gaussians: {new_gaussians['new_xyz'].shape[0]}")
        else:
            print("One-shot clone complete. No gaussians selected; keep source model.")

    
    viewpoint_stack = scene.getTrainCameras().copy()
    viewpoint_indices = list(range(len(viewpoint_stack)))
    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress", dynamic_ncols=True)
    first_iter += 1
    for iteration in range(first_iter, opt.iterations + 1):
        iter_start.record()

        gaussians.update_learning_rate(iteration)
        if mesh_renderer is not None and opt.mesh_train:
            mesh_renderer.update_learning_rate(mesh_pretrain_iterations + iteration)

        # Every 1000 its we increase the levels of SH up to a maximum degree
        if iteration % 1000 == 0:
            gaussians.oneupSHdegree()
            if gaussians.active_sh_degree < gaussians.max_sh_degree:
                print("active_sh_degree: ", gaussians.active_sh_degree)
                print("max_sh_degree: ", gaussians.max_sh_degree)
                print("--------------------------------")

        # Pick a random Camera
        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()
            viewpoint_indices = list(range(len(viewpoint_stack)))
            new_collection = True
        rand_idx = randint(0, len(viewpoint_indices) - 1)
        viewpoint_cam = viewpoint_stack.pop(rand_idx)
        vind = viewpoint_indices.pop(rand_idx)

        # Render
        if (iteration - 1) == debug_from:
            pipe.debug = True

        bg = torch.rand((3), device="cuda") if opt.random_background else background

        # Step 1: Mesh rasterization and depth rendering (if mesh_renderer is available)
        if mesh_renderer is not None:
            # Rasterize mesh once
            pos, verts_view, rast, mask = mesh_renderer.rasterize(viewpoint_cam)
            depth_linear, depth_normalized = mesh_renderer.depth_from_rast(viewpoint_cam, verts_view, rast, mask, normalize=True)
            
            # Compute mesh depth
            mesh_depth = depth_linear # [1, H, W]

            # Step 2.1: Gaussian rendering with mesh depth (for CUDA-level fusion)
            # Note: mesh_depth is passed but CUDA code needs to be modified to use it
            render_pkg = meshgs_render.render(
                viewpoint_cam, gaussians, pipe, bg, 
                use_trained_exp=dataset.train_test_exp, 
                separate_sh=SPARSE_ADAM_AVAILABLE,
                mesh_depth=mesh_depth,
                mesh_depth_tolerance=mesh_depth_tolerance
            )
            
            # Step 2.2: new thread to render mesh color
            with torch.cuda.stream(stream1):
                mesh_rendered = mesh_renderer.color_from_rast(viewpoint_cam, pos, rast, mask, bg_color = bg, enable_antialiasing = True)
            torch.cuda.current_stream().wait_stream(stream1)
            image = mesh_rendered * render_pkg["alpha"] + render_pkg["render"]

        else:
            render_pkg = regular_render(
                viewpoint_cam, gaussians, pipe, bg, 
                use_trained_exp=dataset.train_test_exp, 
                separate_sh=SPARSE_ADAM_AVAILABLE
            )
            image = render_pkg["render"]
        depth = render_pkg["depth"]
        radii = render_pkg["radii"]
        visibility_filter = render_pkg["visibility_filter"]
        viewspace_point_tensor = render_pkg["viewspace_points"]


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
        # if depth_l1_weight(iteration) > 0 and viewpoint_cam.depth_reliable:
        #     mono_invdepth = viewpoint_cam.invdepthmap.cuda()
        #     depth_mask = viewpoint_cam.depth_mask.cuda()

        #     invDepth = render_pkg["depth"]  # GS inverse depth
        #     if mesh_renderer is not None:
        #         mesh_mask = mask.float()
        #         if mesh_mask.dim() == 4:
        #             mesh_mask = mesh_mask.squeeze(-1)
        #         fused_invdepth = (1.0 / (mesh_depth + 1e-6)) * mesh_mask
        #     else:
        #         fused_invdepth = invDepth

        #     Ll1depth_pure = torch.abs((fused_invdepth - mono_invdepth) * depth_mask).mean()
        #     Ll1depth = depth_l1_weight(iteration) * Ll1depth_pure
        #     loss += Ll1depth
        #     Ll1depth = Ll1depth.item()

        loss.backward()

        iter_end.record()

        with torch.no_grad():
            # Progress bar
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            ema_Ll1depth_for_log = 0.4 * Ll1depth + 0.6 * ema_Ll1depth_for_log

            if iteration % 10 == 0:
                progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.{7}f}", "Depth Loss": f"{ema_Ll1depth_for_log:.{7}f}"})
                progress_bar.update(10)
            if iteration == opt.iterations:
                progress_bar.close()

            # Periodic test/train eval report (defines testing_iterations from CLI).
            training_report(tb_writer, iteration, Ll1, loss, l1_loss,
                            iter_start.elapsed_time(iter_end), testing_iterations, scene,
                            regular_render, (pipe, background), dataset.train_test_exp)

            if (iteration in saving_iterations):
                print("\n[ITER {}] Saving Gaussians".format(iteration))
                scene.save(iteration, task_path)
    
            # Densification
            if iteration < opt.densify_until_iter:
                # Keep track of max radii in image-space for pruning
                gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
                gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)
                # if mesh_renderer is not None:
                #     visibility_filter = render_pkg["gs_visibility_under_mesh"] == 1.0
                #     gaussians.visibility[visibility_filter] = 1.0
                #     if new_collection:
                #         prune_mask = gaussians.visibility == 0
                #         # print("gs under mesh points: ", prune_mask.sum())
                #         gaussians.prune_points(prune_mask)
                #         gaussians.visibility = torch.zeros_like(gaussians.visibility)
                #         new_collection = False

                if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                    size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                    gaussians.densify_and_prune(opt.densify_grad_threshold, 0.005, scene.cameras_extent, size_threshold, radii)
                if iteration % opt.opacity_reset_interval == 0 or (dataset.white_background and iteration == opt.densify_from_iter):
                    gaussians.reset_opacity()
            

            # Optimizer step
            if iteration < opt.iterations:
                if dataset.train_test_exp:
                    gaussians.exposure_optimizer.step()
                    gaussians.exposure_optimizer.zero_grad(set_to_none = True)
                # if mesh_renderer is not None and opt.mesh_train and mesh_renderer.optimizer is not None:
                #     mesh_renderer.optimizer.step()
                #     mesh_renderer.optimizer.zero_grad(set_to_none = True)
                if use_sparse_adam:
                    visible = radii > 0
                    gaussians.optimizer.step(visible, radii.shape[0])
                    gaussians.optimizer.zero_grad(set_to_none = True)
                else:
                    gaussians.optimizer.step()
                    gaussians.optimizer.zero_grad(set_to_none = True)

            if (iteration in checkpoint_iterations):
                print("\n[ITER {}] Saving Checkpoint".format(iteration))
                if task_path is not None:
                    torch.save((gaussians.capture(), iteration), os.path.join(scene.model_path, task_path) + "/chkpnt" + str(iteration) + ".pth")
                else:   
                    torch.save((gaussians.capture(), iteration), scene.model_path + "/chkpnt" + str(iteration) + ".pth")

def prepare_output_and_logger(args):    
    if not args.model_path:
        cur_time = datetime.now().strftime("%m%d_%H%M%S")
        
        if os.getenv('OAR_JOB_ID'):
            unique_str = os.getenv('OAR_JOB_ID')
        else:
            unique_str = str(uuid.uuid4())[:5] # 缩短 UUID，反正有时间戳
            
        args.model_path = os.path.join("./output/", f"{cur_time}_{unique_str}")
        
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok = True)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    # Create Tensorboard writer
    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(args.model_path)
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer

def training_report(tb_writer, iteration, Ll1, loss, l1_loss, elapsed, testing_iterations, scene : Scene, renderFunc, renderArgs, train_test_exp):
    if tb_writer:
        tb_writer.add_scalar('train_loss_patches/l1_loss', Ll1.item(), iteration)
        tb_writer.add_scalar('train_loss_patches/total_loss', loss.item(), iteration)
        tb_writer.add_scalar('iter_time', elapsed, iteration)

    # Report test and samples of training set
    if iteration in testing_iterations:
        torch.cuda.empty_cache()
        validation_configs = ({'name': 'test', 'cameras' : scene.getTestCameras()}, 
                              {'name': 'train', 'cameras' : [scene.getTrainCameras()[idx % len(scene.getTrainCameras())] for idx in range(5, 30, 5)]})

        for config in validation_configs:
            if config['cameras'] and len(config['cameras']) > 0:
                l1_test = 0.0
                psnr_test = 0.0
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
                    l1_test += l1_loss(image, gt_image).mean().double()
                    psnr_test += psnr(image, gt_image).mean().double()
                psnr_test /= len(config['cameras'])
                l1_test /= len(config['cameras'])          
                print("\n[ITER {}] Evaluating {}: L1 {} PSNR {}".format(iteration, config['name'], l1_test, psnr_test))
                if tb_writer:
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - l1_loss', l1_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - psnr', psnr_test, iteration)

        if tb_writer:
            tb_writer.add_histogram("scene/opacity_histogram", scene.gaussians.get_opacity, iteration)
            tb_writer.add_scalar('total_points', scene.gaussians.get_xyz.shape[0], iteration)
        torch.cuda.empty_cache()

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6009)
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[7_000, 30_000])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[7_000, 30_000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument('--disable_viewer', action='store_true', default=False)
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default = None)
    parser.add_argument("--load_iteration", type=int, default = None)
    parser.add_argument("--task_path", type=str, default = None)
    parser.add_argument(
        "--init_ply_path",
        type=str,
        default="",
        help="optional external ordinary-3DGS PLY loaded before optimizer setup",
    )
    parser.add_argument(
        "--init_reset_opacity",
        type=float,
        default=None,
        help="optional opacity target for --init_ply_path",
    )
    parser.add_argument(
        "--init_expected_count",
        type=int,
        default=-1,
        help="optional expected vertex count for --init_ply_path",
    )
    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)
    
    print("Optimizing " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    # Start GUI server, configure and run training
    if not args.disable_viewer:
        network_gui.init(args.ip, args.port)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    training(lp.extract(args), op.extract(args), pp.extract(args), args.test_iterations, 
             args.save_iterations, args.checkpoint_iterations, args.start_checkpoint, 
             args.debug_from, args.load_iteration, args.task_path,
             args.init_ply_path, args.init_reset_opacity, args.init_expected_count)

    # All done
    print("\nTraining complete.")
