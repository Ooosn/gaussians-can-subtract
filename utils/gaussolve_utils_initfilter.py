import os
import math

import torch
from tqdm import tqdm
from utils.sh_utils import C0
from utils.seedfix_contract import backproject_sampled_mesh_depth, distinct_camera_support, hybrid_depth_for_seeding, write_json_atomic
from simple_knn._C import distCUDA2

def initialize_empty_train_gaussians_from_source(source_gaussians, training_args):
    """
    Create a fresh trainable GaussianModel aligned to source tensor layouts.
    """
    train_gaussians = source_gaussians.__class__(source_gaussians.max_sh_degree, source_gaussians.optimizer_type)
    train_gaussians.active_sh_degree = source_gaussians.active_sh_degree
    train_gaussians.spatial_lr_scale = source_gaussians.spatial_lr_scale
    train_gaussians.train_test_exp = source_gaussians.train_test_exp
    train_gaussians.pretrained_exposures = getattr(source_gaussians, "pretrained_exposures", None)
    train_gaussians.exposure_mapping = getattr(source_gaussians, "exposure_mapping", None)

    xyz = source_gaussians._xyz
    fdc = source_gaussians._features_dc
    frest = source_gaussians._features_rest
    opacity = source_gaussians._opacity
    scaling = source_gaussians._scaling
    rotation = source_gaussians._rotation

    train_gaussians._xyz = torch.nn.Parameter(torch.empty((0, xyz.shape[1]), device=xyz.device, dtype=xyz.dtype).requires_grad_(True))
    train_gaussians._features_dc = torch.nn.Parameter(torch.empty((0, fdc.shape[1], fdc.shape[2]), device=fdc.device, dtype=fdc.dtype).requires_grad_(True))
    train_gaussians._features_rest = torch.nn.Parameter(torch.empty((0, frest.shape[1], frest.shape[2]), device=frest.device, dtype=frest.dtype).requires_grad_(True))
    train_gaussians._opacity = torch.nn.Parameter(torch.empty((0, opacity.shape[1]), device=opacity.device, dtype=opacity.dtype).requires_grad_(True))
    train_gaussians._scaling = torch.nn.Parameter(torch.empty((0, scaling.shape[1]), device=scaling.device, dtype=scaling.dtype).requires_grad_(True))
    train_gaussians._rotation = torch.nn.Parameter(torch.empty((0, rotation.shape[1]), device=rotation.device, dtype=rotation.dtype).requires_grad_(True))
    train_gaussians.max_radii2D = torch.zeros((0,), device=xyz.device, dtype=xyz.dtype)
    train_gaussians.xyz_gradient_accum = torch.zeros((0, 1), device=xyz.device, dtype=xyz.dtype)
    train_gaussians.denom = torch.zeros((0, 1), device=xyz.device, dtype=xyz.dtype)
    train_gaussians.visibility = torch.zeros((0,), device=xyz.device, dtype=xyz.dtype)

    if train_gaussians.train_test_exp and hasattr(source_gaussians, "_exposure"):
        exposure = source_gaussians._exposure
        train_gaussians._exposure = torch.nn.Parameter(
            exposure.detach().clone().to(device=exposure.device, dtype=exposure.dtype).requires_grad_(True)
        )

    train_gaussians.training_setup(training_args)
    train_gaussians.tmp_radii = torch.zeros((0,), device=xyz.device, dtype=xyz.dtype)
    return train_gaussians


def _select_indices_for_view(
    source_gaussians,
    viewpoint_cam,
    render_pkg_mesh,
    render_pkg_gs,
    gt_image,
    error_threshold=0.05,
    depth_epsilon=0.05,
    surface_thickness=0.05,
    mode="front",          # "front": 只要前向 GS; "back": 只要被 Mesh 挡住/重合的 GS; "both": 深度不做筛选
):
    with torch.no_grad():
        H, W = gt_image.shape[1], gt_image.shape[2]

        mesh_image = render_pkg_mesh["image"]
        err_map_mesh = torch.abs(mesh_image - gt_image).mean(dim=0)  # [H, W]

        depth_mesh = render_pkg_mesh["depth"]
        depth_gs = render_pkg_gs["depth"]
        if depth_mesh.dim() == 3:
            depth_mesh = depth_mesh.squeeze(0)
        if depth_gs.dim() == 3:
            depth_gs = depth_gs.squeeze(0)

        xyz = source_gaussians.get_xyz.detach()
        xyz_hom = torch.cat([xyz, torch.ones_like(xyz[:, :1])], dim=1)

        p_hom = xyz_hom @ viewpoint_cam.full_proj_transform
        p_w = 1.0 / (p_hom[:, 3:4] + 1e-7)
        p_ndc = p_hom[:, :3] * p_w

        p_view = xyz_hom @ viewpoint_cam.world_view_transform
        linear_depth = p_view[:, 2]

        u = ((p_ndc[:, 0] + 1.0) * W - 1.0) * 0.5
        v = ((p_ndc[:, 1] + 1.0) * H - 1.0) * 0.5
        u_idx = u.long()
        v_idx = v.long()

        valid_frustum = (u_idx >= 0) & (u_idx < W) & (v_idx >= 0) & (v_idx < H) & (linear_depth > 0.2)
        candidate_indices = torch.where(valid_frustum)[0]
        if candidate_indices.shape[0] == 0:
            return None

        u_valid = u_idx[candidate_indices]
        v_valid = v_idx[candidate_indices]
        z_valid = linear_depth[candidate_indices]

        # 逐 Gaussian 采样误差和深度信息
        err_per_gauss = err_map_mesh[v_valid, u_valid]
        depth_mesh_sampled = depth_mesh[v_valid, u_valid]
        depth_gs_sampled = depth_gs[v_valid, u_valid]

        err_mask = err_per_gauss > error_threshold

        # 深度关系条件
        # 记 Z_m = depth_mesh_sampled, Z_g = depth_gs_sampled
        # front :  Z_g <  Z_m - eps      （明显在 Mesh 前面）
        # back  :  Z_g >  Z_m + eps      （明显在 Mesh 后面）
        # both  :  front OR back         （显式排除 |Z_g - Z_m| <= eps 这一薄层）
        if mode == "front":
            depth_mask = depth_gs_sampled < (depth_mesh_sampled - depth_epsilon)
        elif mode == "back":
            depth_mask = depth_gs_sampled > (depth_mesh_sampled + depth_epsilon)
        else:  # "both"
            front_mask = depth_gs_sampled < (depth_mesh_sampled - depth_epsilon)
            back_mask = depth_gs_sampled > (depth_mesh_sampled + depth_epsilon)
            depth_mask = front_mask | back_mask

        rendered_z = depth_gs_sampled
        is_surface = torch.abs(z_valid - rendered_z) < surface_thickness

        final_mask = err_mask & depth_mask & is_surface
        final_indices = candidate_indices[final_mask]
        
        if final_indices.shape[0] == 0:
            return None
        return final_indices


def _select_indices_for_pruning_view(
    source_gaussians,
    viewpoint_cam,
    render_pkg_mesh,
    render_pkg_gs,
    gt_image,
    error_threshold=0.05,
    depth_epsilon=0.05,
    surface_thickness=0.5,
):
    """
    选择“可以删掉的高斯点”：
      - Loss 极低（Mesh 已经很好地解释了该区域）：err < error_threshold_low
      - 深度基本重合：|Z_gs - Z_mesh| <= depth_epsilon
      - 位于当前视角的几何表面壳上：|linear_depth - rendered_z| < surface_thickness

    返回：在当前视角下满足条件的高斯索引（1D LongTensor），或 None。
    """
    with torch.no_grad():
        H, W = gt_image.shape[1], gt_image.shape[2]

        mesh_image = render_pkg_mesh["image"]
        err_map_mesh = torch.abs(mesh_image - gt_image).mean(dim=0)  # [H, W]

        depth_mesh = render_pkg_mesh["depth"]
        depth_gs = render_pkg_gs["depth"]
        if depth_mesh.dim() == 3:
            depth_mesh = depth_mesh.squeeze(0)
        if depth_gs.dim() == 3:
            depth_gs = depth_gs.squeeze(0)

        xyz = source_gaussians.get_xyz.detach()
        xyz_hom = torch.cat([xyz, torch.ones_like(xyz[:, :1])], dim=1)

        p_hom = xyz_hom @ viewpoint_cam.full_proj_transform
        p_w = 1.0 / (p_hom[:, 3:4] + 1e-7)
        p_ndc = p_hom[:, :3] * p_w

        p_view = xyz_hom @ viewpoint_cam.world_view_transform
        linear_depth = p_view[:, 2]

        u = ((p_ndc[:, 0] + 1.0) * W - 1.0) * 0.5
        v = ((p_ndc[:, 1] + 1.0) * H - 1.0) * 0.5
        u_idx = u.long()
        v_idx = v.long()

        valid_frustum = (u_idx >= 0) & (u_idx < W) & (v_idx >= 0) & (v_idx < H) & (linear_depth > 0.2)
        candidate_indices = torch.where(valid_frustum)[0]
        if candidate_indices.shape[0] == 0:
            return None

        u_valid = u_idx[candidate_indices]
        v_valid = v_idx[candidate_indices]
        z_valid = linear_depth[candidate_indices]

        # 逐 Gaussian 采样误差和深度信息
        err_per_gauss = err_map_mesh[v_valid, u_valid]
        depth_mesh_sampled = depth_mesh[v_valid, u_valid]
        depth_gs_sampled = depth_gs[v_valid, u_valid]

        # “好点”：误差很小 & 深度几乎一致
        err_mask = err_per_gauss < error_threshold
        depth_close = torch.abs(depth_gs_sampled - depth_mesh_sampled) <= depth_epsilon

        rendered_z = depth_gs_sampled
        is_surface = torch.abs(z_valid - rendered_z) < surface_thickness

        final_mask = err_mask & depth_close & is_surface
        final_indices = candidate_indices[final_mask]

        if final_indices.shape[0] == 0:
            return None
        return final_indices


def _pack_new_gaussians_from_indices(source_gaussians, final_indices, opacity_cap=0.01):
    new_xyz = source_gaussians.get_xyz[final_indices].clone()
    new_features_dc = source_gaussians._features_dc[final_indices].clone()
    new_features_rest = source_gaussians._features_rest[final_indices].clone()
    new_scaling = source_gaussians._scaling[final_indices].clone()
    new_rotation = source_gaussians._rotation[final_indices].clone()
    current_opacity = source_gaussians.get_opacity[final_indices]
    target_opacity = torch.min(current_opacity, torch.ones_like(current_opacity) * opacity_cap)
    new_opacities = source_gaussians.inverse_opacity_activation(target_opacity)
    return {
        "new_xyz": new_xyz,
        "new_features_dc": new_features_dc,
        "new_features_rest": new_features_rest,
        "new_opacities": new_opacities,
        "new_scaling": new_scaling,
        "new_rotation": new_rotation,
        "new_tmp_radii": torch.zeros((final_indices.shape[0]), device="cuda"),
    }


def clone_gaussians_for_patching(
    source_gaussians,       # 原始 3DGS 模型对象 (提供 .get_xyz, .get_features 等)
    viewpoint_cam,          # 当前视角的相机对象 (提供 world_view_transform, full_proj_transform)
    render_pkg_mesh,        # Mesh 渲染结果字典 {'image': ..., 'depth': ...}
    render_pkg_gs,          # 3DGS 渲染结果字典 {'image': ..., 'depth': ...}
    gt_image,               # 真实图像 GT
    
    # --- 阈值参数 ---
    error_threshold=0.02,   # 误差阈值：Loss 大于此值才修补
    depth_epsilon=0.01,     # 深度容差：Mesh 比 GS 远这么多才算“凹陷”
    surface_thickness=0.05, # 表面厚度：只克隆表面这层壳，不克隆内部被遮挡的高斯
    clone_mode="front",     # "front": 仅克隆前向 GS; "back": 仅克隆被 Mesh 挡住/重合的 GS; "both": 前后都克隆
):
    """
    基于深度先验和误差图，从原始 3DGS 中克隆高斯球来修补 Mesh 的空缺。
    """
    
    final_indices = _select_indices_for_view(
        source_gaussians=source_gaussians,
        viewpoint_cam=viewpoint_cam,
        render_pkg_mesh=render_pkg_mesh,
        render_pkg_gs=render_pkg_gs,
        gt_image=gt_image,
        error_threshold=error_threshold,
        depth_epsilon=depth_epsilon,
        surface_thickness=surface_thickness,
        mode=clone_mode,
    )
    if final_indices is None:
        return None
    return _pack_new_gaussians_from_indices(source_gaussians, final_indices, opacity_cap=0.01)


def clone_gaussians_for_patching_multiview(
    source_gaussians,
    cameras,
    mesh_renderer,
    gs_render_fn,
    pipe,
    background,
    random_background=False,
    use_trained_exp=False,
    separate_sh=False,
    error_threshold=0.05,
    depth_epsilon=0.05,
    surface_thickness=0.05,
    clone_mode="front",     # "front" / "back" / "both"
    min_views: int = 1,     # 至少在多少个视角下被选中，才算最终有效（默认 1：原始行为）
):
    """
    多视角走一圈，累计仓库高斯 mask，最后只返回一次去重后的新增高斯。
    """
    with torch.no_grad():
        num_points = source_gaussians.get_xyz.shape[0]
        if num_points == 0:
            return None
        # 记录每个高斯在多少个视角下被选中
        global_counts = torch.zeros((num_points,), dtype=torch.int32, device=source_gaussians.get_xyz.device)

    view_bar = tqdm(cameras, desc="Clone mask over views", dynamic_ncols=True)
    for cam_idx, viewpoint_cam in enumerate(view_bar):
        bg = torch.rand((3), device=background.device) if random_background else background
        pos, verts_view, rast, mask = mesh_renderer.rasterize(viewpoint_cam)
        image_mesh = mesh_renderer.color_from_rast(
            viewpoint_cam, pos, rast, mask, bg_color=bg, enable_antialiasing=True
        )
        depth_mesh_linear, _ = mesh_renderer.depth_from_rast(viewpoint_cam, verts_view, rast, mask, normalize=True)
        gs_pkg = gs_render_fn(
            viewpoint_cam, source_gaussians, pipe, bg,
            use_trained_exp=use_trained_exp,
            separate_sh=separate_sh,
            gs_depth_is_inverse=False
        )
        final_indices = _select_indices_for_view(
            source_gaussians=source_gaussians,
            viewpoint_cam=viewpoint_cam,
            render_pkg_mesh={"image": image_mesh, "depth": depth_mesh_linear},
            render_pkg_gs={"image": gs_pkg["render"], "depth": gs_pkg["depth"]},
            gt_image=viewpoint_cam.original_image.cuda(),
            error_threshold=error_threshold,
            depth_epsilon=depth_epsilon,
            surface_thickness=surface_thickness,
            mode=clone_mode,
        )
        if final_indices is not None:
            global_counts[final_indices] += 1
        # 仅用于可视化：当前满足 min_views 的数量
        current_mask = global_counts >= max(1, min_views)
        view_bar.set_postfix({
            "added_this_view": 0 if final_indices is None else final_indices.shape[0],
            "valid_total": current_mask.sum().item(),
        })
    view_bar.close()
    # 多视角一致性：至少在 min_views 个视角下被选中
    if min_views <= 1:
        merged_indices = torch.where(global_counts > 0)[0]
    else:
        merged_indices = torch.where(global_counts >= min_views)[0]
    if merged_indices.shape[0] == 0:
        return None
    packed = _pack_new_gaussians_from_indices(source_gaussians, merged_indices, opacity_cap=0.01)
    # 同步返回一个布尔 mask，表示哪些原始高斯通过了 min_views 条件
    source_mask = torch.zeros_like(global_counts, dtype=torch.bool)
    source_mask[merged_indices] = True
    packed["source_mask"] = source_mask
    return packed


def prune_gaussians_on_good_mesh_multiview(
    source_gaussians,
    cameras,
    mesh_renderer,
    gs_render_fn,
    pipe,
    background,
    random_background=False,
    use_trained_exp=False,
    separate_sh=False,
    error_threshold=0.05,
    depth_epsilon=0.05,
    surface_thickness=0.5,
    min_views: int = 1,
 ):      
    """
    多视角扫描，找出“寄生在健康 Mesh 表面”的冗余高斯点，并返回一包对应的源高斯：
      - 这些点在大部分视角下 Loss 极低（Mesh 已经完美解释），且
      - GS 深度与 Mesh 深度高度重合。

    主要用于 prune：source_gaussians.prune_points(prune_mask)
    """
    with torch.no_grad():
        num_points = source_gaussians.get_xyz.shape[0]
        if num_points == 0:
            return None
        global_counts = torch.zeros((num_points,), dtype=torch.int32, device=source_gaussians.get_xyz.device)

    view_bar = tqdm(cameras, desc="Prune mask over views", dynamic_ncols=True)
    for cam_idx, viewpoint_cam in enumerate(view_bar):
        bg = torch.rand((3), device=background.device) if random_background else background
        pos, verts_view, rast, mask = mesh_renderer.rasterize(viewpoint_cam)
        image_mesh = mesh_renderer.color_from_rast(
            viewpoint_cam, pos, rast, mask, bg_color=bg, enable_antialiasing=True
        )
        depth_mesh_linear, _ = mesh_renderer.depth_from_rast(viewpoint_cam, verts_view, rast, mask, normalize=True)
        gs_pkg = gs_render_fn(
            viewpoint_cam, source_gaussians, pipe, bg,
            use_trained_exp=use_trained_exp,
            separate_sh=separate_sh,
            gs_depth_is_inverse=False
        )
        final_indices = _select_indices_for_pruning_view(
            source_gaussians=source_gaussians,
            viewpoint_cam=viewpoint_cam,
            render_pkg_mesh={"image": image_mesh, "depth": depth_mesh_linear},
            render_pkg_gs={"image": gs_pkg["render"], "depth": gs_pkg["depth"]},
            gt_image=viewpoint_cam.original_image.cuda(),
            error_threshold=error_threshold,
            depth_epsilon=depth_epsilon,
            surface_thickness=surface_thickness,
        )
        if final_indices is not None:
            global_counts[final_indices] += 1
        current_mask = global_counts >= max(1, min_views)
        view_bar.set_postfix({
            "marked_this_view": 0 if final_indices is None else final_indices.shape[0],
            "prune_total": current_mask.sum().item(),
        })
    view_bar.close()

    if min_views <= 1:
        keep_indices = torch.where(global_counts == 0)[0]
    else:
        keep_indices = torch.where(global_counts < min_views)[0]
        
    if keep_indices.shape[0] == 0:
        return None

    packed = _pack_new_gaussians_from_indices(source_gaussians, keep_indices, opacity_cap=0.01)
    
    source_mask = torch.zeros_like(global_counts, dtype=torch.bool)
    source_mask[keep_indices] = True
    packed["source_mask"] = source_mask
    
    return packed

def tv_loss(img_chw: torch.Tensor) -> torch.Tensor:
    # img_chw: [3, H, W] in [0,1]
    return (img_chw[:, :, 1:] - img_chw[:, :, :-1]).abs().mean() + \
           (img_chw[:, 1:, :] - img_chw[:, :-1, :]).abs().mean()


def _seed_scale_summary(values):
    """Small JSON-safe summary of isotropic seeding scales."""
    values = values.detach().reshape(-1).float()
    if values.numel() == 0:
        return {"count": 0}
    quantile_levels = torch.tensor(
        [0.01, 0.05, 0.50, 0.90, 0.95, 0.99, 1.00],
        device=values.device,
        dtype=values.dtype,
    )
    quantile_values = torch.quantile(values, quantile_levels)
    return {
        "count": int(values.numel()),
        "min": float(values.min().item()),
        "mean": float(values.mean().item()),
        "q01": float(quantile_values[0].item()),
        "q05": float(quantile_values[1].item()),
        "q50": float(quantile_values[2].item()),
        "q90": float(quantile_values[3].item()),
        "q95": float(quantile_values[4].item()),
        "q99": float(quantile_values[5].item()),
        "max": float(quantile_values[6].item()),
        "gt_0p1": int((values > 0.1).sum().item()),
        "gt_0p2": int((values > 0.2).sum().item()),
        "gt_0p5": int((values > 0.5).sum().item()),
        "gt_1": int((values > 1.0).sum().item()),
        "gt_2": int((values > 2.0).sum().item()),
    }


def seed_gaussians_for_patching_multiview(
    gaussians,
    cameras,
    mesh_renderer,
    background,
    random_background=False,
    error_threshold=0.05,
    voxel_size_gaussolves=0.01,
    voxel_size_gaussians=0.01,
    gen_num_per_view=500,
    gs_render_fn=None,
    gaussolve_render_fn=None,
    pipe=None,
    use_trained_exp=False,
    separate_sh=False,
    mesh_depth_tolerance=0.05,
    gaussolves=None,
    gaussolve_mode=False,
    mesh_alpha=False,
    min_views=2,
    min_opacity_mass=1e-4,
    seed_depth_mode="conditional",
    seed_depth_render_fn=None,
    ndf_no_mesh_depth_cutoff=False,
    diagnostic_log_path=None,
):
    """
    [Seedfix: error-guided seeding with metric hybrid depth and camera support]
    """
    with torch.no_grad():
        if gs_render_fn is None:
            raise ValueError("seed_gaussians_for_patching_multiview: gs_render_fn cannot be None")
        if mesh_renderer is None:
            raise ValueError("seed_gaussians_for_patching_multiview: mesh_renderer cannot be None")
        if gaussolve_mode and gaussolve_render_fn is None:
            raise ValueError(
                "seed_gaussians_for_patching_multiview: gaussolve_render_fn cannot be None when gaussolve_mode=True"
            )
        if seed_depth_mode == "conditional" and seed_depth_render_fn is None:
            raise ValueError("conditional seedfix requires a renderer exposing front/total transmittance")
        if int(min_views) < 1:
            raise ValueError("seedfix min_views must be >= 1")
        if float(min_opacity_mass) <= 0.0:
            raise ValueError("seedfix min_opacity_mass must be > 0")
        if seed_depth_mode not in {"raw", "conditional"}:
            raise ValueError("seedfix seed_depth_mode must be raw or conditional")

        base_device = gaussians.get_xyz.device
        base_dtype = gaussians.get_xyz.dtype

        num_points = gaussians.get_xyz.shape[0]
        all_new_xyz = []
        all_new_rgb = []
        all_new_camera_ids = []
        # 额外：只用 mesh 深度的一套 xyz
        all_new_xyz_mesh = []
        all_new_camera_ids_mesh = []
        seedfix_report = {
            "contract": {
                "depth_type": "linear camera-z",
                "moment": "a*(M_front+T_front*mask*z_mesh)+(1-a)*M_total",
                "mass": "a*((1-T_front)+T_front*mask)+(1-a)*(1-T_total)",
                "depth": "moment/effective_mass",
                "support": "distinct camera IDs per voxel",
                "support_filter_scope": "main_and_mesh_ndf",
            },
            "settings": {
                "min_views": int(min_views),
                "min_opacity_mass": float(min_opacity_mass),
                "seed_depth_mode": str(seed_depth_mode),
                "voxel_size_gaussians": float(voxel_size_gaussians),
                "voxel_size_gaussolves": float(voxel_size_gaussolves),
                "gen_num_per_view": int(gen_num_per_view),
                "ndf_no_mesh_depth_cutoff": bool(ndf_no_mesh_depth_cutoff),
            },
            "views": [],
        }
        
        view_bar = tqdm(cameras, desc="Error-guided Seeding", dynamic_ncols=True)
        for cam_idx, viewpoint_cam in enumerate(view_bar):
            # Ensure bg is on the same device/dtype as gaussians (renderers expect CUDA bg)
            if random_background:
                bg = torch.rand((3,), device=base_device, dtype=base_dtype)
            else:
                bg = background.to(device=base_device, dtype=base_dtype)
            
            pos, verts_view, rast, mask = mesh_renderer.rasterize(viewpoint_cam)
            depth_linear, depth_normalized = mesh_renderer.depth_from_rast(viewpoint_cam, verts_view, rast, mask, normalize=True)

            # Step 2.1: Gaussian rendering with mesh depth (for CUDA-level fusion)
            # Note: mesh_depth is passed but CUDA code needs to be modified to use it
            render_pkg = gs_render_fn(
                viewpoint_cam, gaussians, pipe, bg, 
                use_trained_exp=use_trained_exp, 
                separate_sh=separate_sh,
                mesh_depth=depth_linear,
                mesh_depth_tolerance=mesh_depth_tolerance,
                output_inverse_depth = False
            )

            # Step 2.2: new thread to render mesh color
            mesh_rendered = mesh_renderer.color_from_rast(viewpoint_cam, pos, rast, mask, bg_color = bg, enable_antialiasing = True)
            # Three-layer blending formula: C_final = alpha_eff * (C_f + T_f * C_m) + (1.0 - alpha_eff) * C_g
            # Get components
            Cf = render_pkg["render"]  # C_f: front-facing GS [3, H, W]
            Cg = render_pkg["render_all"]  # C_g: total GS [3, H, W]
            Cm = mesh_rendered  # C_m: mesh color [3, H, W]
            Tf = render_pkg["alpha"]  # T_f: transmittance of front-facing GS [1, H, W]
            Df = render_pkg["depth"]  # historical raw front depth moment
            Dg = render_pkg["depth_total"]  # historical raw total depth moment
                        
            if gaussolve_mode is True:
                gaussolve_pkg = gaussolve_render_fn(
                    viewpoint_cam, gaussolves, pipe, torch.zeros_like(bg, device=base_device, dtype=base_dtype),
                    use_trained_exp=use_trained_exp, 
                    separate_sh=separate_sh,
                )

            # Get alpha_eff from mesh renderer (dissolve field)
            if gaussolve_mode is True:
                dissolve_alpha = gaussolve_pkg["render"][:1].clamp(0, 1)  # [1, H, W]
                alpha_eff = 1.0 - dissolve_alpha  # alpha_eff: 1 = keep mesh, 0 = dissolve
            else:
                # Get alpha from mesh renderer if available
                if mesh_alpha and hasattr(mesh_renderer, 'get_alpha_map'):
                    alpha_eff = mesh_renderer.get_alpha_map(viewpoint_cam, pos, rast, mask)  # [1, H, W]
                else:
                    alpha_eff = torch.ones_like(Tf)  # Default: always keep mesh
            
            # Final blended image: C_final = alpha_eff * (C_f + T_f * C_m) + (1.0 - alpha_eff) * C_g
            image = alpha_eff * (Cf + Tf * Cm) + (1.0 - alpha_eff) * Cg
            if seed_depth_mode == "raw":
                # Exact historical control: old renderer, old numerator, no
                # second renderer and no opacity-mass threshold dependency.
                depth = alpha_eff * (Df + Tf * depth_linear * mask) + (1.0 - alpha_eff) * Dg
                effective_mass = torch.full_like(depth, float("nan"))
                depth_valid = torch.isfinite(depth) & (depth > 0.0)
            else:
                # Isolated renderer is used only by the corrected conditional
                # seeding path. Ordinary training renders/losses remain paper code.
                seed_depth_pkg = seed_depth_render_fn(
                    viewpoint_cam, gaussians, pipe, bg,
                    use_trained_exp=use_trained_exp,
                    separate_sh=separate_sh,
                    mesh_depth=depth_linear,
                    mesh_depth_tolerance=mesh_depth_tolerance,
                    output_inverse_depth=False,
                )
                depth, effective_mass, depth_valid = hybrid_depth_for_seeding(
                    seed_depth_pkg["depth_moment_front"],
                    seed_depth_pkg["transmittance_front"],
                    depth_linear,
                    mask,
                    alpha_eff,
                    seed_depth_pkg["depth_moment_total"],
                    seed_depth_pkg["transmittance_total"],
                    mode="conditional",
                    min_opacity_mass=min_opacity_mass,
                )
            # calculate error map (use joint render if available)
            gt_image = viewpoint_cam.original_image.to(device=base_device, dtype=base_dtype)
            # Use float32 for stable pdf/cdf + torch.searchsorted compatibility
            error = torch.abs(image - gt_image).mean(dim=0).flatten().float()
            error[error < float(error_threshold)] = 0.0
            
            if error.sum() == 0:
                seedfix_report["views"].append({
                    "camera_index": int(cam_idx),
                    "status": "no_error_samples",
                })
                continue
                
            # probability density function, Monte Carlo sampling
            pdf = error / error.sum()
            cdf = torch.cumsum(pdf, dim=0)
            
            samples = torch.rand(gen_num_per_view, device=base_device, dtype=torch.float32)
            idxs = torch.searchsorted(cdf, samples)
            # prevent searchsorted from returning index equal to len(cdf)
            idxs = torch.clamp(idxs, 0, cdf.numel() - 1)
            idxs = torch.unique(idxs)
            
            H, W = gt_image.shape[1], gt_image.shape[2]

            v_idx = torch.div(idxs, W, rounding_mode='floor')
            u_idx = idxs % W

            # use *fused* depth for back-projection（保持原逻辑不变）
            if depth.dim() == 3:
                depth = depth.squeeze(0)
                effective_mass = effective_mass.squeeze(0)
                depth_valid = depth_valid.squeeze(0)
            
            z_valid = depth[v_idx, u_idx]
            sampled_mass = effective_mass[v_idx, u_idx]
            sampled_valid = depth_valid[v_idx, u_idx]

            # Reject low-mass/non-finite rays instead of back-projecting them.
            valid_mask = sampled_valid & torch.isfinite(z_valid) & (z_valid > 1e-5)
            seedfix_report["views"].append({
                "camera_index": int(cam_idx),
                "status": "sampled",
                "unique_error_pixels": int(idxs.numel()),
                "depth_mode": str(seed_depth_mode),
                "accepted_metric_depth": int(valid_mask.sum().item()),
                "rejected_by_selected_depth_contract": int((~valid_mask).sum().item()),
                "sampled_below_min_opacity_mass": (
                    int((sampled_mass < float(min_opacity_mass)).sum().item())
                    if seed_depth_mode == "conditional" else None
                ),
                "sampled_mass_min": (
                    float(sampled_mass.min().item()) if seed_depth_mode == "conditional" else None
                ),
                "sampled_mass_mean": (
                    float(sampled_mass.mean().item()) if seed_depth_mode == "conditional" else None
                ),
                "sampled_mass_max": (
                    float(sampled_mass.max().item()) if seed_depth_mode == "conditional" else None
                ),
            })
            fovX = viewpoint_cam.FoVx
            fovY = viewpoint_cam.FoVy
            c2w = torch.inverse(viewpoint_cam.world_view_transform).to(device=base_device, dtype=base_dtype)
            p_world_mesh, valid_mask_mesh = backproject_sampled_mesh_depth(
                u_idx,
                v_idx,
                depth_linear,
                mask,
                width=W,
                height=H,
                fov_x=fovX,
                fov_y=fovY,
                camera_to_world=c2w,
            )
            if p_world_mesh.shape[0] > 0:
                all_new_xyz_mesh.append(p_world_mesh)
                all_new_camera_ids_mesh.append(torch.full(
                    (p_world_mesh.shape[0],), int(cam_idx), device=base_device, dtype=torch.long
                ))

            if not valid_mask.any():
                continue
                
            u_valid = u_idx[valid_mask]
            v_valid = v_idx[valid_mask]
            z_valid = z_valid[valid_mask]

            u_ndc = (u_valid.to(base_dtype) + 0.5) / W * 2.0 - 1.0
            v_ndc = (v_valid.to(base_dtype) + 0.5) / H * 2.0 - 1.0
            
            x_cam = u_ndc * z_valid * math.tan(fovX * 0.5)
            y_cam = v_ndc * z_valid * math.tan(fovY * 0.5)
            p_cam = torch.stack(
                [x_cam, y_cam, z_valid.to(base_dtype), torch.ones_like(z_valid, device=base_device, dtype=base_dtype)],
                dim=-1
            )
            
            # back to world coordinates via camera extrinsic inverse matrix
            p_world = (p_cam @ c2w)[:, :3].to(device=base_device, dtype=base_dtype)
            
            all_new_xyz.append(p_world)
            all_new_camera_ids.append(torch.full(
                (p_world.shape[0],), int(cam_idx), device=base_device, dtype=torch.long
            ))

            # 记录对应像素颜色，用于后面初始化 feature
            rgb_samples = gt_image[:, v_valid, u_valid].permute(1, 0)  # [N,3]
            all_new_rgb.append(rgb_samples.to(device=base_device, dtype=base_dtype))
            view_bar.set_postfix({
                "main_added": int(valid_mask.sum().item()),
                "ndf_added": int(valid_mask_mesh.sum().item()),
            })
            
        view_bar.close()
        
        if len(all_new_xyz) == 0:
            all_pts = torch.empty((0, 3), device=base_device, dtype=base_dtype)
            all_rgb = torch.empty((0, 3), device=base_device, dtype=base_dtype)
            all_camera_ids = torch.empty((0,), device=base_device, dtype=torch.long)
        else:
            all_pts = torch.cat(all_new_xyz, dim=0).to(device=base_device, dtype=base_dtype)
            all_rgb = torch.cat(all_new_rgb, dim=0).to(device=base_device, dtype=base_dtype)
            all_camera_ids = torch.cat(all_new_camera_ids, dim=0).to(device=base_device)

        # 3D voxel downsampling（同时对位置和颜色做聚合）
        voxel_size = voxel_size_gaussians if voxel_size_gaussians > 0 else 0.01
        # prune depth based on min_views
        quantized_idx = torch.round(all_pts / voxel_size).to(torch.long)
        unique_idx, inverse = torch.unique(quantized_idx, dim=0, return_inverse=True)

        N_voxels_raw = unique_idx.shape[0]
        if N_voxels_raw > 0:
            print(f"3D Downsampling: Reduced from {all_pts.shape[0]} to {N_voxels_raw} points!")
        counts = torch.bincount(inverse, minlength=N_voxels_raw).clamp_min(1).to(device=base_device)
        counts = counts.view(-1, 1).to(device=base_device, dtype=base_dtype)

        new_xyz_sum = torch.zeros((N_voxels_raw, 3), device=base_device, dtype=base_dtype)
        new_xyz_sum.scatter_add_(0, inverse.unsqueeze(-1).expand(-1, 3), all_pts)
        new_xyz = new_xyz_sum / counts  # [N_new, 3]

        new_rgb_sum = torch.zeros((N_voxels_raw, 3), device=base_device, dtype=base_dtype)
        new_rgb_sum.scatter_add_(0, inverse.unsqueeze(-1).expand(-1, 3), all_rgb)
        new_rgb = new_rgb_sum / counts  # [N_new, 3], 0-1

        camera_support = distinct_camera_support(inverse, all_camera_ids, N_voxels_raw)
        support_keep = camera_support >= int(min_views)
        support_values, support_counts = torch.unique(camera_support, return_counts=True)
        support_histogram = {
            str(int(value.item())): int(count.item())
            for value, count in zip(support_values, support_counts)
        }
        # Preserve the paper scale initialization as an independent variable:
        # calculate 3-NN scales over every pre-filter voxel, then filter all
        # seed tensors with the same support mask.
        if N_voxels_raw > 0:
            all_xyz_for_scale = torch.cat([gaussians.get_xyz.detach(), new_xyz.detach()], dim=0)
            dist2_all = torch.clamp_min(distCUDA2(all_xyz_for_scale), 1e-7)
            dist2_voxels_raw = dist2_all[-N_voxels_raw:]
            base_scale_raw = (torch.sqrt(dist2_voxels_raw)[..., None].repeat(1, 3)) / 2.0
        else:
            # Avoid Python's ``[-0:] == [0:]`` semantics: no main voxels
            # means no main scales, while the NDF mesh-depth branch continues.
            base_scale_raw = torch.empty((0, 3), device=base_device, dtype=base_dtype)

        new_xyz = new_xyz[support_keep]
        new_rgb = new_rgb[support_keep]
        base_scale = base_scale_raw[support_keep]
        N_new = int(support_keep.sum().item())
        print(
            f"Seedfix camera support >= {int(min_views)}: "
            f"kept {N_new} / {N_voxels_raw} voxels"
        )
        seedfix_report["summary"] = {
            "raw_metric_depth_samples": int(all_pts.shape[0]),
            "raw_voxels": int(N_voxels_raw),
            "accepted_voxels": int(N_new),
            "rejected_by_camera_support": int(N_voxels_raw - N_new),
            "camera_support_histogram": support_histogram,
            "initial_scale_all_voxels": _seed_scale_summary(base_scale_raw[:, 0]),
            "initial_scale_accepted_voxels": _seed_scale_summary(base_scale[:, 0]),
        }
        if N_new == 0:
            seedfix_report["summary"]["main_gaussian_seed_status"] = (
                "empty_after_distinct_camera_support; mesh_depth_ndf_branch_uses_same_distinct_camera_support"
            )

        C = gaussians.get_features.shape[1]
        alpha_color = 1.0 # 或 0.05
        new_features_dc = torch.zeros((N_new, 1, 3), device=base_device, dtype=base_dtype)
        new_features_dc[:, 0, :] = alpha_color * (new_rgb - 0.5) / C0
        new_features_rest = torch.zeros((N_new, C - 1, 3), device=base_device, dtype=base_dtype)
        # 使用“普通初始化”的 scale（不再基于 distCUDA2 的最近邻距离）
        # 原逻辑（基于已有点 + 新点的最近邻距离）：
        new_scaling = gaussians.scaling_inverse_activation(base_scale)
        # 现在改为固定的各向同性初始尺度，交由后续训练自动调整。
        # base_scale = torch.ones((N_new, 3), device=base_device, dtype=base_dtype) * 0.01
        # new_scaling = gaussians.scaling_inverse_activation(base_scale)
        new_rotation = torch.zeros((N_new, 4), device=base_device, dtype=base_dtype)
        new_rotation[:, 0] = 1.0
        new_opacities = gaussians.inverse_opacity_activation(
            torch.ones((N_new, 1), device=base_device, dtype=base_dtype) * 0.005
        )
        
        # return a mask of all False (because these points are generated out of thin air, not cloned from the original model)
        source_mask = torch.zeros((num_points,), dtype=torch.bool, device=base_device)
        
        packed = {
            "new_xyz": new_xyz,
            "new_features_dc": new_features_dc,
            "new_features_rest": new_features_rest,
            "new_opacities": new_opacities,
            "new_scaling": new_scaling,
            "new_rotation": new_rotation,
            "new_tmp_radii": torch.zeros((N_new,), device=base_device, dtype=base_dtype),
            "source_mask": source_mask
        }

        # 再做一次体素融合：这次只用 mesh 深度生成的 xyz
        N_mesh_raw = 0
        N_new_mesh = 0
        mesh_support_histogram = {}
        if len(all_new_xyz_mesh) > 0:
            voxel_size = voxel_size_gaussolves if voxel_size_gaussolves > 0 else 0.01
            all_pts_mesh = torch.cat(all_new_xyz_mesh, dim=0).to(device=base_device, dtype=base_dtype)
            all_camera_ids_mesh = torch.cat(all_new_camera_ids_mesh, dim=0).to(device=base_device)
            quantized_idx_mesh = torch.round(all_pts_mesh / voxel_size).to(torch.long)
            unique_idx_mesh, inverse_mesh = torch.unique(quantized_idx_mesh, dim=0, return_inverse=True)

            N_mesh_raw = int(unique_idx_mesh.shape[0])
            if N_mesh_raw > 0:
                print(f"3D Downsampling (mesh depth): Reduced from {all_pts_mesh.shape[0]} to {N_mesh_raw} points!")
                counts_mesh = torch.bincount(inverse_mesh, minlength=N_mesh_raw).clamp_min(1).to(device=base_device)
                counts_mesh = counts_mesh.view(-1, 1).to(device=base_device, dtype=base_dtype)

                new_xyz_sum_mesh = torch.zeros((N_mesh_raw, 3), device=base_device, dtype=base_dtype)
                new_xyz_sum_mesh.scatter_add_(0, inverse_mesh.unsqueeze(-1).expand(-1, 3), all_pts_mesh)
                new_xyz_mesh_raw = new_xyz_sum_mesh / counts_mesh  # [N_mesh_raw, 3]
                mesh_camera_support = distinct_camera_support(
                    inverse_mesh, all_camera_ids_mesh, N_mesh_raw
                )
                mesh_support_keep = mesh_camera_support >= int(min_views)
                support_values_mesh, support_counts_mesh = torch.unique(
                    mesh_camera_support, return_counts=True
                )
                mesh_support_histogram = {
                    str(int(value.item())): int(count.item())
                    for value, count in zip(support_values_mesh, support_counts_mesh)
                }

                # Compute scales over all raw mesh voxels, then apply the
                # exact same support mask to xyz, scale, rotation, and opacity.
                all_xyz_for_scale_mesh = torch.cat(
                    [gaussians.get_xyz.detach(), new_xyz_mesh_raw.detach()], dim=0
                )
                dist2_all_mesh = torch.clamp_min(distCUDA2(all_xyz_for_scale_mesh), 1e-7)
                dist2_mesh_raw = dist2_all_mesh[-N_mesh_raw:]
                base_scale_mesh_raw = (torch.sqrt(dist2_mesh_raw)[..., None].repeat(1, 3)) / 2.0
                new_xyz_mesh = new_xyz_mesh_raw[mesh_support_keep]
                base_scale_mesh = base_scale_mesh_raw[mesh_support_keep]
                N_new_mesh = int(mesh_support_keep.sum().item())
                print(
                    f"Mesh seed camera support >= {int(min_views)}: "
                    f"kept {N_new_mesh} / {N_mesh_raw} voxels"
                )

                packed["mesh_new_xyz"] = new_xyz_mesh
                packed["mesh_new_opacities"] = gaussians.inverse_opacity_activation(
                    torch.ones((N_new_mesh, 1), device=base_device, dtype=base_dtype) * 0.005
                )
                # mesh 分支的 scale 同样使用固定初始化，而不是基于 distCUDA2 的最近邻距离
                # 原逻辑：
                packed["mesh_new_scaling"] = gaussians.scaling_inverse_activation(base_scale_mesh)
                # base_scale_mesh = torch.ones((N_new_mesh, 3), device=base_device, dtype=base_dtype) * 0.01
                # packed["mesh_new_scaling"] = gaussians.scaling_inverse_activation(base_scale_mesh)
                packed["mesh_new_rotation"] = torch.zeros((N_new_mesh, 4), device=base_device, dtype=base_dtype)
                packed["mesh_new_rotation"][:, 0] = 1.0  # identity quaternion (was missing -> zero quat broke seeded gaussolves)
                packed["mesh_new_tmp_radii"] = torch.zeros((N_new_mesh,), device=base_device, dtype=base_dtype)
        seedfix_report["summary"]["mesh_seed_raw_voxels"] = int(N_mesh_raw)
        seedfix_report["summary"]["mesh_seed_accepted_voxels"] = int(N_new_mesh)
        seedfix_report["summary"]["mesh_seed_rejected_by_camera_support"] = int(N_mesh_raw - N_new_mesh)
        seedfix_report["summary"]["mesh_seed_camera_support_histogram"] = mesh_support_histogram
        write_json_atomic(diagnostic_log_path, seedfix_report)
        return packed
