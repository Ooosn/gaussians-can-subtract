import torch
from simple_knn._C import distCUDA2
from torch import nn
from utils.sh_utils import C0
from tqdm import tqdm
from scene.gaussian_model import GaussianModel

@staticmethod
def collect_bad_mesh_vertices_multiview(
    mesh_renderer,
    cameras,
    source_gaussians: GaussianModel,
    gs_render_fn,
    pipe,
    background,
    random_background=False,
    use_trained_exp=False,
    separate_sh=False,
    mesh_depth_tolerance_dissolve=0.01,
    loss_threshold_dissolve=0.05,
    min_views=1,
):
    """
    多视角统计 mesh 的坏区域，并将对应像素覆盖到的三角面顶点累计为坏顶点掩码。

    判定优先级：
    1) 深度优先：mesh 明显前凸（mesh 深度 + 容差 < 参考深度）直接记坏点；
    2) 深度接近时（|mesh 深度 - 参考深度| <= 容差），再看颜色误差，若误差超过阈值也记坏点。

    参考深度来源优先级：
    1) 若提供 source_gaussians，则使用 3DGS 渲染得到的深度；
    2) 若未提供 3DGS，但相机带有单目 inverse depth（invdepthmap + depth_mask），
       则将其转成深度并与 mesh 深度比较；
    3) 若两者都没有，则返回 None，后续初始化应回退为“全顶点初始化”。
    """


    verts = mesh_renderer.verts
    faces = mesh_renderer.faces
    num_verts = verts.shape[0]
    bad_votes = torch.zeros((num_verts,), dtype=torch.int32, device=verts.device)

    # 判定使用哪种深度来源
    use_gs_depth = source_gaussians is not None
    use_mono_depth = False
    if not use_gs_depth:
        # 尝试从第一个相机判断是否有单目深度
        if isinstance(cameras, (list, tuple)) and len(cameras) > 0:
            first_cam = cameras[0]
        else:
            first_cam = cameras
        if first_cam is not None and hasattr(first_cam, "invdepthmap") and hasattr(first_cam, "depth_mask"):
            use_mono_depth = True

    # 如果既没有 3DGS 模型也没有单目深度，则直接回退为“全顶点初始化”
    if not use_gs_depth and not use_mono_depth:
        print(
            "[collect_bad_mesh_vertices_multiview] "
            "No 3DGS model or monocular depth provided. Fallback to full-vertex initialization."
        )
        return None

    view_bar = tqdm(cameras, desc="Collect dissolve vertices", dynamic_ncols=True)
    for viewpoint_cam in view_bar:
        bg = torch.rand((3), device=background.device) if random_background else background

        pos, verts_view, rast, mesh_mask = mesh_renderer.rasterize(viewpoint_cam)
        mesh_rgb = mesh_renderer.color_from_rast(
            viewpoint_cam, pos, rast, mesh_mask, bg_color=bg, enable_antialiasing=True
        )
        depth_mesh_linear, _ = mesh_renderer.depth_from_rast(
            viewpoint_cam, verts_view, rast, mesh_mask, normalize=True
        )

        # mesh 深度与可见性掩码
        depth_mesh = depth_mesh_linear.squeeze(0) if depth_mesh_linear.dim() == 3 else depth_mesh_linear
        mask2d = mesh_mask.squeeze(0) if mesh_mask.dim() == 3 else mesh_mask

        # 选取参考深度：优先 3DGS，其次单目 inverse depth
        if use_gs_depth:
            gs_pkg = gs_render_fn(
                viewpoint_cam,
                source_gaussians,
                pipe,
                bg,
                use_trained_exp=use_trained_exp,
                separate_sh=separate_sh,
                gs_depth_is_inverse=False,  # 这里得到的是常规深度
            )
            depth_ref = gs_pkg["depth"]
            depth_ref = depth_ref.squeeze(0) if depth_ref.dim() == 3 else depth_ref
            valid_ref_mask = mask2d
        else:
            # 使用相机携带的单目 inverse depth，格式参考 triple_train.py L489-L490
            mono_invdepth = viewpoint_cam.invdepthmap.to(depth_mesh.device)
            depth_mask = viewpoint_cam.depth_mask.to(depth_mesh.device)

            if mono_invdepth.dim() == 3:
                mono_invdepth = mono_invdepth.squeeze(0)
            if depth_mask.dim() == 3:
                depth_mask = depth_mask.squeeze(0)

            # 用户说明：单目深度是“系数匹配过的 inverse”，因此这里将其转为深度做比较
            depth_ref = 1.0 / (mono_invdepth + 1e-6)
            valid_ref_mask = mask2d & depth_mask.bool()

        gt_image = viewpoint_cam.original_image.to(mesh_rgb.device)
        per_pixel_loss = torch.abs(mesh_rgb - gt_image).mean(dim=0)
        if viewpoint_cam.mask is not None:
            cam_mask = viewpoint_cam.mask.to(per_pixel_loss.device)
            if cam_mask.dim() == 3:
                cam_mask = cam_mask.squeeze(0)
            per_pixel_loss = per_pixel_loss * cam_mask

        # 深度优先：明显前凸直接判坏
        depth_bad_mask = valid_ref_mask & (depth_mesh + mesh_depth_tolerance_dissolve < depth_ref)
        # 深度接近时再看 loss
        depth_close_mask = valid_ref_mask & (torch.abs(depth_mesh - depth_ref) <= mesh_depth_tolerance_dissolve)
        loss_bad_mask = per_pixel_loss > loss_threshold_dissolve
        bad_pixel_mask = depth_bad_mask | (depth_close_mask & loss_bad_mask)
        if bad_pixel_mask.sum() == 0:
            continue

        tri_ids = rast[..., 3].long().squeeze(0) - 1
        tri_ids = tri_ids[bad_pixel_mask]
        tri_ids = tri_ids[(tri_ids >= 0) & (tri_ids < faces.shape[0])]
        if tri_ids.numel() == 0:
            continue

        bad_vert_ids = faces[tri_ids].reshape(-1).unique()
        bad_votes[bad_vert_ids] += 1
        view_bar.set_postfix(
            {
                "pixels": int(bad_pixel_mask.sum().item()),
                "depth_bad": int(depth_bad_mask.sum().item()),
                "close_loss_bad": int((depth_close_mask & loss_bad_mask).sum().item()),
                "verts": int((bad_votes >= min_views).sum().item()),
            }
        )
    view_bar.close()

    return bad_votes >= min_views


def collect_bad_mesh_faces_multiview_first_stage(
    mesh_renderer,
    cameras,
    source_gaussians: GaussianModel,
    gs_render_fn,
    pipe,
    background,
    random_background=False,
    use_trained_exp=False,
    separate_sh=False,
    mesh_depth_threshold_1st=2.0,
    mesh_depth_threshold_2nd=5.0,
    min_views_1st=5,
    min_views_2nd=1,
):
    """
    两阶段删面策略：
    1) 中等深度差阈值 + 多视角投票（稳定删）
    2) 大深度差阈值 + 少视角投票（强异常快删）
    最终删除掩码为两路结果并集。
    """
    faces = mesh_renderer.faces
    num_faces = faces.shape[0]
    bad_face_votes_1st = torch.zeros((num_faces,), dtype=torch.int32, device=faces.device)
    bad_face_votes_2nd = torch.zeros((num_faces,), dtype=torch.int32, device=faces.device)

    view_bar = tqdm(cameras, desc="Collect removable faces", dynamic_ncols=True)
    for viewpoint_cam in view_bar:
        bg = torch.rand((3), device=background.device) if random_background else background

        _, verts_view, rast, mesh_mask = mesh_renderer.rasterize(viewpoint_cam)
        depth_mesh_linear, _ = mesh_renderer.depth_from_rast(
            viewpoint_cam, verts_view, rast, mesh_mask, normalize=True
        )
        gs_pkg = gs_render_fn(
            viewpoint_cam,
            source_gaussians,
            pipe,
            bg,
            use_trained_exp=use_trained_exp,
            separate_sh=separate_sh,
            gs_depth_is_inverse=False,
        )

        depth_mesh = depth_mesh_linear.squeeze(0) if depth_mesh_linear.dim() == 3 else depth_mesh_linear
        depth_gs = gs_pkg["depth"].squeeze(0) if gs_pkg["depth"].dim() == 3 else gs_pkg["depth"]
        mask2d = mesh_mask.squeeze(0) if mesh_mask.dim() == 3 else mesh_mask

        depth_abs_diff = torch.abs(depth_mesh - depth_gs)
        removable_pixel_mask_1st = mask2d & (depth_abs_diff > mesh_depth_threshold_1st)
        removable_pixel_mask_2nd = mask2d & (depth_abs_diff > mesh_depth_threshold_2nd)
        if removable_pixel_mask_1st.sum() == 0 and removable_pixel_mask_2nd.sum() == 0:
            continue

        tri_ids = rast[..., 3].long().squeeze(0) - 1
        tri_ids_1st = tri_ids[removable_pixel_mask_1st]
        tri_ids_1st = tri_ids_1st[(tri_ids_1st >= 0) & (tri_ids_1st < num_faces)]
        if tri_ids_1st.numel() > 0:
            bad_face_ids_1st = tri_ids_1st.unique()
            bad_face_votes_1st[bad_face_ids_1st] += 1

        tri_ids_2nd = tri_ids[removable_pixel_mask_2nd]
        tri_ids_2nd = tri_ids_2nd[(tri_ids_2nd >= 0) & (tri_ids_2nd < num_faces)]
        if tri_ids_2nd.numel() > 0:
            bad_face_ids_2nd = tri_ids_2nd.unique()
            bad_face_votes_2nd[bad_face_ids_2nd] += 1

        removable_mask_1st = bad_face_votes_1st >= min_views_1st
        removable_mask_2nd = bad_face_votes_2nd >= min_views_2nd
        removable_mask_union = removable_mask_1st | removable_mask_2nd
        view_bar.set_postfix(
            {
                "pix1": int(removable_pixel_mask_1st.sum().item()),
                "pix2": int(removable_pixel_mask_2nd.sum().item()),
                "f1": int(removable_mask_1st.sum().item()),
                "f2": int(removable_mask_2nd.sum().item()),
                "faces": int(removable_mask_union.sum().item()),
            }
        )
    view_bar.close()

    return (bad_face_votes_1st >= min_views_1st) | (bad_face_votes_2nd >= min_views_2nd)


def apply_face_prune_mask(mesh_renderer, remove_face_mask: torch.Tensor):
    """
    按布尔掩码删除 mesh face，并刷新 renderer 内部 face 缓冲。
    """
    if remove_face_mask is None:
        return 0
    remove_face_mask = remove_face_mask.bool().to(device=mesh_renderer.faces.device)
    if remove_face_mask.shape[0] != mesh_renderer.faces.shape[0]:
        raise ValueError("remove_face_mask size mismatch with mesh faces")

    keep_face_mask = ~remove_face_mask
    removed = int(remove_face_mask.sum().item())
    mesh_renderer.faces = mesh_renderer.faces[keep_face_mask]
    mesh_renderer.faces_contiguous = mesh_renderer.faces.contiguous()
    mesh_renderer.ranges = None
    return removed


@staticmethod
def gaussolve_init(gaussians: GaussianModel, mesh_renderer, vertex_mask=None):
    """
    用指定顶点掩码初始化单通道溶解高斯；若掩码为空则回退为全顶点初始化。
    """
    if mesh_renderer is None:
        raise ValueError("mesh_renderer cannot be None")

    verts = mesh_renderer.verts
    if vertex_mask is None:
        fused_point_cloud = verts.clone()
    else:
        vertex_mask = vertex_mask.bool().to(device=verts.device)
        if vertex_mask.shape[0] != verts.shape[0]:
            raise ValueError("vertex_mask size mismatch with mesh vertices")
        fused_point_cloud = verts[vertex_mask]
        if fused_point_cloud.shape[0] == 0:
            print("No bad vertices found for dissolve init. Fallback to full mesh vertices.")
            fused_point_cloud = verts.clone()

    num_points = fused_point_cloud.shape[0]
    features = torch.zeros(
        (num_points, 1, (gaussians.max_sh_degree + 1) ** 2),
        dtype=torch.float32,
        device=verts.device,
    )
    target_alpha = 0.05
    features[:, 0, 0] = (target_alpha - 0.5) / C0
    dist2 = torch.clamp_min(distCUDA2(fused_point_cloud), 0.0000001)
    base_scales = torch.sqrt(dist2)[..., None].repeat(1, 3)
    scales = gaussians.scaling_inverse_activation(base_scales)
    rots = torch.zeros((num_points, 4), device=verts.device, dtype=torch.float32)
    rots[:, 0] = 1.0
    init_opacity = 0.01
    current_opacity = init_opacity * torch.ones((num_points, 1), dtype=torch.float32, device=verts.device)
    target_opacity = torch.min(current_opacity, torch.ones_like(current_opacity) * init_opacity)
    opacities = gaussians.inverse_opacity_activation(target_opacity)

    gaussians._xyz = nn.Parameter(fused_point_cloud.requires_grad_(True))
    gaussians._features_dc = nn.Parameter(features[:, :, 0:1].transpose(1, 2).contiguous().requires_grad_(True))
    gaussians._features_rest = nn.Parameter(features[:, :, 1:].transpose(1, 2).contiguous().requires_grad_(True))
    gaussians._scaling = nn.Parameter(scales.requires_grad_(True))
    gaussians._rotation = nn.Parameter(rots.requires_grad_(True))
    gaussians._opacity = nn.Parameter(opacities.requires_grad_(True))
    gaussians.max_radii2D = torch.zeros((num_points,), device=verts.device)
    gaussians.xyz_gradient_accum = torch.zeros((num_points, 1), device=verts.device)
    gaussians.denom = torch.zeros((num_points, 1), device=verts.device)
    gaussians.visibility = torch.zeros((num_points,), device=verts.device)

    print(f"Successfully initialized {num_points} dissolve gaussians from selected mesh vertices")
