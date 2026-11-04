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

import torch
import math
from absolute_depth_3dgs_onesh import GaussianRasterizationSettings, GaussianRasterizer
from scene.gaussian_model import GaussianModel
from utils.sh_utils import eval_sh



def gaussolve_render(viewpoint_camera, pc: GaussianModel, pipe, bg_color: torch.Tensor,
                     scaling_modifier: float = 1.0,
                     separate_sh: bool = False,
                     override_color=None,
                     use_trained_exp: bool = False,
                     gs_depth_is_inverse: bool = True,
                     mesh_depth=None,
                     mesh_depth_tolerance: float = 0.01,
                     clamp_output: bool = True):
    """
    Render the scene. 
    
    Background tensor (bg_color) must be on GPU!

    Callers that need an NDF-only no-cutoff ablation must pass
    ``mesh_depth=None``.  The binding converts that to an empty tensor, which
    the CUDA kernel treats as ``use_depth_image`` false.  A zero-valued depth
    tensor must not be used: it is a real depth image and would classify
    nearly every Gaussian as behind the mesh.
    """
 
    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    screenspace_points = torch.zeros_like(pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda") + 0
    try:
        screenspace_points.retain_grad()
    except:
        pass

    # Set up rasterization configuration
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

    raster_settings = GaussianRasterizationSettings(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform,
        projmatrix=viewpoint_camera.full_proj_transform,
        sh_degree=pc.active_sh_degree,
        campos=viewpoint_camera.camera_center,
        prefiltered=False,
        debug=pipe.debug,
        antialiasing = False,
        output_inverse_depth=gs_depth_is_inverse,
        mesh_depth_tolerance=mesh_depth_tolerance
        # znear = viewpoint_camera.znear,
        # zfar = viewpoint_camera.zfar,
    )

    rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    means3D = pc.get_xyz
    means2D = screenspace_points
    opacity = pc.get_opacity

    # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
    # scaling / rotation by the rasterizer.
    scales = None
    rotations = None
    cov3D_precomp = None

    if pipe.compute_cov3D_python:
        cov3D_precomp = pc.get_covariance(scaling_modifier)
    else:
        scales = pc.get_scaling
        rotations = pc.get_rotation

    # If precomputed colors are provided, use them. Otherwise, if it is desired to precompute colors
    # from SHs in Python, do it. If not, then SH -> RGB conversion will be done by rasterizer.
    shs = None
    colors_precomp = None
    if override_color is None:
        if pipe.convert_SHs_python:
            feature_channels = pc.get_features.shape[-1]
            shs_view = pc.get_features.transpose(1, 2).view(-1, feature_channels, (pc.max_sh_degree+1)**2)
            dir_pp = (pc.get_xyz - viewpoint_camera.camera_center.repeat(pc.get_features.shape[0], 1))
            dir_pp_normalized = dir_pp/dir_pp.norm(dim=1, keepdim=True)
            sh_values = eval_sh(pc.active_sh_degree, shs_view, dir_pp_normalized)
            colors_precomp = torch.clamp_min(sh_values + 0.5, 0.0)
        else:
            # onesh rasterizer has no separate dc/shs arguments; pass packed SH tensor.
            shs = pc.get_features
    else:
        colors_precomp = override_color

    # Rasterize visible Gaussians to image, obtain their radii (on screen). 
    rendered_image, radii, depth_image = rasterizer(
        means3D = means3D,
        means2D = means2D,
        shs = shs,
        colors_precomp = colors_precomp,
        opacities = opacity,
        scales = scales,
        rotations = rotations,
        cov3D_precomp = cov3D_precomp,
        depth_image = mesh_depth,
        # shadow = None
        )
        
    # Apply exposure to rendered image (training only)
    if use_trained_exp:
        exposure = pc.get_exposure_from_name(viewpoint_camera.image_name)
        rendered_image = torch.matmul(rendered_image.permute(1, 2, 0), exposure[:3, :3]).permute(2, 0, 1) + exposure[:3, 3,   None, None]

    # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
    # They will be excluded from value updates used in the splitting criteria.
    # Keep the historical clamped output by default.  Scalar-field fitting can
    # request the raw value so over-shooting pixels retain a downward gradient.
    if clamp_output:
        rendered_image = rendered_image.clamp(0, 1)
    out = {
        "render": rendered_image[:1],
        "viewspace_points": screenspace_points,
        "visibility_filter" : (radii > 0).nonzero(),
        "radii": radii,
        "depth" : depth_image,
        # "alpha": final_T
        }
    
    return out


def gaussolve_render_inference(viewpoint_camera, pc: GaussianModel, pipe, bg_color: torch.Tensor,
                               scaling_modifier: float = 1.0,
                               separate_sh: bool = False,
                               override_color=None,
                               use_trained_exp: bool = False,
                               gs_depth_is_inverse: bool = True,
                               mesh_depth=None,
                               mesh_depth_tolerance: float = 0.01,
                               clamp_output: bool = True):
    """Bit-exact inference forward without training-only gradient bookkeeping."""
    if torch.is_grad_enabled():
        raise RuntimeError("gaussolve_render_inference requires torch.no_grad()")

    screenspace_points = torch.zeros_like(pc.get_xyz)
    raster_settings = GaussianRasterizationSettings(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=math.tan(viewpoint_camera.FoVx * 0.5),
        tanfovy=math.tan(viewpoint_camera.FoVy * 0.5),
        bg=bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform,
        projmatrix=viewpoint_camera.full_proj_transform,
        sh_degree=pc.active_sh_degree,
        campos=viewpoint_camera.camera_center,
        prefiltered=False,
        debug=pipe.debug,
        antialiasing=False,
        output_inverse_depth=gs_depth_is_inverse,
        mesh_depth_tolerance=mesh_depth_tolerance,
    )
    rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    scales = None
    rotations = None
    cov3D_precomp = None
    if pipe.compute_cov3D_python:
        cov3D_precomp = pc.get_covariance(scaling_modifier)
    else:
        scales = pc.get_scaling
        rotations = pc.get_rotation

    shs = None
    colors_precomp = None
    if override_color is None:
        if pipe.convert_SHs_python:
            feature_channels = pc.get_features.shape[-1]
            shs_view = pc.get_features.transpose(1, 2).view(
                -1, feature_channels, (pc.max_sh_degree + 1) ** 2
            )
            dir_pp = pc.get_xyz - viewpoint_camera.camera_center.repeat(
                pc.get_features.shape[0], 1
            )
            dir_pp_normalized = dir_pp / dir_pp.norm(dim=1, keepdim=True)
            sh_values = eval_sh(pc.active_sh_degree, shs_view, dir_pp_normalized)
            colors_precomp = torch.clamp_min(sh_values + 0.5, 0.0)
        else:
            shs = pc.get_features
    else:
        colors_precomp = override_color

    rendered_image, _, _ = rasterizer(
        means3D=pc.get_xyz,
        means2D=screenspace_points,
        shs=shs,
        colors_precomp=colors_precomp,
        opacities=pc.get_opacity,
        scales=scales,
        rotations=rotations,
        cov3D_precomp=cov3D_precomp,
        depth_image=mesh_depth,
    )

    if use_trained_exp:
        exposure = pc.get_exposure_from_name(viewpoint_camera.image_name)
        rendered_image = (
            torch.matmul(
                rendered_image.permute(1, 2, 0), exposure[:3, :3]
            ).permute(2, 0, 1)
            + exposure[:3, 3, None, None]
        )
    if clamp_output:
        rendered_image = rendered_image.clamp(0, 1)
    return {"render": rendered_image[:1]}

