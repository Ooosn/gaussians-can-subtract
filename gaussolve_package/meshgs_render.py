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
import numpy as np
from concurrent.futures import ThreadPoolExecutor
from mesh_gs import GaussianRasterizationSettings as mesh_gs_settings
from mesh_gs import GaussianRasterizer as mesh_gs_rasterizer
from gaussian_renderer import render as regular_render
from gaussolve_package.gaussolve_render import gaussolve_render
from scene.gaussian_model import GaussianModel
from utils.sh_utils import eval_sh


def _variable_sh_colors(pc, viewpoint_camera):
    """Evaluate only each Gaussian's retained Reduced SH band."""
    degrees = getattr(pc, "_reduced_sh_degrees", None)
    if degrees is None:
        return None
    degrees = degrees.reshape(-1).long()
    if degrees.numel() != pc.get_xyz.shape[0]:
        raise RuntimeError("Reduced SH degree count does not match Gaussian count")
    features = pc.get_features.transpose(1, 2).view(
        -1, 3, (pc.max_sh_degree + 1) ** 2
    )
    directions = pc.get_xyz - viewpoint_camera.camera_center.repeat(
        pc.get_features.shape[0], 1
    )
    directions = directions / directions.norm(dim=1, keepdim=True)
    colours = torch.empty(
        (pc.get_xyz.shape[0], 3),
        dtype=features.dtype,
        device=features.device,
    )
    for degree in range(int(pc.max_sh_degree) + 1):
        mask = degrees == degree
        if mask.any():
            colours[mask] = eval_sh(degree, features[mask], directions[mask])
    return torch.clamp_min(colours + 0.5, 0.0)
from fast_mesh_gs import GaussianRasterizationSettings as fast_mesh_gs_settings
from fast_mesh_gs import GaussianRasterizer as fast_mesh_gs_rasterizer

try:
    from fast_mesh_gs_variable import (
        GaussianRasterizationSettings as variable_fast_mesh_gs_settings,
    )
    from fast_mesh_gs_variable import (
        GaussianRasterizer as variable_fast_mesh_gs_rasterizer,
    )
except (ImportError, OSError):
    variable_fast_mesh_gs_settings = None
    variable_fast_mesh_gs_rasterizer = None

class MeshGsRender:
    def __init__(
        self,
        mesh_renderer,
        mesh_appearance=None,
        synchronous_streams=False,
    ):
        self.mesh_renderer = mesh_renderer
        # Optional Stage-1 view-dependent appearance module.  Keeping this
        # dependency injected (rather than importing the Stage-1 code here)
        # preserves the historical renderer when it is not supplied.
        self.mesh_appearance = mesh_appearance
        self.synchronous_streams = bool(synchronous_streams)
        # Persistent auxiliary streams avoid per-frame stream construction.
        # Separate host workers are intentional: both Gaussian rasterizers
        # contain a host-side sizing step, so issuing them from independent
        # workers allows the residual, mesh-colour, and NDF branches to overlap
        # after the shared mesh raster/depth pass.
        self._mesh_stream = None
        self._ndf_stream = None
        self._parallel_executor = None
        if not self.synchronous_streams and torch.cuda.is_available():
            self._mesh_stream = torch.cuda.Stream()
            self._ndf_stream = torch.cuda.Stream()
            self._parallel_executor = ThreadPoolExecutor(
                max_workers=2, thread_name_prefix="gaussolve-render"
            )

    def close(self):
        executor = self._parallel_executor
        self._parallel_executor = None
        if executor is not None:
            executor.shutdown(wait=True)

    def mesh_color_from_rast(
        self,
        viewpoint_camera,
        pos,
        rast,
        mask,
        *,
        bg_color,
        enable_antialiasing=True,
    ):
        """Render the base texture and, when present, its Stage-1 appearance."""
        mesh_rendered = self.mesh_renderer.color_from_rast(
            viewpoint_camera,
            pos,
            rast,
            mask,
            bg_color=bg_color,
            enable_antialiasing=enable_antialiasing,
        )
        if self.mesh_appearance is not None:
            mesh_rendered = self.mesh_appearance(
                self.mesh_renderer,
                viewpoint_camera,
                rast,
                mask,
                mesh_rendered,
            )
        return mesh_rendered

    def render_wrapper(
        self,
        viewpoint_camera,
        gaussians,
        pipe,
        bg_color,
        scaling_modifier,
        separate_sh,
        override_color=None,
         use_trained_exp=False,
         mesh_depth_tolerance=0.01,
         gaussolves=None,
         gaussolve_mode=False,
         ndf_no_mesh_depth_cutoff=False,
         *,
         return_components=False,
         suppress_residual=False,
         suppress_mesh_color=False,
     ):
        """Render and fuse the mesh/Gaussian branches.

        The ablation switches are applied to branch tensors immediately before
        fusion.  They never zero an already-fused output, and the default path
        remains unchanged.
        """
        if suppress_residual and suppress_mesh_color:
            raise ValueError(
                "suppress_residual and suppress_mesh_color are mutually exclusive"
            )
        if (suppress_residual or suppress_mesh_color) and self.mesh_renderer is None:
            raise ValueError(
                "branch suppression requires a mesh renderer so the fusion gate remains defined"
            )
        mesh_depth = None
        mesh_color = None
        mesh_rast_data = None
        
        if self.mesh_renderer is not None:
            # Rasterize mesh once
            pos, verts_view, rast, mask = self.mesh_renderer.rasterize(viewpoint_camera)
            depth_linear, depth_normalized = self.mesh_renderer.depth_from_rast(viewpoint_camera, verts_view, rast, mask, normalize=True)
            
            # Compute mesh depth
            mesh_depth = depth_linear # [1, H, W]

            if self.synchronous_streams:
                render_pkg = self.render(
                    viewpoint_camera, gaussians, pipe, bg_color,
                    use_trained_exp=use_trained_exp,
                    separate_sh=separate_sh,
                    mesh_depth=mesh_depth,
                    mesh_depth_tolerance=mesh_depth_tolerance,
                )
                if suppress_mesh_color:
                    mesh_rendered = torch.zeros_like(render_pkg["render"])
                else:
                    mesh_rendered = self.mesh_color_from_rast(
                        viewpoint_camera,
                        pos,
                        rast,
                        mask,
                        bg_color=bg_color,
                        enable_antialiasing=True,
                    )
                if gaussolve_mode is True:
                    ndf_mesh_depth = (
                        None if ndf_no_mesh_depth_cutoff else mesh_depth
                    )
                    gaussolve_pkg = gaussolve_render(
                        viewpoint_camera,
                        gaussolves,
                        pipe,
                        torch.zeros_like(bg_color),
                        use_trained_exp=use_trained_exp,
                        separate_sh=separate_sh,
                        mesh_depth=ndf_mesh_depth,
                        mesh_depth_tolerance=mesh_depth_tolerance,
                    )
            else:
                current_stream = torch.cuda.current_stream()
                mesh_stream = self._mesh_stream
                ndf_stream = self._ndf_stream
                executor = self._parallel_executor
                if mesh_stream is None or ndf_stream is None or executor is None:
                    raise RuntimeError("parallel renderer was not initialized on CUDA")

                # Both auxiliary consumers depend only on tensors produced up
                # to this point.  Their waits do not include the residual render
                # enqueued below, so all three branches can overlap.
                mesh_stream.wait_stream(current_stream)
                ndf_stream.wait_stream(current_stream)
                grad_enabled = torch.is_grad_enabled()

                def render_mesh_colour():
                    with torch.set_grad_enabled(grad_enabled), torch.cuda.device(bg_color.device), torch.cuda.stream(mesh_stream):
                        if suppress_mesh_color:
                            return None
                        return self.mesh_color_from_rast(
                            viewpoint_camera,
                            pos,
                            rast,
                            mask,
                            bg_color=bg_color,
                            enable_antialiasing=True,
                        )

                def render_ndf():
                    with torch.set_grad_enabled(grad_enabled), torch.cuda.device(bg_color.device), torch.cuda.stream(ndf_stream):
                        if gaussolve_mode is not True:
                            return None
                        ndf_mesh_depth = (
                            None if ndf_no_mesh_depth_cutoff else mesh_depth
                        )
                        return gaussolve_render(
                            viewpoint_camera,
                            gaussolves,
                            pipe,
                            torch.zeros_like(bg_color),
                            use_trained_exp=use_trained_exp,
                            separate_sh=separate_sh,
                            mesh_depth=ndf_mesh_depth,
                            mesh_depth_tolerance=mesh_depth_tolerance,
                        )

                mesh_future = executor.submit(render_mesh_colour)
                ndf_future = executor.submit(render_ndf)
                render_pkg = self.render(
                    viewpoint_camera, gaussians, pipe, bg_color,
                    use_trained_exp=use_trained_exp,
                    separate_sh=separate_sh,
                    mesh_depth=mesh_depth,
                    mesh_depth_tolerance=mesh_depth_tolerance,
                )
                mesh_rendered = mesh_future.result()
                gaussolve_pkg = ndf_future.result()
                current_stream.wait_stream(mesh_stream)
                current_stream.wait_stream(ndf_stream)
                if mesh_rendered is None:
                    mesh_rendered = torch.zeros_like(render_pkg["render"])
            if gaussolve_mode is True:
                dissolve_alpha = gaussolve_pkg["render"][:1].clamp(0, 1)
            else:
                dissolve_alpha = 0.0
            # Hybrid blend:
            # C = alpha_eff * (C_f + T_f * C_m) + (1 - alpha_eff) * C_g
            alpha_eff = 1.0 - dissolve_alpha
            Cf = render_pkg["render"]
            Cg = render_pkg["render_all"]
            Tf = render_pkg["alpha"]
            Cm = mesh_rendered
            if suppress_residual:
                # Keep mesh geometry/depth and the NDF gate.  Tf=1 makes this
                # branch exactly (1-D)*Cm rather than a post-fusion zero image.
                Cf = torch.zeros_like(Cf)
                Cg = torch.zeros_like(Cg)
                Tf = torch.ones_like(Tf)
            if suppress_mesh_color:
                Cm = torch.zeros_like(Cm)
            render_pkg["render"] = alpha_eff * (Cf + Tf * Cm) + (1.0 - alpha_eff) * Cg
            if return_components:
                render_pkg["hybrid_components"] = {
                    "Cf": Cf,
                    "Cg": Cg,
                    "Tf": Tf,
                    "Cm": Cm,
                    "D": dissolve_alpha if torch.is_tensor(dissolve_alpha) else torch.zeros_like(Tf),
                }
        else:
            render_pkg = regular_render(
                viewpoint_camera, gaussians, pipe, bg_color, 
                use_trained_exp=use_trained_exp, 
                separate_sh=separate_sh
            )
            if return_components:
                cf = render_pkg["render"]
                render_pkg["hybrid_components"] = {
                    "Cf": cf,
                    "Cg": render_pkg.get("render_all", cf),
                    "Tf": render_pkg.get("alpha", torch.ones_like(cf[:1])),
                    "Cm": torch.zeros_like(cf),
                    "D": torch.zeros_like(cf[:1]),
                }
        
        return render_pkg


    def render(self, viewpoint_camera, pc : GaussianModel, pipe, bg_color : torch.Tensor, scaling_modifier = 1.0, \
               separate_sh = False, override_color = None, use_trained_exp=False, mesh_depth=None, mesh_color=None,
               mesh_depth_tolerance=0.01, output_inverse_depth=True):
        """
        Render the scene. 
        
        Background tensor (bg_color) must be on GPU!
        
        Args:
            mesh_depth: Optional mesh depth [1, H, W] tensor for Gaussian-level fusion.
                    If provided, will be passed to CUDA rasterizer for depth-aware rendering.
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
        raster_settings = mesh_gs_settings(
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
            antialiasing=False,
            output_inverse_depth=output_inverse_depth,
            mesh_depth_tolerance=mesh_depth_tolerance,
            # znear = viewpoint_camera.znear,
            # zfar = viewpoint_camera.zfar,
        )

        rasterizer = mesh_gs_rasterizer(raster_settings=raster_settings)

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
                shs_view = pc.get_features.transpose(1, 2).view(-1, 3, (pc.max_sh_degree+1)**2)
                dir_pp = (pc.get_xyz - viewpoint_camera.camera_center.repeat(pc.get_features.shape[0], 1))
                dir_pp_normalized = dir_pp/dir_pp.norm(dim=1, keepdim=True)
                sh2rgb = eval_sh(pc.active_sh_degree, shs_view, dir_pp_normalized)
                colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0)
            else:
                if separate_sh:
                    dc, shs = pc.get_features_dc, pc.get_features_rest
                else:
                    shs = pc.get_features
        else:
            colors_precomp = override_color

        # return color, depth, norm, alpha, radii, extra
        # Rasterize visible Gaussians to image, obtain their radii (on screen). 
        if separate_sh:
            rendered_front, radii, alpha_T, depth, visibility, color_total, invdepth_total = rasterizer(
                means3D = means3D,
                means2D = means2D,
                dc = dc,
                shs = shs,
                colors_precomp = colors_precomp,
                opacities = opacity,
                scales = scales,
                rotations = rotations,
                cov3D_precomp = cov3D_precomp,
                depth_image = mesh_depth,
                )
        else:
            rendered_front, radii, alpha_T, depth, visibility, color_total, invdepth_total = rasterizer(
                means3D = means3D,
                means2D = means2D,
                shs = shs,
                colors_precomp = colors_precomp,
                opacities = opacity,
                scales = scales,
                rotations = rotations,
                cov3D_precomp = cov3D_precomp,
                depth_image = mesh_depth,
                )
            
        # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
        # They will be excluded from value updates used in the splitting criteria.
        rendered_front = rendered_front.clamp(0, 1)
        out = {
            "render": rendered_front[:3],  # C_f
            "viewspace_points": screenspace_points,
            "visibility_filter" : (radii > 0).nonzero(),
            "radii": radii,
            "depth": depth,  # Fixed: was depth_image
            "alpha": alpha_T,
            "render_all": color_total,
            "depth_total": invdepth_total,
            "gs_visibility_under_mesh": visibility,
            }

        return out


    def render_hard(self, viewpoint_camera, pc: GaussianModel, pipe, bg_color: torch.Tensor,
                    scaling_modifier: float = 1.0,
                    separate_sh: bool = False,
                    override_color=None,
                    use_trained_exp: bool = False,
                    mesh_depth=None,
                    mesh_depth_tolerance: float = 0.01,
                    output_inverse_depth: bool = True):
        """Hardz variant: uses mesh_gs_hard rasterizer which fully discards back-of-mesh
        Gaussians (no soft NDF / dissolve-field path). Returns the same dict keys as render()
        for downstream compatibility, but render_all / depth_total are zero placeholders."""

        screenspace_points = torch.zeros_like(pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda") + 0
        try:
            screenspace_points.retain_grad()
        except Exception:
            pass

        tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
        tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)
        raster_settings = mesh_gs_hard_settings(
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
            antialiasing=False,
            output_inverse_depth=output_inverse_depth,
            mesh_depth_tolerance=mesh_depth_tolerance,
        )
        rasterizer = mesh_gs_hard_rasterizer(raster_settings=raster_settings)

        means3D = pc.get_xyz
        means2D = screenspace_points
        opacity = pc.get_opacity
        scales = None
        rotations = None
        cov3D_precomp = None
        if pipe.compute_cov3D_python:
            cov3D_precomp = pc.get_covariance(scaling_modifier)
        else:
            scales = pc.get_scaling
            rotations = pc.get_rotation

        # mesh_gs_hard kernel has a backward shape bug on the SH path
        # (returns grad_sh of shape [N, 0, 3] instead of [N, 16, 3]). We sidestep it
        # by always converting SH -> RGB in Python and passing colors_precomp.
        shs = None
        colors_precomp = None
        if override_color is None:
            shs_view = pc.get_features.transpose(1, 2).view(-1, 3, (pc.max_sh_degree + 1) ** 2)
            dir_pp = (pc.get_xyz - viewpoint_camera.camera_center.repeat(pc.get_features.shape[0], 1))
            dir_pp_normalized = dir_pp / dir_pp.norm(dim=1, keepdim=True)
            sh2rgb = eval_sh(pc.active_sh_degree, shs_view, dir_pp_normalized)
            colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0)
        else:
            colors_precomp = override_color

        color, radii, alpha_T, invdepths = rasterizer(
            means3D=means3D, means2D=means2D, shs=shs,
            colors_precomp=colors_precomp, opacities=opacity,
            scales=scales, rotations=rotations, cov3D_precomp=cov3D_precomp,
            depth_image=mesh_depth,
        )

        color = color.clamp(0, 1)
        # Fill render_all with zeros so downstream code that mixes (1-alpha_eff)*Cg sees no contribution.
        zero_color = torch.zeros_like(color)
        # gs_visibility_under_mesh is a *per-Gaussian* flag in the soft path. The hardz path
        # discards back-of-mesh Gaussians outright, so report all-zero (no Gaussian flagged
        # as "visible only because mesh covers it") — same shape [N] as soft path.
        N = means3D.shape[0]
        zero_per_gauss = torch.zeros(N, device=means3D.device, dtype=torch.float32)
        return {
            "render": color[:3],
            "viewspace_points": screenspace_points,
            "visibility_filter": (radii > 0).nonzero(),
            "radii": radii,
            "depth": invdepths,
            "alpha": alpha_T,
            "render_all": zero_color[:3],
            "depth_total": invdepths,
            "gs_visibility_under_mesh": zero_per_gauss,
        }


    def fast_render(self, viewpoint_camera, pc: GaussianModel, pipe, bg_color: torch.Tensor,
                    scaling_modifier: float = 1.0,
                    separate_sh: bool = False,
                    override_color=None,
                    use_trained_exp: bool = False,
                    mesh_depth=None,
                    mesh_color=None,
                    mesh_depth_tolerance: float = 0.01,
                    output_inverse_depth: bool = True):
        """
        Quick render variant of meshgs render.
        当前实现与 render 完全一致，后续会在 CUDA 中接 quick path。
        """

        screenspace_points = torch.zeros_like(pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda") + 0
        try:
            screenspace_points.retain_grad()
        except Exception:
            pass

        tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
        tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)
        reduced_degrees = getattr(pc, "_reduced_sh_degrees", None)
        use_variable_fast = bool(
            reduced_degrees is not None
            and variable_fast_mesh_gs_settings is not None
            and not separate_sh
            and override_color is None
            and not pipe.convert_SHs_python
        )
        settings_type = (
            variable_fast_mesh_gs_settings
            if use_variable_fast
            else fast_mesh_gs_settings
        )
        rasterizer_type = (
            variable_fast_mesh_gs_rasterizer
            if use_variable_fast
            else fast_mesh_gs_rasterizer
        )
        raster_settings = settings_type(
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
            antialiasing=False,
            output_inverse_depth=output_inverse_depth,
            mesh_depth_tolerance=mesh_depth_tolerance,
        )

        rasterizer = rasterizer_type(raster_settings=raster_settings)

        means3D = pc.get_xyz
        means2D = screenspace_points
        opacity = pc.get_opacity

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
            if use_variable_fast:
                shs = pc.get_features
            elif pipe.convert_SHs_python:
                shs_view = pc.get_features.transpose(1, 2).view(-1, 3, (pc.max_sh_degree + 1) ** 2)
                dir_pp = (pc.get_xyz - viewpoint_camera.camera_center.repeat(pc.get_features.shape[0], 1))
                dir_pp_normalized = dir_pp / dir_pp.norm(dim=1, keepdim=True)
                sh2rgb = eval_sh(pc.active_sh_degree, shs_view, dir_pp_normalized)
                colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0)
            else:
                if separate_sh:
                    dc, shs = pc.get_features_dc, pc.get_features_rest
                else:
                    shs = pc.get_features
        else:
            colors_precomp = override_color

        if separate_sh:
            rendered_front, alpha, rendered_total = rasterizer(
                means3D=means3D,
                means2D=means2D,
                dc=dc,
                shs=shs,
                colors_precomp=colors_precomp,
                opacities=opacity,
                scales=scales,
                rotations=rotations,
                cov3D_precomp=cov3D_precomp,
                depth_image=mesh_depth,
            )
        else:
            rasterizer_arguments = dict(
                means3D=means3D,
                means2D=means2D,
                shs=shs,
                colors_precomp=colors_precomp,
                opacities=opacity,
                scales=scales,
                rotations=rotations,
                cov3D_precomp=cov3D_precomp,
                depth_image=mesh_depth,
            )
            if use_variable_fast:
                rasterizer_arguments["degrees"] = reduced_degrees.reshape(-1).to(
                    device=means3D.device, dtype=torch.int32
                )
            rendered_front, alpha, rendered_total = rasterizer(
                **rasterizer_arguments
            )

        rendered_front = rendered_front.clamp(0, 1)
        rendered_total = rendered_total.clamp(0, 1)
        out = {
            "render": rendered_front[:3],
            "render_all": rendered_total[:3],
            "viewspace_points": screenspace_points,
            "visibility_filter": None,
            "radii": None,
            "depth": None,
            "depth_total": None,
            "alpha": alpha,
            "gs_visibility_under_mesh": None,
        }

        return out

    def prune_render(self, viewpoint_camera, pc: GaussianModel, pipe, bg_color: torch.Tensor,
                     scaling_modifier: float = 1.0,
                     separate_sh: bool = False,
                     override_color=None,
                     use_trained_exp: bool = False,
                     mesh_depth=None,
                     mesh_color=None,
                     mesh_depth_tolerance: float = 0.01,
                     output_inverse_depth: bool = False,
                     *,
                     alpha_eff_override=None,
                     gaussolve_mode: bool = False):
        """
        专门用于自删减的渲染：走 mesh_gs_self_prune 路径，返回 per-Gaussian 贡献度。

        ``alpha_eff_override`` must be the same effective mesh weight used by
        the formal Hybrid blend (``1 - NDF_dissolve_alpha``).  The old
        fallback to a mesh-renderer alpha map was unsafe: that map is not the
        NDF dissolve field and silently made all back-of-mesh Gaussians look
        unused.  Every caller must provide the map explicitly; non-NDF
        callers should pass an all-ones map for the alpha_eff=1 blend.
        """

        if alpha_eff_override is None:
            raise ValueError(
                "prune_render requires an explicit formal Hybrid "
                "alpha_eff_override (1 - NDF dissolve alpha); pass an "
                "all-ones map for non-NDF callers."
            )

        screenspace_points = torch.zeros_like(pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda") + 0
        try:
            screenspace_points.retain_grad()
        except Exception:
            pass

        # 设置 self-prune 版的 raster settings
        tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
        tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)
        raster_settings = mesh_gs_self_prune.GaussianRasterizationSettings(
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
            antialiasing=False,
            output_inverse_depth=output_inverse_depth,
            mesh_depth_tolerance=mesh_depth_tolerance,
        )

        means3D = pc.get_xyz
        means2D = screenspace_points
        opacity = pc.get_opacity

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
                shs_view = pc.get_features.transpose(1, 2).view(-1, 3, (pc.max_sh_degree + 1) ** 2)
                dir_pp = (pc.get_xyz - viewpoint_camera.camera_center.repeat(pc.get_features.shape[0], 1))
                dir_pp_normalized = dir_pp / dir_pp.norm(dim=1, keepdim=True)
                sh2rgb = eval_sh(pc.active_sh_degree, shs_view, dir_pp_normalized)
                colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0)
            else:
                if separate_sh:
                    dc, shs = pc.get_features_dc, pc.get_features_rest
                else:
                    shs = pc.get_features
        else:
            colors_precomp = override_color

        # Compute mesh depth for front/back classification.  The effective
        # blend weight must come from the caller's formal Hybrid path; never
        # silently substitute the mesh renderer's unrelated alpha map.
        alpha_eff = None
        if alpha_eff_override is not None:
            if not torch.is_tensor(alpha_eff_override):
                raise TypeError("alpha_eff_override must be a torch.Tensor")
            if alpha_eff_override.ndim == 2:
                alpha_eff = alpha_eff_override.unsqueeze(0)
            elif alpha_eff_override.ndim == 3:
                alpha_eff = alpha_eff_override
            else:
                raise ValueError(
                    "alpha_eff_override must have shape [H, W] or [1, H, W]"
                )
            expected_shape = (
                1,
                int(viewpoint_camera.image_height),
                int(viewpoint_camera.image_width),
            )
            if tuple(alpha_eff.shape) != expected_shape:
                raise ValueError(
                    f"alpha_eff_override shape {tuple(alpha_eff.shape)} "
                    f"!= expected {expected_shape}"
                )
            if not alpha_eff.is_floating_point():
                raise TypeError("alpha_eff_override must use a floating dtype")
            if not bool(torch.isfinite(alpha_eff).all().item()):
                raise ValueError("alpha_eff_override must contain only finite values")
            if not bool(((alpha_eff >= 0.0) & (alpha_eff <= 1.0)).all().item()):
                raise ValueError("alpha_eff_override values must lie in [0, 1]")
            alpha_eff = alpha_eff.detach().to(
                device=means3D.device, dtype=torch.float32
            ).contiguous()

        if mesh_depth is None and self.mesh_renderer is not None:
            pos, verts_view, rast, mask = self.mesh_renderer.rasterize(viewpoint_camera)
            depth_linear, _ = self.mesh_renderer.depth_from_rast(
                viewpoint_camera, verts_view, rast, mask, normalize=False
            )
            mesh_depth = depth_linear  # [1, H, W]
        elif mesh_depth is None and alpha_eff is not None:
            raise ValueError(
                "alpha_eff_override requires mesh_depth or a mesh_renderer "
                "for front/back classification"
            )

        # 使用 GaussianRasterizer，这样会走原来的 None 处理逻辑
        rasterizer = mesh_gs_self_prune.GaussianRasterizer(raster_settings=raster_settings)
        rendered_front, radii, alpha, depth, visibility, gaussian_contrib, rendered_total, depth_total = rasterizer(
            means3D=means3D,
            means2D=means2D,
            opacities=opacity,
            shs=shs,
            colors_precomp=colors_precomp,
            scales=scales,
            rotations=rotations,
            cov3D_precomp=cov3D_precomp,
            depth_image=mesh_depth,
            alpha_eff=alpha_eff,
        )

        rendered_front = rendered_front.clamp(0, 1)
        rendered_total = rendered_total.clamp(0, 1)
        out = {
            "render": rendered_front[:3],
            "render_all": rendered_total[:3],
            "viewspace_points": screenspace_points,
            "visibility_filter": visibility,
            "radii": radii,
            "depth": depth,
            "depth_total": depth_total,
            "alpha": alpha,
            "gs_visibility_under_mesh": None,
            "gaussian_contrib": gaussian_contrib,
        }

        return out
