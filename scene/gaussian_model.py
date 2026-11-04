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
import numpy as np
from utils.general_utils import inverse_sigmoid, get_expon_lr_func, build_rotation
from torch import nn
import os
import json
from utils.system_utils import mkdir_p
from plyfile import PlyData, PlyElement
from utils.sh_utils import RGB2SH
from simple_knn._C import distCUDA2
from utils.graphics_utils import BasicPointCloud
from utils.general_utils import strip_symmetric, build_scaling_rotation

try:
    from diff_gaussian_rasterization import SparseGaussianAdam
except:
    pass

class GaussianModel:

    def setup_functions(self):
        def build_covariance_from_scaling_rotation(scaling, scaling_modifier, rotation):
            L = build_scaling_rotation(scaling_modifier * scaling, rotation)
            actual_covariance = L @ L.transpose(1, 2)
            symm = strip_symmetric(actual_covariance)
            return symm
        
        self.scaling_activation = torch.exp
        self.scaling_inverse_activation = torch.log

        self.covariance_activation = build_covariance_from_scaling_rotation

        self.opacity_activation = torch.sigmoid
        self.inverse_opacity_activation = inverse_sigmoid

        self.rotation_activation = torch.nn.functional.normalize


    def __init__(self, sh_degree, optimizer_type="default"):
        self.active_sh_degree = 0
        self.optimizer_type = optimizer_type
        self.max_sh_degree = sh_degree  
        self._xyz = torch.empty(0)
        self._features_dc = torch.empty(0)
        self._features_rest = torch.empty(0)
        self._scaling = torch.empty(0)
        self._rotation = torch.empty(0)
        self._opacity = torch.empty(0)
        self.max_radii2D = torch.empty(0)
        self.xyz_gradient_accum = torch.empty(0)
        self.denom = torch.empty(0)
        self.optimizer = None
        self.percent_dense = 0
        # Optional split stability diagnostic.  The default intentionally
        # preserves the historical torch.normal path byte-for-byte in spirit;
        # callers opt into deterministic principal-axis children explicitly.
        self.densify_split_mode = "random"
        self.densify_split_max_offset_fraction = 0.05
        self.spatial_lr_scale = 0
        self.setup_functions()

        self.visibility = torch.zeros(0, device="cuda")

    def capture(self):
        return (
            self.active_sh_degree,
            self._xyz,
            self._features_dc,
            self._features_rest,
            self._scaling,
            self._rotation,
            self._opacity,
            self.max_radii2D,
            self.xyz_gradient_accum,
            self.denom,
            self.optimizer.state_dict(),
            self.spatial_lr_scale,
        )
    
    def restore(self, model_args, training_args):
        (self.active_sh_degree, 
        self._xyz, 
        self._features_dc, 
        self._features_rest,
        self._scaling, 
        self._rotation, 
        self._opacity,
        self.max_radii2D, 
        xyz_gradient_accum, 
        denom,
        opt_dict, 
        self.spatial_lr_scale) = model_args
        self.training_setup(training_args)
        self.xyz_gradient_accum = xyz_gradient_accum
        self.denom = denom
        self.optimizer.load_state_dict(opt_dict)

    @property
    def get_scaling(self):
        return self.scaling_activation(self._scaling)
    
    @property
    def get_rotation(self):
        return self.rotation_activation(self._rotation)
    
    @property
    def get_xyz(self):
        return self._xyz
    
    @property
    def get_features(self):
        features_dc = self._features_dc
        features_rest = self._features_rest
        return torch.cat((features_dc, features_rest), dim=1)
    
    @property
    def get_features_dc(self):
        return self._features_dc
    
    @property
    def get_features_rest(self):
        return self._features_rest
    
    @property
    def get_opacity(self):
        return self.opacity_activation(self._opacity)
    
    @property
    def get_exposure(self):
        return self._exposure

    def get_exposure_from_name(self, image_name):
        if self.pretrained_exposures is None:
            return self._exposure[self.exposure_mapping[image_name]]
        else:
            return self.pretrained_exposures[image_name]
    
    def get_covariance(self, scaling_modifier = 1):
        return self.covariance_activation(self.get_scaling, scaling_modifier, self._rotation)

    def oneupSHdegree(self):
        if self.active_sh_degree < self.max_sh_degree:
            self.active_sh_degree += 1
            
    def merge_gaussians(self, gaussians, max_sh_degree = 3):
        if not gaussians:
            return
        
        num_new_coeffs = (max_sh_degree + 1) ** 2 - 1

        old_params = self._features_rest.clone()
        num_old_coeffs = old_params.shape[1]   
        self._features_rest = nn.Parameter(torch.zeros(old_params.shape[0], num_new_coeffs, old_params.shape[2], device="cuda", dtype=torch.float))
        self._features_rest[:, :num_old_coeffs, :] = old_params
        
        for gaussian in gaussians:
            old_params = gaussian._features_rest.clone()
            num_old_coeffs = old_params.shape[1] 
            gaussian._features_rest = nn.Parameter(torch.zeros(old_params.shape[0], num_new_coeffs, old_params.shape[2], device="cuda", dtype=torch.float))
            gaussian._features_rest[:, :num_old_coeffs, :] = old_params
            
            self._xyz = torch.cat((self._xyz, gaussian._xyz), dim=0)
            self._features_dc = torch.cat((self._features_dc, gaussian._features_dc), dim=0)
            self._features_rest = torch.cat((self._features_rest, gaussian._features_rest), dim=0)
            self._opacity = torch.cat((self._opacity, gaussian._opacity), dim=0)
            self._scaling = torch.cat((self._scaling, gaussian._scaling), dim=0)
            self._rotation = torch.cat((self._rotation, gaussian._rotation), dim=0)
            try:
                self.object_id = torch.cat((self.object_id, gaussian.object_id), dim=0)
                print("object_id is defined: ", gaussian.object_id[0])
            except:
                print("object_id is not defined")
                print("object_id is not defined")
                print("object_id is not defined")
                
    def create_from_pcd(self, pcd : BasicPointCloud, cam_infos : int, spatial_lr_scale : float, train_test_exp : bool):
        self.spatial_lr_scale = spatial_lr_scale
        fused_point_cloud = torch.tensor(np.asarray(pcd.points)).float().cuda()
        fused_color = RGB2SH(torch.tensor(np.asarray(pcd.colors)).float().cuda())
        features = torch.zeros((fused_color.shape[0], 3, (self.max_sh_degree + 1) ** 2)).float().cuda()
        features[:, :3, 0 ] = fused_color
        features[:, 3:, 1:] = 0.0

        print("Number of points at initialisation : ", fused_point_cloud.shape[0])

        dist2 = torch.clamp_min(distCUDA2(torch.from_numpy(np.asarray(pcd.points)).float().cuda()), 0.0000001)
        scales = torch.log(torch.sqrt(dist2))[...,None].repeat(1, 3)
        rots = torch.zeros((fused_point_cloud.shape[0], 4), device="cuda")
        rots[:, 0] = 1

        opacities = self.inverse_opacity_activation(0.1 * torch.ones((fused_point_cloud.shape[0], 1), dtype=torch.float, device="cuda"))

        self._xyz = nn.Parameter(fused_point_cloud.requires_grad_(True))
        self._features_dc = nn.Parameter(features[:,:,0:1].transpose(1, 2).contiguous().requires_grad_(True))
        self._features_rest = nn.Parameter(features[:,:,1:].transpose(1, 2).contiguous().requires_grad_(True))
        self._scaling = nn.Parameter(scales.requires_grad_(True))
        self._rotation = nn.Parameter(rots.requires_grad_(True))
        self._opacity = nn.Parameter(opacities.requires_grad_(True))
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")
        self.exposure_mapping = {cam_info.image_name: idx for idx, cam_info in enumerate(cam_infos)}
        self.pretrained_exposures = None
        exposure = torch.eye(3, 4, device="cuda")[None].repeat(len(cam_infos), 1, 1)
        self._exposure = nn.Parameter(exposure.requires_grad_(True))

        self.train_test_exp = train_test_exp
        self.visibility = torch.zeros((fused_point_cloud.shape[0]), device="cuda")

    def training_setup(self, training_args):
        self.percent_dense = training_args.percent_dense
        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")

        l = [
            {'params': [self._xyz], 'lr': training_args.position_lr_init * self.spatial_lr_scale, "name": "xyz"},
            {'params': [self._features_dc], 'lr': training_args.feature_lr, "name": "f_dc"},
            {'params': [self._features_rest], 'lr': training_args.feature_lr / 20.0, "name": "f_rest"},
            {'params': [self._opacity], 'lr': training_args.opacity_lr, "name": "opacity"},
            {'params': [self._scaling], 'lr': training_args.scaling_lr, "name": "scaling"},
            {'params': [self._rotation], 'lr': training_args.rotation_lr, "name": "rotation"}
        ]

        if self.optimizer_type == "default":
            self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)
        elif self.optimizer_type == "sparse_adam":
            try:
                self.optimizer = SparseGaussianAdam(l, lr=0.0, eps=1e-15)
            except:
                # A special version of the rasterizer is required to enable sparse adam
                self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)
        if self.train_test_exp:
            self.exposure_optimizer = torch.optim.Adam([self._exposure])

        self.xyz_scheduler_args = get_expon_lr_func(lr_init=training_args.position_lr_init*self.spatial_lr_scale,
                                                    lr_final=training_args.position_lr_final*self.spatial_lr_scale,
                                                    lr_delay_mult=training_args.position_lr_delay_mult,
                                                    max_steps=training_args.position_lr_max_steps)
        
        self.exposure_scheduler_args = get_expon_lr_func(training_args.exposure_lr_init, training_args.exposure_lr_final,
                                                        lr_delay_steps=training_args.exposure_lr_delay_steps,
                                                        lr_delay_mult=training_args.exposure_lr_delay_mult,
                                                        max_steps=training_args.iterations)

    def update_learning_rate(self, iteration):
        ''' Learning rate scheduling per step '''
        if self.train_test_exp:
            if self.pretrained_exposures is None:
                for param_group in self.exposure_optimizer.param_groups:
                    param_group['lr'] = self.exposure_scheduler_args(iteration)

        for param_group in self.optimizer.param_groups:
            if param_group["name"] == "xyz":
                lr = self.xyz_scheduler_args(iteration)
                param_group['lr'] = lr
                return lr

    def construct_list_of_attributes(self):
        l = ['x', 'y', 'z', 'nx', 'ny', 'nz']
        # All channels except the 3 DC
        for i in range(self._features_dc.shape[1]*self._features_dc.shape[2]):
            l.append('f_dc_{}'.format(i))
        for i in range(self._features_rest.shape[1]*self._features_rest.shape[2]):
            l.append('f_rest_{}'.format(i))
        l.append('opacity')
        for i in range(self._scaling.shape[1]):
            l.append('scale_{}'.format(i))
        for i in range(self._rotation.shape[1]):
            l.append('rot_{}'.format(i))
        return l

    def save_ply(self, path):
        mkdir_p(os.path.dirname(path))

        xyz = self._xyz.detach().cpu().numpy()
        normals = np.zeros_like(xyz)
        f_dc = self._features_dc.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        f_rest = self._features_rest.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        opacities = self._opacity.detach().cpu().numpy()
        scale = self._scaling.detach().cpu().numpy()
        rotation = self._rotation.detach().cpu().numpy()

        dtype_full = [(attribute, 'f4') for attribute in self.construct_list_of_attributes()]

        elements = np.empty(xyz.shape[0], dtype=dtype_full)
        attributes = np.concatenate((xyz, normals, f_dc, f_rest, opacities, scale, rotation), axis=1)
        elements[:] = list(map(tuple, attributes))
        el = PlyElement.describe(elements, 'vertex')
        PlyData([el]).write(path)

    def reset_opacity(self):
        opacities_new = self.inverse_opacity_activation(torch.min(self.get_opacity, torch.ones_like(self.get_opacity)*0.01))
        optimizable_tensors = self.replace_tensor_to_optimizer(opacities_new, "opacity")
        self._opacity = optimizable_tensors["opacity"]

    def load_ply(self, path, train_test_exp = False):
        plydata = PlyData.read(path)
        self.train_test_exp = train_test_exp
        if train_test_exp:
            exposure_file = os.path.join(os.path.dirname(path), os.pardir, os.pardir, "exposure.json")
            if os.path.exists(exposure_file):
                with open(exposure_file, "r") as f:
                    exposures = json.load(f)
                self.pretrained_exposures = {image_name: torch.FloatTensor(exposures[image_name]).requires_grad_(False).cuda() for image_name in exposures}
                print(f"Pretrained exposures loaded.")
            else:
                print(f"No exposure to be loaded at {exposure_file}")
                self.pretrained_exposures = None
        xyz = np.stack((np.asarray(plydata.elements[0]["x"]),
                        np.asarray(plydata.elements[0]["y"]),
                        np.asarray(plydata.elements[0]["z"])),  axis=1)

        try:
            opacities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis]
        except (KeyError, ValueError):
            # 如果 PLY 文件中没有 opacity 字段，使用默认值（0.99 表示几乎完全不透明）
            print("PLY 文件中没有 opacity 字段，使用默认值（0.99 表示几乎完全不透明）")
            from utils.general_utils import inverse_sigmoid
            default_opacity_value = 0.99  # 实际透明度值
            opacities = np.full((xyz.shape[0], 1), inverse_sigmoid(torch.tensor(default_opacity_value)).item())
        
        features_dc = np.zeros((xyz.shape[0], 3, 1))
        # features_dc[:, 0, 0] = np.asarray(plydata.elements[0]["f_dc_0"])
        # features_dc[:, 1, 0] = np.asarray(plydata.elements[0]["f_dc_1"])
        # features_dc[:, 2, 0] = np.asarray(plydata.elements[0]["f_dc_2"])
        features_dc = np.zeros((xyz.shape[0], 3, 1))
        try:
            features_dc[:, 0, 0] = np.asarray(plydata.elements[0]["f_dc_0"])
            features_dc[:, 1, 0] = np.asarray(plydata.elements[0]["f_dc_1"])
            features_dc[:, 2, 0] = np.asarray(plydata.elements[0]["f_dc_2"])
        except (KeyError, ValueError):
            # 如果 PLY 文件中没有 f_dc 字段，使用默认值 0（对应灰色 RGB = 0.5, 0.5, 0.5）
            print("PLY 文件中没有 f_dc 字段，使用默认值 0（灰色）")
            # features_dc 已经是全 0，无需额外设置

        # extra_f_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("f_rest_")]
        # extra_f_names = sorted(extra_f_names, key = lambda x: int(x.split('_')[-1]))
        # assert len(extra_f_names)==3*(self.max_sh_degree + 1) ** 2 - 3
        # features_extra = np.zeros((xyz.shape[0], len(extra_f_names)))
        # for idx, attr_name in enumerate(extra_f_names):
        #     features_extra[:, idx] = np.asarray(plydata.elements[0][attr_name])
        # # Reshape (P,F*SH_coeffs) to (P, F, SH_coeffs except DC)
        # features_extra = features_extra.reshape((features_extra.shape[0], 3, (self.max_sh_degree + 1) ** 2 - 1))
        
        extra_f_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("f_rest_")]
        extra_f_names = sorted(extra_f_names, key = lambda x: int(x.split('_')[-1]))
        expected_count = 3*(self.max_sh_degree + 1) ** 2 - 3

        if len(extra_f_names) != expected_count:
            print(f"PLY 文件中 f_rest_ 字段数量不匹配：期望 {expected_count} 个，实际 {len(extra_f_names)} 个，使用默认值 0")
            # 创建正确数量的全 0 数组
            features_extra = np.zeros((xyz.shape[0], expected_count))
        else:
            features_extra = np.zeros((xyz.shape[0], len(extra_f_names)))
            for idx, attr_name in enumerate(extra_f_names):
                features_extra[:, idx] = np.asarray(plydata.elements[0][attr_name])

        # Reshape (P,F*SH_coeffs) to (P, F, SH_coeffs except DC)
        features_extra = features_extra.reshape((features_extra.shape[0], 3, (self.max_sh_degree + 1) ** 2 - 1))

        scale_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("scale_")]
        scale_names = sorted(scale_names, key = lambda x: int(x.split('_')[-1]))
        scales = np.zeros((xyz.shape[0], len(scale_names)))
        for idx, attr_name in enumerate(scale_names):
            scales[:, idx] = np.asarray(plydata.elements[0][attr_name])

        rot_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("rot")]
        rot_names = sorted(rot_names, key = lambda x: int(x.split('_')[-1]))
        rots = np.zeros((xyz.shape[0], len(rot_names)))
        for idx, attr_name in enumerate(rot_names):
            rots[:, idx] = np.asarray(plydata.elements[0][attr_name])

        self._xyz = nn.Parameter(torch.tensor(xyz, dtype=torch.float, device="cuda").requires_grad_(True))
        self._features_dc = nn.Parameter(torch.tensor(features_dc, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
        self._features_rest = nn.Parameter(torch.tensor(features_extra, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
        self._opacity = nn.Parameter(torch.tensor(opacities, dtype=torch.float, device="cuda").requires_grad_(True))
        self._scaling = nn.Parameter(torch.tensor(scales, dtype=torch.float, device="cuda").requires_grad_(True))
        self._rotation = nn.Parameter(torch.tensor(rots, dtype=torch.float, device="cuda").requires_grad_(True))

        self.active_sh_degree = self.max_sh_degree

        self.visibility = torch.zeros((xyz.shape[0]), device="cuda") 
        self.max_radii2D = torch.zeros((xyz.shape[0]), device="cuda")
        self.xyz_gradient_accum = torch.zeros((xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((xyz.shape[0], 1), device="cuda")

    def replace_tensor_to_optimizer(self, tensor, name):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if group["name"] == name:
                stored_state = self.optimizer.state.get(group['params'][0], None)
                stored_state["exp_avg"] = torch.zeros_like(tensor)
                stored_state["exp_avg_sq"] = torch.zeros_like(tensor)

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(tensor.requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def _prune_optimizer(self, mask):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:
                stored_state["exp_avg"] = stored_state["exp_avg"][mask]
                stored_state["exp_avg_sq"] = stored_state["exp_avg_sq"][mask]

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter((group["params"][0][mask].requires_grad_(True)))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(group["params"][0][mask].requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def prune_points(self, mask):
        n_before = int(self.get_xyz.shape[0])
        valid_points_mask = ~mask
        optimizable_tensors = self._prune_optimizer(valid_points_mask)

        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]

        self.xyz_gradient_accum = self.xyz_gradient_accum[valid_points_mask]

        try:
            self.denom = self.denom[valid_points_mask]
            self.max_radii2D = self.max_radii2D[valid_points_mask]
            self.tmp_radii = self.tmp_radii[valid_points_mask]
        except:
            pass

        self.visibility = self.visibility[valid_points_mask]
        return n_before - int(self.get_xyz.shape[0])

    def cat_tensors_to_optimizer(self, tensors_dict):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            assert len(group["params"]) == 1
            extension_tensor = tensors_dict[group["name"]]
            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:

                stored_state["exp_avg"] = torch.cat((stored_state["exp_avg"], torch.zeros_like(extension_tensor)), dim=0)
                stored_state["exp_avg_sq"] = torch.cat((stored_state["exp_avg_sq"], torch.zeros_like(extension_tensor)), dim=0)

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]

        return optimizable_tensors

    def densification_postfix(self, new_xyz, new_features_dc, new_features_rest, new_opacities, new_scaling, new_rotation, new_tmp_radii):
        d = {"xyz": new_xyz,
        "f_dc": new_features_dc,
        "f_rest": new_features_rest,
        "opacity": new_opacities,
        "scaling" : new_scaling,
        "rotation" : new_rotation}

        optimizable_tensors = self.cat_tensors_to_optimizer(d)
        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]

        if self.tmp_radii is not None:
            self.tmp_radii = torch.cat((self.tmp_radii, new_tmp_radii))

        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")

        self.visibility = torch.cat((self.visibility, torch.zeros(new_tmp_radii.shape[0], device="cuda")))

    def densify_and_split(
        self,
        grads,
        grad_threshold,
        scene_extent,
        N=2,
        surface_split_query=None,
    ):
        split_mode = getattr(self, "densify_split_mode", "random")
        max_offset_fraction = float(
            getattr(self, "densify_split_max_offset_fraction", 0.05)
        )
        if split_mode not in {"random", "deterministic_principal"}:
            raise ValueError(f"unsupported densify_split_mode: {split_mode!r}")
        if max_offset_fraction <= 0.0:
            raise ValueError("densify_split_max_offset_fraction must be > 0")
        if split_mode == "deterministic_principal" and N != 2:
            raise ValueError("deterministic_principal split mode supports only N=2")
        n_init_points = self.get_xyz.shape[0]
        # Extract points that satisfy the gradient condition
        padded_grad = torch.zeros((n_init_points), device="cuda")
        padded_grad[:grads.shape[0]] = grads.squeeze()
        selected_pts_mask = torch.where(padded_grad >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(selected_pts_mask,
                                              torch.max(self.get_scaling, dim=1).values > self.percent_dense*scene_extent)

        selected_scaling = self.get_scaling[selected_pts_mask]
        if split_mode == "random":
            stds = selected_scaling.repeat(N, 1)
            means = torch.zeros((stds.size(0), 3), device="cuda")
            samples = torch.normal(mean=means, std=stds)
        else:
            # Two deterministic children lie on the local longest-scale axis,
            # symmetrically around the parent.  The cap is expressed in world
            # units and the original rotation maps local offsets to world.
            parent_count = int(selected_scaling.shape[0])
            longest_axis = torch.argmax(selected_scaling, dim=1)
            max_scale = torch.max(selected_scaling, dim=1).values
            cap = torch.full_like(
                max_scale,
                float(max_offset_fraction) * float(scene_extent),
            )
            magnitude = torch.minimum(0.8 * max_scale, cap)
            base = torch.zeros(
                (parent_count, 3),
                device=selected_scaling.device,
                dtype=selected_scaling.dtype,
            )
            parent_indices = torch.arange(
                parent_count, device=selected_scaling.device
            )
            base[parent_indices, longest_axis] = magnitude
            # Keep all positive children followed by all negative children.
            # ``repeat(N, ...)`` below uses the same parent ordering.
            samples = torch.cat((base, -base), dim=0)
        rots = build_rotation(self._rotation[selected_pts_mask]).repeat(N,1,1)
        new_xyz = torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1) + self.get_xyz[selected_pts_mask].repeat(N, 1)
        surface_stats = {}
        if surface_split_query is not None and selected_pts_mask.any():
            new_xyz, surface_stats = surface_split_query.adjust_child_positions(
                self.get_xyz[selected_pts_mask],
                new_xyz,
                N,
            )
        new_scaling = self.scaling_inverse_activation(self.get_scaling[selected_pts_mask].repeat(N,1) / (0.8*N))
        new_rotation = self._rotation[selected_pts_mask].repeat(N,1)
        new_features_dc = self._features_dc[selected_pts_mask].repeat(N,1,1)
        new_features_rest = self._features_rest[selected_pts_mask].repeat(N,1,1)
        new_opacity = self._opacity[selected_pts_mask].repeat(N,1)
        new_tmp_radii = self.tmp_radii[selected_pts_mask].repeat(N)
        created_count = int(new_xyz.shape[0])

        self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacity, new_scaling, new_rotation, new_tmp_radii)

        prune_filter = torch.cat((selected_pts_mask, torch.zeros(N * selected_pts_mask.sum(), device="cuda", dtype=bool)))
        pruned_parent_count = int(self.prune_points(prune_filter))
        return {
            "selected_parents": pruned_parent_count,
            "created_children": created_count,
            "net_added": created_count - pruned_parent_count,
            **surface_stats,
        }

    def densify_and_clone(self, grads, grad_threshold, scene_extent):
        # Extract points that satisfy the gradient condition
        selected_pts_mask = torch.where(torch.norm(grads, dim=-1) >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(selected_pts_mask,
                                              torch.max(self.get_scaling, dim=1).values <= self.percent_dense*scene_extent)
        
        new_xyz = self._xyz[selected_pts_mask]
        new_features_dc = self._features_dc[selected_pts_mask]
        new_features_rest = self._features_rest[selected_pts_mask]
        new_opacities = self._opacity[selected_pts_mask]
        new_scaling = self._scaling[selected_pts_mask]
        new_rotation = self._rotation[selected_pts_mask]

        new_tmp_radii = self.tmp_radii[selected_pts_mask]
        created_count = int(new_xyz.shape[0])

        self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacities, new_scaling, new_rotation, new_tmp_radii)
        return {
            "selected_parents": created_count,
            "created_children": created_count,
            "net_added": created_count,
        }

    def densify_and_prune(
        self,
        max_grad,
        min_opacity,
        extent,
        max_screen_size,
        radii,
        surface_split_query=None,
    ):
        n_before = int(self.get_xyz.shape[0])
        grads = self.xyz_gradient_accum / self.denom
        grads[grads.isnan()] = 0.0

        # Guard against radii size mismatch when gaussians were pruned mid-iter
        # (e.g. by new_collection prune in triple_train.py between render and densify).
        n = self._xyz.shape[0]
        if radii.shape[0] != n:
            if radii.shape[0] > n:
                radii = radii[:n]
            else:
                pad = torch.zeros(n - radii.shape[0], device=radii.device, dtype=radii.dtype)
                radii = torch.cat([radii, pad])
        self.tmp_radii = radii
        clone_stats = self.densify_and_clone(grads, max_grad, extent)
        split_stats = self.densify_and_split(
            grads,
            max_grad,
            extent,
            surface_split_query=surface_split_query,
        )

        prune_mask = (self.get_opacity < min_opacity).squeeze()
        if max_screen_size:
            big_points_vs = self.max_radii2D > max_screen_size
            big_points_ws = self.get_scaling.max(dim=1).values > 0.1 * extent
            prune_mask = torch.logical_or(torch.logical_or(prune_mask, big_points_vs), big_points_ws)
        pruned_count = int(self.prune_points(prune_mask))
        n_after = int(self.get_xyz.shape[0])
        tmp_radii = self.tmp_radii
        self.tmp_radii = None

        torch.cuda.empty_cache()
        return {
            "n_before": n_before,
            "n_after": n_after,
            "clone_parents": clone_stats["selected_parents"],
            "clone_created": clone_stats["created_children"],
            "split_parents": split_stats["selected_parents"],
            "split_created": split_stats["created_children"],
            "split_net_added": split_stats["net_added"],
            "opacity_or_size_pruned": pruned_count,
            **{
                key: value
                for key, value in split_stats.items()
                if key.startswith("surface_")
            },
        }

    def add_densification_stats(self, viewspace_point_tensor, update_filter):
        self.xyz_gradient_accum[update_filter] += torch.norm(viewspace_point_tensor.grad[update_filter,:2], dim=-1, keepdim=True)
        self.denom[update_filter] += 1


    def gaussolve_init(self, mesh_renderer, **kwargs):
        """
        根据 mesh 的顶点初始化高斯点。
        """
        from gaussolve_package.gaussolve_model import gaussolve_init as gaussolve_init_func     
        gaussolve_init_func(gaussians=self, mesh_renderer=mesh_renderer, **kwargs)
