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

from argparse import ArgumentParser, Namespace
import sys
import os

class GroupParams:
    pass

class ParamGroup:
    def __init__(self, parser: ArgumentParser, name : str, fill_none = False):
        group = parser.add_argument_group(name)
        for key, value in vars(self).items():
            shorthand = False
            if key.startswith("_"):
                shorthand = True
                key = key[1:]
            t = type(value)
            value = value if not fill_none else None 
            if shorthand:
                if t == bool:
                    group.add_argument("--" + key, ("-" + key[0:1]), default=value, action="store_true")
                else:
                    group.add_argument("--" + key, ("-" + key[0:1]), default=value, type=t)
            else:
                if t == bool:
                    group.add_argument("--" + key, default=value, action="store_true")
                else:
                    group.add_argument("--" + key, default=value, type=t)

    def extract(self, args):
        group = GroupParams()
        for arg in vars(args).items():
            if arg[0] in vars(self) or ("_" + arg[0]) in vars(self):
                setattr(group, arg[0], arg[1])
        return group

class ModelParams(ParamGroup): 
    def __init__(self, parser, sentinel=False):
        self.sh_degree = 3
        self._source_path = ""
        self._model_path = ""
        self._images = "images"
        self.depths = ""
        self.masks = ""
        self._resolution = -1
        self._white_background = False
        self.train_test_exp = False
        self.data_device = "cuda"
        self.eval = False

        # gaussolve parameters
        self.gaussolve_mode = False
        self.mesh_path = ""
        self.mesh_texture_path = ""
        self.gaussolve_sh_degree = 3
        self.gaussolve_densification = False
        self.gaussolve_reset_opacity = False
        self.mesh_face_prune_before_train = True
        self.mesh_face_prune_depth_threshold = 0.2
        self.mesh_face_prune_min_views = 1
        self.mesh_alpha_mode = False
        self.force_vertex_alpha = False
        self.mesh_alpha_from_scratch = False

        super().__init__(parser, "Loading Parameters", sentinel)

    def extract(self, args):
        g = super().extract(args)
        g.source_path = os.path.abspath(g.source_path)
        return g

class PipelineParams(ParamGroup):
    def __init__(self, parser):
        self.convert_SHs_python = False
        self.compute_cov3D_python = False
        self.debug = False
        self.antialiasing = False
        super().__init__(parser, "Pipeline Parameters")

class OptimizationParams(ParamGroup):
    def __init__(self, parser):
        self.iterations = 30_000
        self.position_lr_init = 0.00016
        self.position_lr_final = 0.0000016
        self.position_lr_delay_mult = 0.01
        self.position_lr_max_steps = 30_000
        self.feature_lr = 0.0025
        self.opacity_lr = 0.025
        self.scaling_lr = 0.005
        self.rotation_lr = 0.001
        self.exposure_lr_init = 0.01
        self.exposure_lr_final = 0.001
        self.exposure_lr_delay_steps = 0
        self.exposure_lr_delay_mult = 0.0
        self.percent_dense = 0.01
        self.lambda_dssim = 0.2
        self.densification_interval = 100
        self.opacity_reset_interval = 3000
        self.densify_from_iter = 500
        self.densify_until_iter = 15_000
        self.densify_grad_threshold = 0.0002
        self.depth_l1_weight_init = 1.0
        self.depth_l1_weight_final = 0.01
        # Optional diagnostic override for a short prefix run.  Zero keeps the
        # historical behavior (schedule horizon == opt.iterations); a positive
        # value lets a 6k stop use the exact 30k depth schedule explicitly.
        self.depth_schedule_max_steps = 0
        # Diagnostic only: run mesh-color and NDF render work in the current
        # stream instead of the historical unsynchronized auxiliary streams.
        self.synchronous_render_streams = False
        # Diagnostic only: keep the standard opacity prune but disable the
        # post-reset screen-size (max_radii2D/world-scale) prune branch.
        self.disable_screen_size_prune = False
        # Diagnostic only: keep COLMAP camera order and consume cameras in
        # sequence instead of random shuffle/pop.
        self.deterministic_camera_schedule = False
        # Diagnostic only: keep the initialized NDF in the alpha blend but do
        # not update its parameters or run its densification branch.
        self.freeze_ndf_training = False
        # Diagnostic only: train NDF appearance/opacity while keeping its
        # spatial geometry fixed at initialization.
        self.freeze_ndf_geometry = False
        # Diagnostic only: prevent the hybrid image loss from updating NDF
        # through the dissolve-alpha gate; NDF supervision remains active.
        self.detach_ndf_alpha_from_main = False
        # Scale all NDF optimizer groups without changing main-GS/mesh rates.
        self.ndf_lr_scale = 1.0
        # Freeze the NDF field through this iteration, then resume its normal
        # optimizer/densification path.  Zero preserves the historical mode.
        self.ndf_freeze_until_iter = 0
        self.random_background = False
        self.optimizer_type = "default"

        self.mesh_train = False
        self.mesh_pretrain_iterations = 0
        self.mesh_lr_init = 0.001
        self.mesh_lr_final = 0.0005
        self.mesh_lr_delay_steps = 0
        self.mesh_lr_delay_mult = 0.0
        self.use_sigmoid_rgb = True
        self.use_sigmoid_alpha = True
        # 如果为 True，则在 mesh_alpha_mode 下完全忽略 mesh / 纹理自带的 alpha，
        # 直接用训练脚本里初始化的可学习 alpha（from scratch）
        self.seed_gaussians = False
        self.seed_gaussians_gen_num_per_view = 100
        self.seed_gaussians_interval = 1500
        self.seed_gaussians_start_iter = 6000
        self.mesh_start_finetune_iteration = 16000
        self.error_threshold_densification = 0.01
        self.mesh_depth_tolerance = 0.05
        # NDF-only ablation: when enabled, the scalar dissolve-field
        # renderer receives no mesh depth image and therefore cannot apply
        # the mesh front/back cutoff.  Main Mesh/GS fusion and depth loss
        # continue to receive the mesh depth image unchanged.
        self.ndf_no_mesh_depth_cutoff = False
        self.gaussolve_reg_sparse = 1e-3
        self.gaussolve_reg_tv = 2e-4
        self.gaussolve_reg_bin = 5e-4
        self.gaussolve_reg_warmup_iters = 2000
        # Diagnostic only: preserve the post-initialization Gaussian population so
        # gradient-starved points are not removed before they can be measured.
        self.gradient_probe_lock_population = False
        # InitFilter accepts any source whose PLY/points3D rows pass the exact
        # geometry contract; the actual source row count is pinned internally.
        self.initfilter_allow_unlocked_source = True
        self.initfilter_expected_source_count = -1
        self.gaussolve_sup_enable = True
        self.gaussolve_sup_weight = 0.1
        self.gaussolve_sup_relerr_thr = 0.20
        self.gaussolve_sup_relerr_temp = 0.05
        self.gaussolve_sup_hard_relerr_thr = 0.30
        super().__init__(parser, "Optimization Parameters")

def get_combined_args(parser : ArgumentParser):
    cmdlne_string = sys.argv[1:]
    cfgfile_string = "Namespace()"
    args_cmdline = parser.parse_args(cmdlne_string)

    try:
        cfgfilepath = os.path.join(args_cmdline.model_path, "cfg_args")
        print("Looking for config file in", cfgfilepath)
        with open(cfgfilepath) as cfg_file:
            print("Config file found: {}".format(cfgfilepath))
            cfgfile_string = cfg_file.read()
    except TypeError:
        print("Config file not found at")
        pass
    args_cfgfile = eval(cfgfile_string)

    merged_dict = vars(args_cfgfile).copy()
    for k,v in vars(args_cmdline).items():
        if v != None:
            merged_dict[k] = v
    return Namespace(**merged_dict)
