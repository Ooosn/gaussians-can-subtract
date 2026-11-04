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
from torch import nn
import numpy as np
from PIL import Image
from utils.graphics_utils import getWorld2View2, getProjectionMatrix
from utils.general_utils import PILtoTorch
import cv2
import torchvision

class Camera(nn.Module):
    def __init__(self, colmap_id, R, T, FoVx, FoVy, image, bg,
                 image_name, uid, pl_pos = None, pl_intensity = None,
                 trans=np.array([0.0, 0.0, 0.0]), scale=1.0, data_device = "cuda",
                 is_hdr=False, image_path=None, use_hgs_finetune=False,
                 image_width=None, image_height=None, invdepthmap=None, depth_params=None,
                 mask=None, depth_path=None, mask_path=None, is_nerf_synthetic = None
                 ):
        super(Camera, self).__init__()

        self.uid = uid
        self.colmap_id = colmap_id
        self.R = R
        self.T = T
        self.FoVx = FoVx
        self.FoVy = FoVy
        self.image_name = image_name
        self.image_path = image_path
        
        self.is_hdr = is_hdr
        self.invdepthmap = invdepthmap
        self.depth_params = depth_params
        self.mask = mask
        self.depth_path = depth_path
        self.mask_path = mask_path
        self.is_nerf_synthetic = is_nerf_synthetic
        self.original_image = None
        
        if not use_hgs_finetune:
            self.cam_pose_adj = torch.nn.Parameter(torch.zeros((1, 6), requires_grad=True).cuda())
            self.pl_adj = torch.nn.Parameter(torch.zeros((1, 3), requires_grad=True).cuda())
        else:
            self.cam_pose_adj = torch.nn.Parameter(torch.zeros((1, 6), requires_grad=False).cuda())
            self.pl_adj = torch.nn.Parameter(torch.zeros((1, 3), requires_grad=False).cuda())

        self.R_cu = torch.tensor(R, dtype=torch.float32).cuda()
        self.T_cu = torch.tensor(T, dtype=torch.float32).cuda()

        try:
            self.data_device = torch.device(data_device)
        except Exception as e:
            print(e)
            print(f"[Warning] Custom device {data_device} failed, fallback to default cuda device" )
            self.data_device = torch.device("cuda")
        
        self.bg = torch.tensor(bg, dtype=torch.float32).to(self.data_device)
        if pl_pos is not None:
            self.pl_pos = torch.tensor(pl_pos, dtype=torch.float32).unsqueeze(0).cuda()
            self.pl_pos_init = torch.tensor(pl_pos, dtype=torch.float32).unsqueeze(0).cuda()
        else:
            self.pl_pos = None
            self.pl_pos_init = None

                
        #if gt_alpha_mask is not None:
            #self.original_image *= gt_alpha_mask.to(self.data_device)
        #else:
            #self.original_image *= torch.ones((1, self.image_height, self.image_width), device=self.data_device)
        if image is not None:
            self.init_image()

        self.zfar = 1000.0
        self.znear = 0.01

        self.trans = trans
        self.scale = scale
        self.trans_cu = torch.tensor(trans, dtype=torch.float32).cuda()
        self.scale_cu = torch.tensor(scale, dtype=torch.float32).cuda()
        # self.cx = self.image_width / 2.0
        # self.cy = self.image_height / 2.0
        self.world_view_transform = torch.tensor(getWorld2View2(R, T, trans, scale)).transpose(0, 1).cuda()
        self.projection_matrix = getProjectionMatrix(znear=self.znear, zfar=self.zfar, fovX=self.FoVx, fovY=self.FoVy).transpose(0,1).cuda()
        self.full_proj_transform = (self.world_view_transform.unsqueeze(0).bmm(self.projection_matrix.unsqueeze(0))).squeeze(0)
        self.camera_center = self.world_view_transform.inverse()[3, :3]
        
    def init_image(self):
        image = Image.open(self.image_path)
        image = np.array(image.convert("RGB"))
        if self.is_hdr:
            self.original_image = image.to(self.data_device)
        else:
            image = torch.tensor(np.array(image)/255.0, dtype=torch.float32).to(self.data_device)
            self.original_image = image.clamp(0.0, 1.0).to(self.data_device)
        self.original_image = self.original_image.permute(2, 0, 1)
        self.image_width = self.original_image.shape[2]
        self.image_height = self.original_image.shape[1]
        
        if self.depth_path != "":
            try:
                if self.is_nerf_synthetic:
                    self.invdepthmap = cv2.imread(self.depth_path, -1).astype(np.float32) / 512
                else:
                    self.invdepthmap = cv2.imread(self.depth_path, -1).astype(np.float32) / float(2**16)
            except FileNotFoundError:
                print(f"Error: The depth file at path '{self.depth_path}' was not found.")
                raise
            except IOError:
                print(f"Error: Unable to open the image file '{self.depth_path}'. It may be corrupted or an unsupported format.")
                raise
            except Exception as e:
                print(f"An unexpected error occurred when trying to read depth at {self.depth_path}: {e}")
                raise
        else:
            self.invdepthmap = None
        if self.mask_path != "":
            try:
                self.mask = np.array(Image.open(self.mask_path))/255.0
            except FileNotFoundError:
                print(f"Error: The mask file at path '{self.mask_path}' was not found.")
                raise
            except IOError:
                print(f"Error: Unable to open the image file '{self.mask_path}'. It may be corrupted or an unsupported format.")
                raise
            except Exception as e:
                print(f"An unexpected error occurred when trying to read mask at {self.mask_path}: {e}")
                raise
        
        if self.invdepthmap is not None:
            self.depth_mask = torch.ones(1, self.image_height, self.image_width).to(self.data_device)
            # self.invdepthmap = cv2.resize(invdepthmap, (self.image_width, self.image_height))
            
            self.invdepthmap[self.invdepthmap < 0] = 0
            self.depth_reliable = True

            if self.depth_params is not None:
                if self.depth_params["scale"] < 0.2 * self.depth_params["med_scale"] or self.depth_params["scale"] > 5 * self.depth_params["med_scale"]:
                    self.depth_reliable = False
                    self.depth_mask *= 0
                
                if self.depth_params["scale"] > 0:
                    self.invdepthmap = self.invdepthmap * self.depth_params["scale"] + self.depth_params["offset"]

            if self.invdepthmap.ndim != 2:
                self.invdepthmap = self.invdepthmap[..., 0]
            self.invdepthmap = torch.from_numpy(self.invdepthmap[None]).to(self.data_device)
        else:
            self.depth_reliable = False
            
        if self.mask is not None:
            self.mask = torch.tensor(self.mask, dtype=torch.float32).to(self.data_device).unsqueeze(0)
            
            if self.original_image is not None:
                self.original_image = self.original_image * self.mask + self.bg.unsqueeze(-1).unsqueeze(-1) * (1-self.mask)
        else:
            self.mask = None
    
    def del_tensor(self):
        self.original_image = None
        self.invdepthmap = None
        self.mask = None
        self.depth_mask = None
    
class MiniCam:
    def __init__(self, width, height, fovy, fovx, znear, zfar, world_view_transform, full_proj_transform):
        self.image_width = width
        self.image_height = height    
        self.FoVy = fovy
        self.FoVx = fovx
        self.znear = znear
        self.zfar = zfar
        self.world_view_transform = world_view_transform
        self.full_proj_transform = full_proj_transform
        view_inv = torch.inverse(self.world_view_transform)
        self.camera_center = view_inv[3][:3]

