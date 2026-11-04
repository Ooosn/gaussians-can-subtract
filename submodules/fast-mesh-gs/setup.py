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

from setuptools import setup
from torch.utils.cpp_extension import CUDAExtension, BuildExtension
import os
os.path.dirname(os.path.abspath(__file__))

version_info = """
### Mesh GS 版本解释
1. 支持了高斯球级别的 Mesh 遮挡剔除功能。
2. 优化了背景混合逻辑：
   - Forward 阶段不再强制在 Kernel 内混合背景。
   - Backward 阶段支持传入外部背景梯度 (dL_dT_final)，
     解决了高斯球无法正确通过变透明来露出背景的问题。
3. 移除了硬编码的背景色梯度计算，提升了通用性。
"""

setup(
    name="fast_mesh_gs",
    packages=['fast_mesh_gs'],
    version="1.0.0",
    long_description=version_info,
    description="Mesh-occluded Gaussian Splatting",
    long_description_content_type="text/markdown",
    ext_modules=[
        CUDAExtension(
            name="fast_mesh_gs._C",
            sources=[
            "cuda_rasterizer/rasterizer_impl.cu",
            "cuda_rasterizer/forward.cu",
            "cuda_rasterizer/backward.cu",
            "rasterize_points.cu",
            "ext.cpp"],
            extra_compile_args={"nvcc": [
                "-I" + os.path.join(os.path.dirname(os.path.abspath(__file__)), "third_party/glm/"),
                "-allow-unsupported-compiler",
            ]})
        ],
    cmdclass={
        'build_ext': BuildExtension
    }
)
