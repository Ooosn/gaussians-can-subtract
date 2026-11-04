#
# Mesh renderer using nvdiffrast
# Simple texture-based rendering without shaders
#

import torch
import torch.nn as nn
import numpy as np
import trimesh
import os
import nvdiffrast.torch as dr
from PIL import Image
import torch.nn.functional as F
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from utils.general_utils import get_expon_lr_func
Image.MAX_IMAGE_PIXELS = None


class MeshRenderer(nn.Module):
    """Simple mesh renderer using nvdiffrast with texture mapping only."""

    @staticmethod
    def _to_logits(x, eps=1e-6):
        x = x.clamp(eps, 1.0 - eps)
        return torch.logit(x)
    
    @staticmethod
    def _clamp_color(x):
        """Clamp color values to [0, 1] range."""
        return x.clamp(0.0, 1.0)
    
    def __init__(
        self,
        mesh_path,
        texture_path=None,
        device="cuda",
        use_sigmoid_rgb=False,
        use_sigmoid_alpha=True,
        use_opengl=False,
        mesh_alpha_mode=False,
        alpha_from_scratch=False,
        force_vertex_alpha=False,
    ):
    
        # 动态切换上下文：训练用 CUDA，极致渲染用 OpenGL
        if use_opengl:
            self.ctx = dr.RasterizeGLContext()
        else:
            self.ctx = dr.RasterizeCudaContext()

        super(MeshRenderer, self).__init__()

        self.device = device
        # 是否启用可学习的 mesh alpha（当纹理/顶点没有 alpha 时）
        self.mesh_alpha_mode = bool(mesh_alpha_mode)
        # 控制是否“完全从 scratch 学 alpha”：如果为 True，则忽略纹理 / 顶点里自带的 alpha 通道，
        # 统一用我们在本文件里初始化的 alpha 值（例如 0.9）作为起点。
        self.alpha_from_scratch = bool(alpha_from_scratch)
        # 强制使用“顶点空间 alpha”流程（初始化与渲染插值都走顶点分支），
        # 即使当前存在 texture + uv 也不会走 texture-space alpha。
        self.force_vertex_alpha = bool(force_vertex_alpha)
        
        # --- 1. load vertices and faces ---
        mesh = trimesh.load(mesh_path, force='mesh', process=False)
        
        # vertices and faces
        self.verts = torch.tensor(mesh.vertices, dtype=torch.float32, device=device)
        self.faces = torch.tensor(mesh.faces, dtype=torch.int32, device=device)
        self.faces_contiguous = self.faces.contiguous()

        ones = torch.ones(self.verts.shape[0], 1, device=device)
        self.verts_h = torch.cat([self.verts, ones], dim=1).contiguous()
        
        # normals (if needed for lighting)
        if hasattr(mesh, 'vertex_normals') and mesh.vertex_normals is not None:
            self.normals = torch.tensor(mesh.vertex_normals, dtype=torch.float32, device=device)
            detected_normals = True
        else:
            mesh.fix_normals()
            self.normals = torch.tensor(mesh.vertex_normals, dtype=torch.float32, device=device)
            detected_normals = False
        
        # --- 3. loading vertex colors (if any) ---
        self.vertex_colors = None
        self.vertex_alpha = None
        detected_vertex_colors = False

        vc = None
        if hasattr(mesh.visual, "vertex_colors") and mesh.visual.vertex_colors is not None:
            cand = np.asarray(mesh.visual.vertex_colors)
            if cand.ndim == 2 and cand.shape[0] == self.verts.shape[0] and cand.shape[1] >= 3:
                vc = cand

        # Some glTF files store vertex colors in COLOR_0 attributes.
        if vc is None:
            for holder in (getattr(mesh, "visual", None), mesh):
                if holder is None or not hasattr(holder, "vertex_attributes"):
                    continue
                attrs = getattr(holder, "vertex_attributes", None)
                if isinstance(attrs, dict) and ("COLOR_0" in attrs):
                    cand = np.asarray(attrs["COLOR_0"])
                    if cand.ndim == 2 and cand.shape[0] == self.verts.shape[0] and cand.shape[1] >= 3:
                        vc = cand
                        break

        if vc is not None:
            vc = vc.astype(np.float32)
            if float(vc.max()) > 1.0:
                vc = vc / 255.0
            vc = np.clip(vc, 0.0, 1.0)
            self.vertex_colors = torch.tensor(vc[:, :3], dtype=torch.float32, device=device)
            detected_vertex_colors = True
            if vc.shape[1] >= 4:
                self.vertex_alpha = torch.tensor(vc[:, 3], dtype=torch.float32, device=device)

        # --- 4. loading texture ---
        self.texture = None
        self.texture_alpha = None
        if hasattr(mesh.visual, 'material') and hasattr(mesh.visual.material, 'baseColorTexture'):
            texture_img = mesh.visual.material.baseColorTexture
            # Try to get RGBA first
            if texture_img.mode == 'RGBA':
                texture_array = np.array(texture_img, dtype=np.float32) / 255.0
                self.texture = torch.tensor(texture_array[:, :, :3], dtype=torch.float32, device=device)  # [H,W,3]
                self.texture_alpha = torch.tensor(texture_array[:, :, 3], dtype=torch.float32, device=device)  # [H,W]
                detected_texture = True
            else:
                texture_img_rgb = texture_img.convert("RGB")
                texture_array = np.array(texture_img_rgb) / 255.0
                self.texture = torch.tensor(texture_array, dtype=torch.float32, device=device)  # [H,W,3]
                detected_texture = True
        elif texture_path:
            texture_img = Image.open(texture_path)
            if texture_img.mode == 'RGBA':
                texture_array = np.array(texture_img, dtype=np.float32) / 255.0
                self.texture = torch.tensor(texture_array[:, :, :3], dtype=torch.float32, device=device)  # [H,W,3]
                self.texture_alpha = torch.tensor(texture_array[:, :, 3], dtype=torch.float32, device=device)  # [H,W]
            else:
                texture_img_rgb = texture_img.convert("RGB")
                texture_array = np.array(texture_img_rgb) / 255.0
                self.texture = torch.tensor(texture_array, dtype=torch.float32, device=device)  # [H,W,3]
            detected_texture = True
        else:
            detected_texture = False
            
        # UV processing
        if hasattr(mesh.visual, 'uv') and mesh.visual.uv is not None:
            self.uv = torch.tensor(mesh.visual.uv, dtype=torch.float32, device=device)
            # flip V axis to adapt OpenGL
            self.uv[:, 1] = 1.0 - self.uv[:, 1]
            detected_uv = True
        else:
            detected_uv = False
            self.uv = None

        # Stable sidecar load path for learned vertex alpha.
        if self.force_vertex_alpha and (not self.alpha_from_scratch) and (self.vertex_alpha is None):
            mesh_dir = os.path.dirname(mesh_path)
            mesh_stem = os.path.splitext(os.path.basename(mesh_path))[0]
            # Priority:
            # 1) exact sidecar for current mesh name (mesh_final -> mesh_final.vertex_alpha.npy)
            # 2) mesh_pretrained.vertex_alpha.npy in same directory
            # 3) any *.vertex_alpha.npy in same directory with matching vertex count
            candidate_sidecars = [
                os.path.join(mesh_dir, f"{mesh_stem}.vertex_alpha.npy"),
                os.path.join(mesh_dir, "mesh_pretrained.vertex_alpha.npy"),
            ]
            try:
                for fname in os.listdir(mesh_dir):
                    if fname.endswith(".vertex_alpha.npy"):
                        path = os.path.join(mesh_dir, fname)
                        if path not in candidate_sidecars:
                            candidate_sidecars.append(path)
            except Exception:
                pass

            for sidecar_path in candidate_sidecars:
                if not os.path.exists(sidecar_path):
                    continue
                try:
                    alpha_np = np.load(sidecar_path).astype(np.float32).reshape(-1)
                except Exception:
                    continue
                if alpha_np.shape[0] == int(self.verts.shape[0]):
                    self.vertex_alpha = torch.tensor(np.clip(alpha_np, 0.0, 1.0), dtype=torch.float32, device=device)
                    print(f"[MeshRenderer] Loaded vertex alpha sidecar: {sidecar_path}")
                    break
            
        # --- 5. Parameterization mode (set before alpha initialization) ---
        # Control whether to use sigmoid for RGB and alpha
        # True: use sigmoid (logits space, better for optimization, but small gradients at extremes)
        # False: use direct [0,1] values with clamp (more intuitive, better gradients, but need boundary handling)
        self.use_sigmoid_rgb = use_sigmoid_rgb  # Default: use sigmoid for RGB
        self.use_sigmoid_alpha = use_sigmoid_alpha  # Default: use sigmoid for alpha
        
        # --- 6. Initialize differentiable alpha parameter ---
        # 规则（与你的需求一致）：
        # - 若“贴图着色”（texture+uv），则 alpha 也必须在贴图空间（pixel-space）表示并可学习：alpha_tex ∈ R^{1×Ht×Wt×1}
        # - 若“顶点着色”（vertex_colors），则 alpha 在顶点空间表示并可学习：alpha_vtx ∈ R^{1×V}
        # - mesh_alpha_from_scratch 仅决定是否使用已有 alpha 初始化（贴图 alpha / 顶点 alpha）；为 True 时统一从 0.9 初始化
        alpha_max = 0.99  # 避免初值到 1.0 导致 sigmoid 饱和

        use_texture_shading = (self.texture is not None and self.uv is not None and (not self.force_vertex_alpha))
        use_vertex_shading = ((not use_texture_shading) and ((self.vertex_colors is not None) or (self.vertex_alpha is not None)))

        self.alpha = None
        if not self.mesh_alpha_mode:
            # 未开启 alpha mode：不初始化 alpha，不参与融合
            self.alpha = None
        else:
            if self.force_vertex_alpha and (not self.alpha_from_scratch) and (self.vertex_alpha is None):
                raise RuntimeError(
                    "force_vertex_alpha=True but no vertex alpha was loaded from mesh "
                    "(missing RGBA/COLOR_0 and sidecar)."
                )
            if use_texture_shading:
                # 贴图空间 alpha（与 texture 同分辨率）
                if self.texture is not None:
                    ht, wt = int(self.texture.shape[0]), int(self.texture.shape[1])
                else:
                    ht, wt = 1, 1

                if (not self.alpha_from_scratch) and (self.texture_alpha is not None):
                    # 直接用 RGBA 贴图的 alpha 初始化（pixel-space），后续会在 training_setup 里转 logits/Parameter
                    alpha_tex = torch.clamp(self.texture_alpha, max=alpha_max)  # [Ht, Wt]
                    self.alpha = alpha_tex.unsqueeze(0).unsqueeze(-1)  # [1, Ht, Wt, 1]
                    print(f"Initialized raw alpha from texture alpha channel (texture-space): {self.alpha.shape}, clamped to max {alpha_max}")
                else:
                    # 没有贴图 alpha 或 from_scratch：新建一张可学习 alpha 贴图
                    self.alpha = torch.ones((1, ht, wt, 1), device=device) * 0.98
                    print(f"Initialized raw alpha from scratch (texture-space): {self.alpha.shape}, init=0.9")

            elif use_vertex_shading:
                # 顶点空间 alpha
                if (not self.alpha_from_scratch) and (self.vertex_alpha is not None):
                    vertex_alpha_clamped = torch.clamp(self.vertex_alpha, max=alpha_max)  # [V]
                    self.alpha = vertex_alpha_clamped.unsqueeze(0)  # [1, V]
                    print(f"Initialized raw alpha from vertex alpha channel (vertex-space): {self.alpha.shape}, clamped to max {alpha_max}")
                else:
                    self.alpha = torch.ones((1, self.verts.shape[0]), device=device) * 0.98
                    print(f"Initialized raw alpha from scratch (vertex-space): {self.alpha.shape}, init=0.9")

            else:
                # 没有贴图也没有顶点颜色：退化到顶点空间 alpha
                self.alpha = torch.ones((1, self.verts.shape[0]), device=device) * 0.98
                print(f"Initialized raw alpha from scratch (fallback vertex-space): {self.alpha.shape}, init=0.9")
        # --- 7. default values cache ---
        self.default_bg = torch.tensor([1., 1., 1.], device=device)
        self.ranges = None 
        self.optimizer = None
        self.lr_scheduler = None
        self.current_lr = 0.0
        self.material_trainable = False
        self._direct_parameter_step_hook = None
        
        
        # Pre-allocate ranges tensor (reused for each render)
        # self.ranges = torch.tensor([[0, self.faces.shape[0]]], dtype=torch.int32, device="cpu")

        console = Console()
        # create a table
        table = Table(title="Mesh Renderer Settings", show_header=True, header_style="bold magenta")
        table.add_column("Property", style="cyan")
        table.add_column("Value", style="green")

        table.add_row("Vertex Colors", str(detected_vertex_colors))
        table.add_row("Texture", str(detected_texture))
        table.add_row("UV", str(detected_uv))
        table.add_row("Normals", str(detected_normals))

        console.print(Panel(table, expand=False, border_style="blue"))


    _EXACT_RENDER_TENSOR_NAMES = (
        "verts",
        "verts_h",
        "faces",
        "faces_contiguous",
        "uv",
        "normals",
        "texture",
        "vertex_colors",
        "alpha",
        "texture_alpha",
        "vertex_alpha",
        "ranges",
        "default_bg",
    )

    def exact_render_state(self):
        """Return a CPU snapshot of every raw tensor/flag used by rendering.

        The snapshot intentionally stores raw parameter values rather than
        activated or quantized mesh data.  It is a render-only artifact: the
        nvdiffrast context and optimizer are reconstructed by the caller, while
        tensor values, device-relevant layout, flags, and ``Parameter`` identity
        are preserved exactly.
        """
        tensors = {}
        tensor_devices = {}
        parameter_identity = {}
        parameter_requires_grad = {}
        for name in self._EXACT_RENDER_TENSOR_NAMES:
            value = getattr(self, name, None)
            if value is None:
                tensors[name] = None
                tensor_devices[name] = None
                parameter_identity[name] = False
                parameter_requires_grad[name] = False
                continue
            if not torch.is_tensor(value):
                raise TypeError(f"exact render state field {name!r} is not a tensor or None")
            tensors[name] = value.detach().to(device="cpu").clone()
            tensor_devices[name] = str(value.device)
            parameter_identity[name] = isinstance(value, nn.Parameter)
            parameter_requires_grad[name] = bool(value.requires_grad)
        return {
            "format": "mesh_renderer_exact_render_state_v1",
            "version": 1,
            "tensors": tensors,
            "tensor_devices": tensor_devices,
            "parameter_identity": parameter_identity,
            "parameter_requires_grad": parameter_requires_grad,
            "flags": {
                "use_sigmoid_rgb": bool(self.use_sigmoid_rgb),
                "use_sigmoid_alpha": bool(self.use_sigmoid_alpha),
                "material_trainable": bool(self.material_trainable),
                "mesh_alpha_mode": bool(self.mesh_alpha_mode),
                "alpha_from_scratch": bool(self.alpha_from_scratch),
                "force_vertex_alpha": bool(self.force_vertex_alpha),
            },
        }

    def load_exact_render_state(self, payload):
        """Restore :meth:`exact_render_state` onto an already-constructed mesh.

        The caller must construct this instance from the original mesh first;
        this method then replaces all render tensors with the raw snapshot so
        no GLB texture/vertex quantization remains in the render path.
        """
        if not isinstance(payload, dict) or payload.get("format") != "mesh_renderer_exact_render_state_v1":
            raise ValueError("unsupported MeshRenderer exact render state")
        if int(payload.get("version", -1)) != 1:
            raise ValueError("unsupported MeshRenderer exact render state version")
        tensors = payload.get("tensors")
        identities = payload.get("parameter_identity")
        requires_grad = payload.get("parameter_requires_grad")
        devices = payload.get("tensor_devices", {})
        flags = payload.get("flags")
        if not isinstance(tensors, dict) or not isinstance(identities, dict) or not isinstance(requires_grad, dict) or not isinstance(flags, dict):
            raise ValueError("malformed MeshRenderer exact render state")
        target_device = torch.device(getattr(self, "device", "cuda"))
        for name in self._EXACT_RENDER_TENSOR_NAMES:
            value = tensors.get(name)
            if value is not None and not torch.is_tensor(value):
                raise TypeError(f"exact render state field {name!r} is not a tensor or None")
            if value is None:
                restored = None
            else:
                restore_device = torch.device("cpu") if name == "ranges" and str(devices.get(name, "")).startswith("cpu") else target_device
                restored = value.to(device=restore_device).clone()
                if bool(identities.get(name, False)):
                    restored = nn.Parameter(restored, requires_grad=bool(requires_grad.get(name, False)))
                elif bool(requires_grad.get(name, False)) and restored.is_floating_point():
                    restored.requires_grad_(True)
            # Module.__setattr__ does not allow replacing an existing
            # Parameter with a plain Tensor until its registry entry is removed.
            if name in self._parameters:
                del self._parameters[name]
            if name in self._buffers:
                del self._buffers[name]
            setattr(self, name, restored)
        for name in (
            "use_sigmoid_rgb",
            "use_sigmoid_alpha",
            "material_trainable",
            "mesh_alpha_mode",
            "alpha_from_scratch",
            "force_vertex_alpha",
        ):
            if name not in flags:
                raise ValueError(f"exact render state flag missing: {name}")
            setattr(self, name, bool(flags[name]))
        # A render-only restore must not retain optimizer references to the
        # pre-restore Parameter objects.
        self.optimizer = None
        self.lr_scheduler = None
        self.current_lr = 0.0
        self._direct_parameter_step_hook = None
        return self

    @classmethod
    def from_exact_render_state(cls, payload, device="cuda", use_opengl=False):
        """Construct a renderer directly from a raw exact-state payload.

        Unlike ``__init__``, this path does not read or quantize a GLB/PLY.
        It creates only the rasterizer context needed by a later GPU render
        and then restores every tensor/flag through ``load_exact_render_state``.
        CPU construction is intentionally context-free for portable state
        tests and offline manifest inspection.
        """
        instance = cls.__new__(cls)
        nn.Module.__init__(instance)
        instance.device = str(device)
        device_type = torch.device(device).type
        if device_type == "cuda":
            instance.ctx = dr.RasterizeGLContext() if use_opengl else dr.RasterizeCudaContext()
        else:
            instance.ctx = None
        instance.optimizer = None
        instance.lr_scheduler = None
        instance.current_lr = 0.0
        instance._direct_parameter_step_hook = None
        instance.load_exact_render_state(payload)
        return instance


    def training_setup(self, training_args):
        if self._direct_parameter_step_hook is not None:
            self._direct_parameter_step_hook.remove()
            self._direct_parameter_step_hook = None
        trainable_params = []
        alpha_params = []
        if self.texture is not None:
            if not isinstance(self.texture, nn.Parameter):
                if self.use_sigmoid_rgb:
                    self.texture = nn.Parameter(self._to_logits(self.texture))
                else:
                    self.texture = nn.Parameter(self.texture.clamp(0.0, 1.0))
            trainable_params.append(self.texture)
        if self.vertex_colors is not None:
            if not isinstance(self.vertex_colors, nn.Parameter):
                if self.use_sigmoid_rgb:
                    self.vertex_colors = nn.Parameter(self._to_logits(self.vertex_colors))
                else:
                    self.vertex_colors = nn.Parameter(self.vertex_colors.clamp(0.0, 1.0))
            trainable_params.append(self.vertex_colors)
        # Alpha is always trainable when exists
        if self.alpha is not None:
            if not isinstance(self.alpha, nn.Parameter):
                # Convert to Parameter based on parameterization mode
                if self.use_sigmoid_alpha:
                    # If alpha is not in logits (e.g., in [0,1] range), convert to logits
                    if self.alpha.max() <= 1.0 and self.alpha.min() >= 0.0:
                        # Alpha is in [0,1], convert to logits
                        self.alpha = nn.Parameter(self._to_logits(self.alpha))
                    else:
                        # Alpha is already in logits, just wrap as Parameter
                        self.alpha = nn.Parameter(self.alpha)
                else:
                    # Direct mode: ensure alpha is in [0,1] range
                    if self.alpha.max() > 1.0 or self.alpha.min() < 0.0:
                        # Alpha is in logits, convert to [0,1]
                        self.alpha = nn.Parameter(torch.sigmoid(self.alpha))
                    else:
                        # Alpha is already in [0,1], just clamp and wrap as Parameter
                        self.alpha = nn.Parameter(self.alpha.clamp(0.0, 1.0))

            # 如果是 mesh_alpha_mode，下游会用它来做融合场，可以给它单独的 param group，后续设置更大的 lr。
            if self.mesh_alpha_mode:
                alpha_params.append(self.alpha)
            else:
                trainable_params.append(self.alpha)

        if len(trainable_params) == 0 and len(alpha_params) == 0:
            self.optimizer = None
            self.lr_scheduler = None
            self.current_lr = 0.0
            self.material_trainable = False
            return

        param_groups = []
        if len(trainable_params) > 0:
            param_groups.append({"params": trainable_params, "name": "mesh_rgb"})
        if len(alpha_params) > 0:
            param_groups.append({"params": alpha_params, "name": "mesh_alpha"})

        self.optimizer = torch.optim.Adam(param_groups, lr=0.0, eps=1e-8)
        if not self.use_sigmoid_rgb and hasattr(self.optimizer, "register_step_post_hook"):
            self._direct_parameter_step_hook = self.optimizer.register_step_post_hook(
                lambda optimizer, args, kwargs: self._project_direct_parameters()
            )
        self.lr_scheduler = get_expon_lr_func(
            lr_init=training_args.mesh_lr_init,
            lr_final=training_args.mesh_lr_final,
            lr_delay_steps=training_args.mesh_lr_delay_steps,
            lr_delay_mult=training_args.mesh_lr_delay_mult,
            max_steps=training_args.iterations
        )
        self.material_trainable = True
        self.update_learning_rate(0)

    @torch.no_grad()
    def _project_direct_parameters(self):
        """Project direct [0,1] material parameters once per optimizer step.

        This keeps the render hot path to one texture lookup.  In particular it
        avoids materializing an activated 8K texture on every rendered view.
        """
        if not self.use_sigmoid_rgb:
            if isinstance(self.texture, nn.Parameter):
                self.texture.clamp_(0.0, 1.0)
            if isinstance(self.vertex_colors, nn.Parameter):
                self.vertex_colors.clamp_(0.0, 1.0)
        if not self.use_sigmoid_alpha and isinstance(self.alpha, nn.Parameter):
            self.alpha.clamp_(0.0, 1.0)


    def update_learning_rate(self, iteration):
        if self.optimizer is None or self.lr_scheduler is None:
            self.current_lr = 0.0
            return self.current_lr

        lr = self.lr_scheduler(iteration)
        # 给 alpha 一个稍大的 lr，只在 mesh_alpha_mode=True 且 alpha 单独成组时生效
        alpha_lr_mult = 5.0
        for param_group in self.optimizer.param_groups:
            group_name = param_group.get("name", "")
            if group_name == "mesh_alpha" and self.mesh_alpha_mode:
                param_group["lr"] = lr * alpha_lr_mult
            else:
                param_group["lr"] = lr
        self.current_lr = lr
        return lr


    def get_learning_rate(self):
        return self.current_lr


    def freeze(self, rgb=True, alpha=True):
        """
        Freeze parameters.
        
        Args:
            rgb: If True, freeze RGB parameters (texture/vertex_colors)
            alpha: If True, freeze alpha parameter
        """
        if rgb:
            if self.texture is not None and isinstance(self.texture, nn.Parameter):
                self.texture.requires_grad = False
            if self.vertex_colors is not None and isinstance(self.vertex_colors, nn.Parameter):
                self.vertex_colors.requires_grad = False
        
        if alpha:
            if self.alpha is not None:
                self.alpha.requires_grad = False
    
    def freeze_rgb(self):
        """Freeze only RGB parameters (texture/vertex_colors)."""
        self.freeze(rgb=True, alpha=False)
    
    def freeze_alpha(self):
        """Freeze only alpha parameter."""
        self.freeze(rgb=False, alpha=True)
    
    def freeze_all(self):
        """Freeze all parameters (RGB and alpha)."""
        self.freeze(rgb=True, alpha=True)
            
    def unfreeze(self, rgb=True, alpha=True):
        """
        Unfreeze parameters.
        
        Args:
            rgb: If True, unfreeze RGB parameters (texture/vertex_colors)
            alpha: If True, unfreeze alpha parameter
        """
        if rgb:
            if self.texture is not None and isinstance(self.texture, nn.Parameter):
                self.texture.requires_grad = True
            if self.vertex_colors is not None and isinstance(self.vertex_colors, nn.Parameter):
                self.vertex_colors.requires_grad = True
        
        if alpha:
            if self.alpha is not None:
                self.alpha.requires_grad = True
    
    def unfreeze_rgb(self):
        """Unfreeze only RGB parameters (texture/vertex_colors)."""
        self.unfreeze(rgb=True, alpha=False)
    
    def unfreeze_alpha(self):
        """Unfreeze only alpha parameter."""
        self.unfreeze(rgb=False, alpha=True)
    
    def unfreeze_all(self):
        """Unfreeze all parameters (RGB and alpha)."""
        self.unfreeze(rgb=True, alpha=True)


    def rasterize(self, camera):
        """
        only rasterize, return the basic data that will be used for all subsequent calculations.
        this part is "public", you only need to call it once.
        """
        H = int(camera.image_height)
        W = int(camera.image_width)

        view = camera.open_gl_view  # [4, 4]
        mvp = camera.open_gl_mvp     # [4, 4]

        # 齐次坐标
        pos = (self.verts_h @ mvp).to(torch.float32).contiguous()          # [V, 4] clip-space
        verts_view = (self.verts_h @ view)                               # [V, 4] view-space

        # 光栅化
        rast, _ = dr.rasterize(self.ctx, pos.unsqueeze(0),
                               self.faces_contiguous, (H, W),
                               ranges=self.ranges)
        mask = rast[..., 3] > 0  # [H, W]

        return pos, verts_view, rast, mask

    def depth_from_rast(self, camera, verts_view, rast, mask, normalize=False):
        """
        calculate depth on the given rasterization results (no rasterize).
        thread A can call this separately.
        """
        # view-space z
        depth_view, _ = dr.interpolate(verts_view[..., 2:3].contiguous(), rast, self.faces_contiguous)  # [1, H, W, 1]
        depth_view = depth_view.squeeze(0).squeeze(-1)  # [H, W]

        znear = getattr(camera, 'znear', 0.01)
        zfar  = getattr(camera, 'zfar', 1000.0)

        depth_linear = torch.abs(depth_view)  # [H, W]
        depth_linear[~mask.squeeze(0)] = zfar
        
        if normalize:
            depth_linear_normalized = depth_linear.clone()
            depth_linear_normalized[~mask.squeeze(0)] = 0
            depth_min = depth_linear_normalized.min()
            depth_max = depth_linear_normalized.max()
            depth_normalized = (depth_linear_normalized - depth_min) / (depth_max - depth_min)  # [H, W]
            depth_normalized[~mask.squeeze(0)] = 1.0
            return depth_linear.unsqueeze(0), depth_normalized.unsqueeze(0)
        else:
            return depth_linear.unsqueeze(0), None

    def color_from_rast(self, camera, pos, rast, mask, \
                        bg_color=None, enable_antialiasing=False, texture_override=None):
        """
        calculate final RGB on the given rasterization results (no rasterize).
        thread B can call this separately.
        """
        tex = texture_override if texture_override is not None else self.texture
        if bg_color is None:
            bg_color = self.default_bg

        # 2. texture sampling
        if tex is not None and self.uv is not None:
            # Apply activation based on parameterization mode
            if texture_override is None and self.material_trainable:
                if self.use_sigmoid_rgb:
                    tex = torch.sigmoid(tex)
                # Direct RGB parameters are projected after optimizer.step().
                # Sampling them raw avoids an 8K full-texture clamp per view.
            # interpolate UV
            uv_map, _ = dr.interpolate(self.uv, rast, self.faces_contiguous)
        
            # ensure texture is [1, H, W, C] and contiguous
            if tex.ndim == 3:
                tex = tex.unsqueeze(0) # [1, H, W, 3]
            if not tex.is_contiguous():
                tex = tex.contiguous()

            color_map = dr.texture(tex, uv_map, filter_mode='linear')
        elif self.vertex_colors is not None:
            # no texture, use vertex colors
            if self.material_trainable:
                if self.use_sigmoid_rgb:
                    vertex_colors = torch.sigmoid(self.vertex_colors)
                else:
                    vertex_colors = self.vertex_colors
            else:
                vertex_colors = self.vertex_colors
            color_map, _ = dr.interpolate(vertex_colors, rast, self.faces_contiguous)  # [1, H, W, 3]
        else:
            # fallback: pure color or normals
            color_map = torch.tensor([0.6, 0.6, 0.6], device=self.device).view(1, 1, 1, 3)

        # 3. blend background
        # mask shape [1, H, W], expand to [1, H, W, 3]
        mask_3d = mask.unsqueeze(-1).float()
        
        # broadcast background color
        if not isinstance(bg_color, torch.Tensor):
            if isinstance(bg_color, (int, float)):
                bg_color = torch.ones(3, device=self.device) * bg_color
            else:
                bg_color = torch.tensor(bg_color, dtype=torch.float32, device=self.device)
        bg_expand = bg_color.view(1, 1, 1, 3)

        # blend
        rendered = color_map * mask_3d + bg_expand * (1.0 - mask_3d)

        # 4. antialiasing (AA)
        # do it at the end
        if enable_antialiasing:
            rendered = dr.antialias(rendered, rast, pos, self.faces_contiguous)
        else:
            rendered = rendered.clamp(0.0, 1.0)
        
        # adjust output format [1, H, W, 3] -> [3, H, W]
        return rendered.squeeze(0).permute(2, 0, 1).clamp(0.0, 1.0)
    
    def get_alpha_map(self, camera, pos, rast, mask):
        """
        Render alpha map from differentiable alpha parameter.
        Returns alpha values in [0, 1] range (after sigmoid if use_sigmoid_alpha=True, 
        or after clamp if use_sigmoid_alpha=False).
        """
        H = int(camera.image_height)
        W = int(camera.image_width)

        # 判断当前是贴图着色还是顶点着色
        use_texture_shading = (self.texture is not None and self.uv is not None and (not self.force_vertex_alpha))
        use_vertex_shading = (not use_texture_shading and self.vertex_colors is not None)

        # 1）贴图着色：alpha 在贴图空间表示（self.alpha 为 [1,Ht,Wt,1]）
        if use_texture_shading and self.mesh_alpha_mode and (self.alpha is not None) and (self.alpha.dim() == 4):
            uv_map, _ = dr.interpolate(self.uv, rast, self.faces_contiguous)  # [1, H, W, 2]

            # self.alpha 可能是 logits（Parameter）或 raw [0,1]
            if self.use_sigmoid_alpha and isinstance(self.alpha, nn.Parameter):
                alpha_tex = torch.sigmoid(self.alpha)
            else:
                alpha_tex = self._clamp_color(self.alpha)

            if not alpha_tex.is_contiguous():
                alpha_tex = alpha_tex.contiguous()

            alpha_map = dr.texture(alpha_tex, uv_map, filter_mode='linear')  # [1, H, W, 1]
            alpha_map = alpha_map.squeeze(-1)  # [1, H, W]

        else:
            # 2）其它情况：在“顶点空间”插值 alpha（包括：
            #    - 顶点着色 + 顶点 alpha / scratch 新建的 alpha
            #    - 贴图着色但 alpha 不在贴图空间时（例如没有 uv / 无 texture 等））
            if self.alpha is None:
                # 若依然没有 alpha，就返回全 1（不做溶解）
                return torch.ones((1, H, W), device=self.device)

            # Get alpha values.
            # Policy:
            # - Load-time alpha is always stored as raw [0,1] tensor.
            # - Training-time (after training_setup) alpha may become nn.Parameter in logits space when use_sigmoid_alpha=True.
            # Therefore, only apply sigmoid when we are in sigmoid mode AND alpha is a Parameter (logits).
            if self.use_sigmoid_alpha and isinstance(self.alpha, nn.Parameter):
                alpha_values = torch.sigmoid(self.alpha)  # [1, V] logits -> [0,1]
            else:
                alpha_values = self._clamp_color(self.alpha)  # [1, V] raw -> [0,1]
            alpha_values = alpha_values.squeeze(0)  # [V]

            # Interpolate alpha at rasterized positions
            alpha_map, _ = dr.interpolate(
                alpha_values.unsqueeze(0).unsqueeze(-1),  # [1, V, 1]
                rast,
                self.faces_contiguous
            )  # [1, H, W, 1]

            alpha_map = alpha_map.squeeze(0).squeeze(-1)  # [H, W]
        
        # Apply mask
        mask_2d = mask.squeeze(0) if mask.dim() == 3 else mask
        alpha_map = alpha_map
        
        # 统一输出形状为 [1, H, W]
        if alpha_map.dim() == 2:      # [H, W]（顶点 alpha 分支）
            alpha_map = alpha_map.unsqueeze(0)  # [1, H, W]
        # 贴图 alpha 分支本身已是 [1, H, W]
        return alpha_map

    def render(self, camera, bg_color=None, enable_antialiasing=False):
        """
        convenient interface: calculate RGB + depth at once, internal依然是“先 rasterize，再分别算深度和颜色”。
        """
        pos, verts_view, rast, mask = self.rasterize(camera)
        # depth_linear, depth_normalized = self.depth_from_rast(camera, verts_view, rast, mask)
        rendered = self.color_from_rast(camera, pos, rast, mask, bg_color = bg_color, enable_antialiasing = enable_antialiasing)
        return rendered# depth_linear, depth_normalized


    def save_model(self, output_path):
        """
        Save current mesh representation.
        - Texture mode: export textured material; vertex colors are omitted.
        - Vertex-color mode: export vertex colors only; material is omitted.
        """
        print(f"Exporting model to {output_path}...")

        vertices = self.verts.detach().cpu().numpy()
        faces = self.faces.detach().cpu().numpy()
        visual = None

        # Texture branch (preferred when texture + UV are available).
        if self.texture is not None and self.uv is not None:
            print(self.material_trainable)
            # Apply activation based on parameterization mode
            if self.material_trainable:
                if self.use_sigmoid_rgb:
                    tex_to_export = torch.sigmoid(self.texture)
                else:
                    tex_to_export = self._clamp_color(self.texture)
            else:
                tex_to_export = self.texture
            tex_numpy = (tex_to_export.detach().cpu().clamp(0.0, 1.0).numpy() * 255).astype(np.uint8)
            h, w = tex_numpy.shape[:2]
            
            # Add alpha channel to texture if available.
            # When force_vertex_alpha=True with vertex alpha ([1, V]), keep texture as RGB
            # and store alpha in vertex colors for reload compatibility.
            if self.alpha is not None and not (self.force_vertex_alpha and self.alpha.dim() == 2):
                # 两种情况：
                # - 贴图空间 alpha: self.alpha 为 [1,Ht,Wt,1]，直接导出
                # - 顶点空间 alpha: self.alpha 为 [1,V]，沿用旧逻辑投影到纹理坐标
                if self.alpha.dim() == 4:
                    if self.use_sigmoid_alpha and isinstance(self.alpha, nn.Parameter):
                        alpha_tex = torch.sigmoid(self.alpha)
                    else:
                        alpha_tex = self._clamp_color(self.alpha)
                    alpha_tex_uint8 = (alpha_tex[0, :, :, 0].detach().cpu().clamp(0.0, 1.0).numpy() * 255).astype(np.uint8)
                else:
                    # Get alpha values based on parameterization mode
                    if self.use_sigmoid_alpha and isinstance(self.alpha, nn.Parameter):
                        alpha_values = torch.sigmoid(self.alpha).squeeze(0).detach().cpu().numpy()  # [V]
                    else:
                        alpha_values = self._clamp_color(self.alpha).squeeze(0).detach().cpu().numpy()  # [V]
                    
                    # Sample alpha at texture coordinates via UV
                    if self.uv is not None:
                        uv_coords = self.uv.detach().cpu().numpy()  # [V, 2]
                        # Map UV to texture coordinates (flip V for export)
                        tex_x = (uv_coords[:, 0] * (w - 1)).astype(int)
                        tex_y = ((1.0 - uv_coords[:, 1]) * (h - 1)).astype(int)  # Flip V for export
                        tex_x = np.clip(tex_x, 0, w - 1)
                        tex_y = np.clip(tex_y, 0, h - 1)
                        
                        # Create alpha texture by sampling vertex alpha at UV positions
                        alpha_tex = np.ones((h, w), dtype=np.float32)
                        # Accumulate alpha values at texture coordinates
                        for i, (x, y) in enumerate(zip(tex_x, tex_y)):
                            # Use max to handle multiple vertices mapping to same pixel
                            alpha_tex[y, x] = max(alpha_tex[y, x], alpha_values[i])
                        
                        # Convert to uint8
                        alpha_tex_uint8 = (np.clip(alpha_tex, 0.0, 1.0) * 255).astype(np.uint8)
                    else:
                        # Fallback: use mean alpha
                        alpha_tex_uint8 = np.full((h, w), int(alpha_values.mean() * 255), dtype=np.uint8)
                
                # Combine RGB + A -> RGBA
                tex_rgba = np.concatenate([tex_numpy, alpha_tex_uint8[..., np.newaxis]], axis=-1)
                tex_img = Image.fromarray(tex_rgba, mode='RGBA')
            else:
                tex_img = Image.fromarray(tex_numpy)

            # nvdiffrast uses flipped V. Export a flipped copy only.
            out_uv = self.uv.detach().cpu().numpy().copy()
            out_uv[:, 1] = 1.0 - out_uv[:, 1]

            from trimesh.visual.material import PBRMaterial
            material = PBRMaterial(
                baseColorTexture=tex_img,
                baseColorFactor=[255, 255, 255, 255], 
                metallicFactor=0.0,
                roughnessFactor=1.0
            )
            # Improve compatibility for viewers with limited glTF PBR support.
            try:
                material.doubleSided = True
                material.emissiveFactor = [0.0, 0.0, 0.0]
                # Use alpha mode if we have alpha channel
                if self.alpha is not None:
                    material.alphaMode = "BLEND"  # Support transparency
                else:
                    material.alphaMode = "OPAQUE"
            except Exception:
                pass
            visual = trimesh.visual.TextureVisuals(
                uv=out_uv,
                material=material
            )

        # Vertex-color branch (material=None).
        elif self.vertex_colors is not None:
            # Apply activation based on parameterization mode
            if self.material_trainable:
                if self.use_sigmoid_rgb:
                    vc_to_export = torch.sigmoid(self.vertex_colors)
                else:
                    vc_to_export = self._clamp_color(self.vertex_colors)
            else:
                vc_to_export = self.vertex_colors
            vc_numpy = (vc_to_export.detach().cpu().clamp(0.0, 1.0).numpy() * 255).astype(np.uint8)
            
            # Add alpha channel from differentiable alpha parameter
            if self.alpha is not None:
                # Get alpha values based on parameterization mode
                if self.use_sigmoid_alpha and isinstance(self.alpha, nn.Parameter):
                    alpha_values = torch.sigmoid(self.alpha).squeeze(0).detach().cpu().numpy()  # [V]
                else:
                    alpha_values = self._clamp_color(self.alpha).squeeze(0).detach().cpu().numpy()  # [V]
                alpha_uint8 = (np.clip(alpha_values, 0.0, 1.0) * 255).astype(np.uint8)
            else:
                # Default: fully opaque
                alpha_uint8 = np.full((vc_numpy.shape[0],), 255, dtype=np.uint8)
            
            vc_rgba = np.concatenate([vc_numpy, alpha_uint8[:, np.newaxis]], axis=1)
            visual = trimesh.visual.ColorVisuals(vertex_colors=vc_rgba)

        if visual is None:
            print("Error: No texture/UV or vertex colors available to save.")
            return

        export_vertex_rgba = None
        # If color is texture-driven but alpha is vertex-driven, persist alpha in vertex RGBA.
        if self.force_vertex_alpha and self.alpha is not None and self.alpha.dim() == 2:
            if self.use_sigmoid_alpha and isinstance(self.alpha, nn.Parameter):
                alpha_values = torch.sigmoid(self.alpha).squeeze(0).detach().cpu().numpy()  # [V]
            else:
                alpha_values = self._clamp_color(self.alpha).squeeze(0).detach().cpu().numpy()  # [V]
            alpha_uint8 = (np.clip(alpha_values, 0.0, 1.0) * 255).astype(np.uint8)

            if self.vertex_colors is not None:
                if self.material_trainable:
                    if self.use_sigmoid_rgb:
                        vc_to_export = torch.sigmoid(self.vertex_colors)
                    else:
                        vc_to_export = self._clamp_color(self.vertex_colors)
                else:
                    vc_to_export = self.vertex_colors
                vc_rgb_uint8 = (vc_to_export.detach().cpu().clamp(0.0, 1.0).numpy() * 255).astype(np.uint8)
            else:
                vc_rgb_uint8 = np.full((vertices.shape[0], 3), 255, dtype=np.uint8)

            export_vertex_rgba = np.concatenate([vc_rgb_uint8, alpha_uint8[:, np.newaxis]], axis=1)

        mesh_export = trimesh.Trimesh(
            vertices=vertices,
            faces=faces,
            visual=visual,
            vertex_colors=export_vertex_rgba,
            process=False
        )
        if export_vertex_rgba is not None:
            # trimesh glTF exporter may fail on uint8 vertex_attributes (KeyError: '|u1').
            # Write COLOR_0 as float32 [0,1] to keep exporter stable.
            color0_float = export_vertex_rgba.astype(np.float32) / 255.0
            try:
                if hasattr(mesh_export.visual, "vertex_attributes"):
                    mesh_export.visual.vertex_attributes["COLOR_0"] = color0_float
            except Exception:
                pass
            try:
                if hasattr(mesh_export, "vertex_attributes"):
                    mesh_export.vertex_attributes["COLOR_0"] = color0_float
            except Exception:
                pass
            sidecar_path = os.path.splitext(output_path)[0] + ".vertex_alpha.npy"
            np.save(sidecar_path, np.clip(alpha_values.astype(np.float32), 0.0, 1.0))
            print(f"Saved vertex alpha sidecar to {sidecar_path}")

        # Some viewers rely on explicit normals for shaded display.
        if hasattr(self, "normals") and self.normals is not None:
            mesh_export.vertex_normals = self.normals.detach().cpu().numpy()
        mesh_export.export(output_path)
        print("Done.")
