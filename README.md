# Gaussians Can Subtract: Neural Dissolve Fields for Hybrid Scene Rendering

**WACV 2027** · [Project page](https://ooosn.github.io/gaussians-can-subtract/) · [Paper (PDF)](https://ooosn.github.io/gaussians-can-subtract/assets/paper.pdf) · [Supplementary (PDF)](https://ooosn.github.io/gaussians-can-subtract/assets/supplementary.pdf) · arXiv (coming soon)

Yiyan Wang<sup>1,\*</sup>, Haohan Liu<sup>2,\*</sup>, Tetsuo Shibuya<sup>1</sup><br>
<sup>1</sup>The University of Tokyo, <sup>2</sup>Fuzhou University, <sup>\*</sup>Equal contribution

Official code release of **Gaussolve**. Instead of only adding Gaussian residuals on top of a textured mesh, Gaussolve learns a view-dependent Neural Dissolve Field that selectively *subtracts* inconsistent mesh content, and lets residual Gaussians *refill* the revealed regions through continuous transmittance fusion.

The release contains only the three reported training routes:

- `3dgs`: standard 3DGS residual backbone;
- `mcmc`: 3DGS-MCMC residual backbone;
- `reduced`: Full Reduced-3DGS residual backbone with SH sparsity, multi-view SH-band culling, mercy pruning, and variable-degree rendering.

The mesh, scalar-SH Neural Dissolve Field (NDF), continuous fusion, and loss-driven dual densification are shared by all three routes.

## Environment

Create a CUDA-enabled PyTorch environment, install `requirements.txt`, and build the bundled extensions:

```powershell
python -m pip install -r requirements.txt
.\install_extensions.ps1
```

This release contains code only; scene data and meshes are not included. Set `DATA` and `MESH` to your local prepared scene and mesh. The scene must follow the usual 3DGS/COLMAP layout. When depth supervision is used, supply Depth Anything maps with `--depths PATH`; otherwise omit that option. Hybrid training proceeds through mesh appearance, NDF initialization, 9k Gaussian warm-up, and the final hybrid stage.

## 1. Mesh appearance

The mesh geometry remains fixed. This stage optimizes the projected RGB texture using training views only:

```powershell
python pretrain_mesh.py --source_path DATASET --mesh_path MESH.glb `
  --output_dir OUTPUT\mesh --iterations 3000 --variants texture `
  --black_background
```

The resulting checkpoint contains `mesh_pretrained.glb` and `environment.pt`.

## 2. Multi-view mesh-texel NDF initialization

```powershell
python initialize_ndf.py --source_path DATASET `
  --stage1_checkpoint OUTPUT\mesh\texture\iteration_3000 `
  --output_dir OUTPUT\ndf_init --uv_resolution 512 --min_views 2 `
  --voxel_size 0.04 --black_background
```

The hybrid trainer consumes `OUTPUT\ndf_init\uv_loss_surface_points.pt`.

## 3. Gaussian warm-up

For the default route, train the residual 3DGS checkpoint to the handoff iteration:

```powershell
python train_3dgs.py -s DATASET -m OUTPUT\warmup --iterations 9000 `
  --save_iterations 9000 --eval
```

For MCMC and Reduced variants, use the corresponding official backbone to produce its own 9k checkpoint. Reduced grouped checkpoints can be converted to the common SH3 handoff layout with:

```powershell
python tools\convert_reduced_checkpoint.py --help
```

## 4. Hybrid training

Use the common public entry point and select the residual backbone:

```powershell
python train.py --backbone 3dgs [GAUSSOLVE OPTIONS]
python train.py --backbone mcmc [GAUSSOLVE OPTIONS] [MCMC OPTIONS]
python train.py --backbone reduced [GAUSSOLVE OPTIONS] [REDUCED OPTIONS]
```

Run `python train.py --backbone BACKBONE --help` for the complete argument list. The released trainers save an exact render-state bundle at the final iteration so that offline rendering uses the same mesh, NDF, cameras, and fusion state as training.

### Training script (3DGS backbone)

The short `train.sh` is a single Playroom example and runs the four commands
from Sections 1--4:

```bash
bash train.sh
```

Set the local `DATA`, `MESH`, and `OUT` paths at the top of `train.sh` for the
scene to process. The default command enables mesh/appearance joint
optimization from global iteration 20,000; omit `--mesh_train` for the
fixed-mesh variant.

## Rendering and decomposition

Verify a saved bundle and render its test-view decomposition:

```powershell
python render.py verify --report-dir OUTPUT\exact_render_state
python render.py decompose --report-dir OUTPUT\exact_render_state `
  --output-dir OUTPUT\decomposition
```

The decomposition includes GT, mesh, final rendering, raw dissolve field, erase/refill terms, mesh contribution, and residual contribution. `Final` is checked against the image saved by the training report before any PNG is accepted.

## Rendering benchmark

`tools/benchmark.py` measures mesh-only, NDF-only, residual-only, baseline GS, serial Gaussolve, and three-stream Gaussolve. The parallel renderer uses persistent CUDA streams and independent host dispatch after the shared mesh raster/depth pass. Serial and parallel images must pass the built-in equality gate before timings are reported.

```powershell
python tools\benchmark.py --help
```

Model loading, disk I/O, image saving, metrics, and depth visualization are outside the timed region. Mesh rasterization remains inside every timed frame.

## Citation

```bibtex
@inproceedings{wang2027gaussians,
  title     = {Gaussians Can Subtract: Neural Dissolve Fields for Hybrid Scene Rendering},
  author    = {Wang, Yiyan and Liu, Haohan and Shibuya, Tetsuo},
  booktitle = {Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision (WACV)},
  year      = {2027}
}
```

## License

This code builds on [3D Gaussian Splatting](https://github.com/graphdeco-inria/gaussian-splatting) and is released under the Gaussian-Splatting License (see [LICENSE.md](LICENSE.md)), which permits non-commercial research and evaluation use only. Bundled third-party components (for example GLM under `submodules/*/third_party/glm`) keep their own licenses.

The project page lives in [`docs/`](docs/) and is served with GitHub Pages.
