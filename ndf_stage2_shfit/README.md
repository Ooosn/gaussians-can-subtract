# NDF initialization

`uv_loss_surface_init.py` aggregates mesh reconstruction error in the UV atlas over training views, selects consistently high-error texels, maps them to surface points, and performs 3D voxel deduplication. The resulting `uv_loss_surface_points.pt` is consumed by the released hybrid trainers through `--external_ndf_init_mode direct_points`.

Use the public wrapper from the repository root:

```powershell
python initialize_ndf.py --help
```
