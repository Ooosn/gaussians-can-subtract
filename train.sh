#!/usr/bin/env bash

# One hybrid-training example. DATA and MESH are local inputs; change them for another scene.
DATA=data/scene_path
MESH=$DATA/scene.glb
OUT=runs/scene_path

python pretrain_mesh.py --source_path "$DATA" --mesh_path "$MESH" \
  --output_dir "$OUT/stage1_texture_3k" --iterations 3000 \
  --variants texture --black_background

python initialize_ndf.py --source_path "$DATA" \
  --stage1_checkpoint "$OUT/stage1_texture_3k/texture/iteration_3000" \
  --output_dir "$OUT/ndf_init" --uv_resolution 512 --black_background

python train_3dgs.py -s "$DATA" -m "$OUT/pure_depth9" \
  --depths "$DATA/depths" --data_device cpu --iterations 9000 \
  --test_iterations 3000 6000 9000 --save_iterations 3000 6000 9000 \
  --disable_viewer --eval

PURE9="$OUT/pure_depth9/point_cloud/iteration_9000/point_cloud.ply"
PURE9_COUNT=$(awk '$1 == "element" && $2 == "vertex" {print $3; exit}' "$PURE9")

python train.py --backbone 3dgs -s "$DATA" -m "$OUT/hybrid_9000" \
  --depths "$DATA/depths" --iterations 30000 \
  --test_iterations 9000 12000 15000 20000 25000 30000 \
  --save_iterations 9000 12000 15000 20000 25000 30000 \
  --disable_viewer --eval --full_train_metrics \
  --mesh_path "$MESH" \
  --stage1_appearance_path "$OUT/stage1_texture_3k/texture/iteration_3000/environment.pt" \
  --mesh_train --mesh_direct_rgb \
  --mesh_start_finetune_iteration 20000 \
  --mesh_lr_init 0.00025 --mesh_lr_final 0.000125 \
  --pretrained_main_ply_path "$PURE9" --pretrained_main_expected_count "$PURE9_COUNT" \
  --pretrained_main_start_iteration 9000 \
  --pretrained_main_color_mode reset --pretrained_main_opacity_mode reset \
  --pretrained_main_opacity_reset_value 0.01 \
  --hybrid_start_iteration 9000 \
  --main_densify_until_global 15000 --ndf_densify_until_global 15000 \
  --mesh_joint_start_global 20000 \
  --gaussolve_mode --gaussolve_densification \
  --external_ndf_init_mode direct_points \
  --external_ndf_init_path "$OUT/ndf_init/uv_loss_surface_points.pt" \
  --ndf_no_mesh_depth_cutoff \
  --seed_gaussians --seed_gaussians_gen_num_per_view 300

# Fixed-mesh variant: omit --mesh_train in the hybrid command.
