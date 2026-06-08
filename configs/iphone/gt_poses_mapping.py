"""
Run SplaTAM mapping-only with external poses (no tracking).

This script takes a transforms.json with poses from an external source
(e.g., COLMAP, ARKit, VGGT) and runs SplaTAM's mapping pipeline only,
using those poses as ground truth. This produces a high-quality Gaussian
map without relying on SplaTAM's tracking.

Usage:
  1. Prepare your dataset in nerfcapture format with good poses in transforms.json
  2. Set scene_name and num_frames below
  3. Run: python scripts/splatam.py configs/iphone/gt_poses_mapping.py

The key setting is:
  tracking > use_gt_poses = True
This tells SplaTAM to use the poses from transforms.json instead of
estimating them via tracking optimization.
"""
import os
from os.path import join as p_join

primary_device = "cuda:0"
seed = 0

# ========== EDIT THESE ==========
base_dir = "/mnt/ws-frb/users/ycs/datasets/splatam"
scene_name = "YOUR_SCENE_NAME"  # <-- CHANGE THIS
num_frames = -1                  # -1 = use all frames
# ================================

depth_scale = 10.0
overwrite = False

full_res_width = 1920
full_res_height = 1440
downscale_factor = 2.0
densify_downscale_factor = 4.0

map_every = 1
if num_frames > 0 and num_frames < 25:
    keyframe_every = int(num_frames // 5)
else:
    keyframe_every = 5
mapping_window_size = 32
# tracking_iters is irrelevant when use_gt_poses=True, but need to set it
tracking_iters = 1
mapping_iters = 100  # more mapping iters since we're not spending time on tracking

config = dict(
    workdir=f"./experiments/iphone/{scene_name}",
    run_name="gt_poses_mapping",
    overwrite=overwrite,
    depth_scale=depth_scale,
    num_frames=num_frames,
    seed=seed,
    primary_device=primary_device,
    map_every=map_every,
    keyframe_every=keyframe_every,
    mapping_window_size=mapping_window_size,
    report_global_progress_every=100,
    eval_every=1,
    scene_radius_depth_ratio=3,
    mean_sq_dist_method="projective",
    gaussian_distribution="isotropic",
    report_iter_progress=False,
    load_checkpoint=False,
    checkpoint_time_idx=0,
    save_checkpoints=False,
    checkpoint_interval=100,
    use_wandb=True,
    wandb=dict(
        entity="ycsun2113-university-of-michigan",
        project="SplaTAM",
        group="GT_Poses_Mapping",
        name=f"GTpose_{scene_name}_{seed}",
        save_qual=False,
        eval_save_qual=True,
    ),
    data=dict(
        dataset_name="nerfcapture",
        basedir=base_dir,
        sequence=scene_name,
        desired_image_height=int(full_res_height // downscale_factor),
        desired_image_width=int(full_res_width // downscale_factor),
        densification_image_height=int(full_res_height // densify_downscale_factor),
        densification_image_width=int(full_res_width // densify_downscale_factor),
        start=0,
        end=-1,
        stride=1,
        num_frames=num_frames,
    ),
    tracking=dict(
        # === KEY SETTING: Use poses from transforms.json ===
        use_gt_poses=True,
        forward_prop=True,
        visualize_tracking_loss=False,
        num_iters=tracking_iters,  # Not used when use_gt_poses=True
        use_sil_for_loss=True,
        sil_thres=0.99,
        use_l1=True,
        use_depth_loss_thres=False,
        depth_loss_thres=20000,
        ignore_outlier_depth_loss=False,
        use_uncertainty_for_loss_mask=False,
        use_uncertainty_for_loss=False,
        use_chamfer=False,
        loss_weights=dict(
            im=0.5,
            depth=1.0,
        ),
        lrs=dict(
            means3D=0.0,
            rgb_colors=0.0,
            unnorm_rotations=0.0,
            logit_opacities=0.0,
            log_scales=0.0,
            cam_unnorm_rots=0.0,  # No tracking optimization needed
            cam_trans=0.0,
        ),
    ),
    mapping=dict(
        num_iters=mapping_iters,
        add_new_gaussians=True,
        sil_thres=0.5,
        use_l1=True,
        ignore_outlier_depth_loss=False,
        use_sil_for_loss=False,
        use_uncertainty_for_loss_mask=False,
        use_uncertainty_for_loss=False,
        use_chamfer=False,
        loss_weights=dict(
            im=0.5,
            depth=1.0,
        ),
        lrs=dict(
            means3D=0.0001,
            rgb_colors=0.0025,
            unnorm_rotations=0.001,
            logit_opacities=0.05,
            log_scales=0.001,
            cam_unnorm_rots=0.0,  # Don't optimize poses during mapping
            cam_trans=0.0,
        ),
        prune_gaussians=True,
        pruning_dict=dict(
            start_after=0,
            remove_big_after=0,
            stop_after=20,
            prune_every=20,
            removal_opacity_threshold=0.005,
            final_removal_opacity_threshold=0.005,
            reset_opacities=True,
            reset_opacities_every=500,
        ),
        use_gaussian_splatting_densification=False,
        densify_dict=dict(
            start_after=500,
            remove_big_after=3000,
            stop_after=5000,
            densify_every=100,
            grad_thresh=0.0002,
            num_to_split_into=2,
            removal_opacity_threshold=0.005,
            final_removal_opacity_threshold=0.005,
            reset_opacities_every=3000,
        ),
    ),
    viz=dict(
        render_mode='color',
        offset_first_viz_cam=True,
        show_sil=False,
        visualize_cams=True,
        viz_w=600, viz_h=340,
        viz_near=0.01, viz_far=100.0,
        view_scale=2,
        viz_fps=5,
        enter_interactive_post_online=False,
    ),
)
