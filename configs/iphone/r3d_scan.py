import os
from os.path import join as p_join

primary_device = "cuda:0"
seed = 0

base_dir = "/mnt/ws-frb/users/ycs/datasets/splatam"  # Root Directory containing the dataset
# scene_name = "scan_01_0329"  # Scan Name
# num_frames = 1317  # Number of frames to use (-1 for all)
# scene_name = "scan_01_0329_sub60"  # Scan Name
# num_frames = 439  # Number of frames to use (-1 for all)
# scene_name = "scan_01_0329_sub15"  # Scan Name
# num_frames = 1756  # Number of frames to use (-1 for all)
scene_name = "r3d_0405_sub10_all"  # Scan Name
num_frames = 1004  # Number of frames to use (-1 for all)
depth_scale = 10.0  # Depth Scale used when saving depth
overwrite = False

# Record3D iPad Pro capture: 1440x1920 (portrait)
full_res_width = 1920
full_res_height = 1440
downscale_factor = 2.0  # Downscale to 360x480 for tracking (high res is slow)
densify_downscale_factor = 4.0

map_every = 1
if num_frames < 25:
    keyframe_every = int(num_frames // 5)
else:
    keyframe_every = 5
mapping_window_size = 32
tracking_iters = 60
mapping_iters = 60

config = dict(
    workdir=f"./experiments/iphone/{scene_name}",
    run_name="SplaTAM_R3D",
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
    checkpoint_time_idx=130,
    save_checkpoints=False,
    checkpoint_interval=5,
    use_wandb=True,
    wandb=dict(
        entity="ycsun2113-university-of-michigan",
        project="SplaTAM",
        group="iPhone_R3D",
        name=f"R3D_{scene_name}_{seed}",
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
        use_gt_poses=False,
        forward_prop=True,
        visualize_tracking_loss=False,
        num_iters=tracking_iters,
        use_sil_for_loss=True,
        sil_thres=0.99,
        use_l1=True,
        use_depth_loss_thres=True,
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
            cam_unnorm_rots=0.001,
            cam_trans=0.004,
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
            cam_unnorm_rots=0.0000,
            cam_trans=0.0000,
        ),
        prune_gaussians=True,
        pruning_dict=dict(
            start_after=0,
            remove_big_after=0,
            stop_after=20,
            prune_every=20,
            removal_opacity_threshold=0.005,
            final_removal_opacity_threshold=0.005,
            reset_opacities=False,
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
