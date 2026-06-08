from os.path import join as p_join

primary_device = "cuda:0"

base_dir = "/mnt/ws-frb/users/ycs/datasets/splatam"
scene_name = "r3d_0405_sub10_all"
params_path = f"./experiments/iphone/{scene_name}/SplaTAM_R3D/params.npz"

group_name = "iPhone_R3D"
run_name = f"{scene_name}_post_splatam_opt"

# Record3D iPad Pro capture: 1440x1920 (portrait)
full_res_width = 1920
full_res_height = 1440
downscale_factor = 2.0
densify_downscale_factor = 4.0

config = dict(
    workdir=f"./experiments/iphone/{scene_name}",
    run_name=run_name,
    seed=0,
    primary_device=primary_device,
    mean_sq_dist_method="projective",
    gaussian_distribution="isotropic",
    report_iter_progress=False,
    use_wandb=True,
    wandb=dict(
        entity="ycsun2113-university-of-michigan",
        project="SplaTAM",
        group=group_name,
        name=run_name,
        save_qual=False,
        eval_save_qual=True,
    ),
    data=dict(
        dataset_name="nerfcapture",
        basedir=base_dir,
        sequence=scene_name,
        downscale_factor=downscale_factor,
        densify_downscale_factor=densify_downscale_factor,
        desired_image_height=int(full_res_height // downscale_factor),
        desired_image_width=int(full_res_width // downscale_factor),
        densification_image_height=int(full_res_height // densify_downscale_factor),
        densification_image_width=int(full_res_width // densify_downscale_factor),
        start=0,
        end=-1,
        stride=1,
        num_frames=-1,
        eval_stride=1,
        eval_num_frames=-1,
        param_ckpt_path=params_path,
    ),
    train=dict(
        num_iters_mapping=15000,
        sil_thres=0.4,
        use_sil_for_loss=True,
        loss_weights=dict(
            im=0.7,
            depth=0.9,
        ),
        lrs_mapping=dict(
            means3D=0.00032,
            rgb_colors=0.0025,
            unnorm_rotations=0.001,
            logit_opacities=0.05,
            log_scales=0.005,
            cam_unnorm_rots=0.0000,
            cam_trans=0.0000,
        ),
        lrs_mapping_means3D_final=0.0000032,
        lr_delay_mult=0.01,
        use_gaussian_splatting_densification=True,
        densify_dict=dict(
            start_after=100,
            remove_big_after=1000,
            stop_after=15000,
            densify_every=100,
            grad_thresh=0.0002,
            num_to_split_into=2,
            removal_opacity_threshold=0.005,
            final_removal_opacity_threshold=0.005,
            reset_opacities=True,
            reset_opacities_every=1000,
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
        enter_interactive_post_online=True,
    ),
)
