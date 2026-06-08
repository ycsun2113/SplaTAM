#!/bin/bash
DATA_ROOT="/mnt/ws-frb/users/ycs/datasets/splatam"
LOG_ROOT="/mnt/ws-frb/users/ycs/exp_results/vggt-slam"
OUT_ROOT="/mnt/ws-frb/users/ycs/exp_results/gsplat"
SCENE_NAME="r3d_nerfstudio"
TIME_STAMP="2026-04-04_214453"

# ns-process-data record3d \
#     --data ${DATA_ROOT}/${SCENE_NAME}/raw \
#     --output_dir ${DATA_ROOT}/${SCENE_NAME}/processed \
#     --num-downscales 2 \
#     --max_dataset_size 1000 
#         # --ply ${LOG_ROOT}/r3d_0403_sub20_all_pointcloud.ply 

ns-train splatfacto \
    --data ${DATA_ROOT}/${SCENE_NAME}/vggt_processed \
    --output-dir ${OUT_ROOT} \
    --experiment-name ${SCENE_NAME} \
    --max-num-iterations 50000 \
    --pipeline.model.num-downscales 2 \
    --pipeline.model.cull-alpha-thresh 0.1 \
    --pipeline.model.reset-alpha-every 25 \
    --pipeline.model.sh-degree 2 \
    --pipeline.model.use-bilateral-grid True \
    --optimizers.means.optimizer.lr 0.0001 \
    --vis wandb






ns-train splatfacto \
    --data ${DATA_ROOT}/${SCENE_NAME}/processed \
    --output-dir ${OUT_ROOT}/${SCENE_NAME} \
    --experiment-name ${SCENE_NAME} \
    --load-dir ${OUT_ROOT}/${SCENE_NAME}/splatfacto/2026-04-04_214453/nerfstudio_models \
    --max-num-iterations 50000 \
    --pipeline.model.num-downscales 2 \
    --pipeline.model.reset-alpha-every 25 \
    --pipeline.model.sh-degree 2 \
    --vis wandb

ns-train splatfacto \
    --data ${DATA_ROOT}/${SCENE_NAME}/vggt_processed \
    --output-dir ${OUT_ROOT}/${SCENE_NAME} \
    --experiment-name ${SCENE_NAME} \
    --load-dir ${OUT_ROOT}/${SCENE_NAME}/splatfacto/2026-04-04_222859/nerfstudio_models \
    --max-num-iterations 50000 \
    --pipeline.model.num-downscales 2 \
    --pipeline.model.reset-alpha-every 25 \
    --pipeline.model.sh-degree 2 \
    --vis wandb

# ns-export gaussian-splat \
#     --load-config ${OUT_ROOT}/${SCENE_NAME}/splatfacto/${TIME_STAMP}/config.yml \
#     --output-dir ${OUT_ROOT}/${SCENE_NAME}/splatfacto/gsplat



ns-process-data record3d \
    --data ${DATA_ROOT}/${SCENE_NAME}/raw \
    --output_dir ${DATA_ROOT}/${SCENE_NAME}/processed \
    --num-downscales 2 \
    --max_dataset_size 1500 
        # --ply ${LOG_ROOT}/r3d_0403_sub20_all_pointcloud.ply

ns-train splatfacto \
    --data ${DATA_ROOT}/${SCENE_NAME}/processed \
    --output-dir ${OUT_ROOT} \
    --experiment-name ${SCENE_NAME} \
    --max-num-iterations 50000 \
    --vis wandb