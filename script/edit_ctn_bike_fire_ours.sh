ROOT=/data/b09401064/tmp/ev/final/style-and-object-personalized-gaussian-editor
DATA_DIR=${ROOT}/dataset

python launch.py --config configs/edit-ctn.yaml \
--train --gpu 0 \
trainer.max_steps=1600 system.cache_dir="ground_fire" system.seg_prompt="grass" system.prompt_processor.prompt="make the grass on fire"  system.max_densify_percent=0.01 system.anchor_weight_init_g0=1 system.anchor_weight_init=0.5 system.anchor_weight_multiplier=1.5  system.loss.lambda_anchor_color=0 system.loss.lambda_anchor_geo=5 system.loss.lambda_anchor_scale=5 system.loss.lambda_anchor_opacity=0 system.densify_from_iter=100 system.densify_until_iter=5000 system.densification_interval=300 \
data.source=${DATA_DIR}/bicycle \
system.gs_source=${DATA_DIR}/bicycle/point_cloud/iteration_30000/point_cloud.ply \
system.loggers.wandb.enable=true system.loggers.wandb.name="edit_ctn_bike_fire"