ROOT=/data/b09401064/tmp/ev/final/style-and-object-personalized-gaussian-editor
DATA_DIR=${ROOT}/dataset

python launch.py --config configs/add.yaml \
    --train --gpu 1 \
    data.source=${DATA_DIR}/bicycle \
    system.gs_source=${DATA_DIR}/bicycle/point_cloud/iteration_7000/point_cloud.ply \
    system.inpaint_prompt="a cat on the bench" \
    system.refine_prompt="make it a cat" \
    system.cache_overwrite=False system.cache_dir="add_cat"  \
    trainer.max_steps=1  \
    system.loggers.wandb.enable=true \
    system.loggers.wandb.name="add_cat"