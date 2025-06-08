accelerate launch src/diffusers_training.py \
--pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4" \
--instance_data_dir=./data/$1/ --instance_prompt="$2" \
 --resolution=512 --train_batch_size=2 --learning_rate=1e-5 --lr_warmup_steps=0 \
 --max_train_steps=2000 --scale_lr --hflip --modifier_token "$3" --modelname "delta_$1.bin" \
 --initializer_token "$4"

# LD_LIBRARY_PATH="/data/b09401064/tmp/anaconda/lib" python src/convert.py --ckpt sd-v1-4.ckpt \
# --delta_ckpt custom-diffusion-model/delta_$1.bin --modelname delta_$1.ckpt --mode diffuser-to-compvis
# LD_LIBRARY_PATH="/data/b09401064/tmp/anaconda/lib" python convert_ckpt.py --delta_ckpt custom-diffusion-model/delta_$1.ckpt \
# --ckpt sd-v1-4.ckpt --dump_path ../model_$1 --original_config_file stable-diffusion/configs/stable-diffusion/v1-inference.yaml