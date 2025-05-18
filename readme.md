# EV 2025 Spring final project

This project is a clone and extension from the original work [buaacyw/GaussianEditor](https://github.com/buaacyw/GaussianEditor).

## Install environment
```bash
conda env create -f environment.yaml
conda activate GaussianEditor

pip install -r base_requirements.txt
pip install --no-deps -r git_requirements.txt

bash download.sh # download dataset and checkpoints to dataset/
bash download_wonder3d.sh # download checkpoints to ckpt/

# Download DPT models, ref: https://github.com/EPFL-VILAB/omnidata
mkdir dpt
cd dpt
wget 'https://zenodo.org/records/10447888/files/omnidata_dpt_depth_v2.ckpt?download=1'
cd ..
```

To use Gaussian Editor's web UI, install their modified version of viser:
```bash
mkdir extern && cd extern
git clone https://github.com/heheyas/viser 
pip install -e viser
cd ..
```
For the "added" function, download the Wonder3D dataset:
```bash
sh download_wonder3d.sh
```

To load the model before launching the server: 
```bash
$ python
>>> import torch
>>> from diffusers import (
    StableDiffusionControlNetInpaintPipeline,
    ControlNetModel,
    DDIMScheduler,
)

controlnet = ControlNetModel.from_pretrained(
    "lllyasviel/control_v11p_sd15_inpaint", torch_dtype=torch.float16
)
pipe = StableDiffusionControlNetInpaintPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    controlnet=controlnet,
    torch_dtype=torch.float16,
)
pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
```

To launch the web UI server, 
```bash
python webui.py --gs_source <your-ply-file> --colmap_dir <dataset-dir>
```
where `--gs_source` refers to the pre-trained .ply file (something like ../../point_cloud.ply), and `--colmap_dir` refers to where the Colmap output resides (the colmap output sparse folder should be the subfolder of `--colmap_dir`).
See the [original repo](https://github.com/buaacyw/GaussianEditor/blob/master/docs/webui.md) for more information.

## Checkpoint conversion
### Custom diffusion
Custom diffusion provides a script to convert the CompVis `delta_model.ckpt` to Diffusers `delta.bin`, please refer the [repo](https://github.com/adobe-research/custom-diffusion/tree/main?tab=readme-ov-file#checkpoint-conversions-for-stable-diffusion-v1-4).

### Alternative method
If the method above fails, follow the steps below to perform the conversion.

1. Clone the Custom-diffusion and Diffusers repositories.
```bash
git clone https://github.com/adobe-research/custom-diffusion.git
git clone https://github.com/huggingface/diffusers.git
```
2. Add the following line to `custom-diffusion/sample.py`:
```python
torch.save(model.state_dict(), "delta_model.ckpt")
```
3. Run the following commands to save `delta_model.ckpt`:
```bash
cd custom-diffusion
python sample.py --prompt <prompt> --delta_ckpt <the downloaded checkpoint path> --ckpt <pretrained-model-path>
```
4. Run the following commands to perform the conversion:
```bash
cd ../diffusers
python scripts/convert_original_stable_diffusion_to_diffusers.py --checkpoint_path ../custom-diffusion/delta_model.ckpt --dump_path <the path to save the model> --original_config_file ../custom-diffusion/stable-diffusion/configs/stable-diffusion/v1-inference.yaml
```
5. After completing the steps above, you can use the model as follows:
```python
import torch
from diffusers import StableDiffusionPipeline

pipe = StableDiffusionPipeline.from_pretrained(<dump_path>).to("cuda")
img = pipe(prompt, num_inference_steps=100, guidance_scale=6.0, eta=1.0)
img[0][0].save(filename)
```