# EV 2025 Spring final project

**Group 5: GaussianEditor-based Style transfer and Object Personalization Inpainting**

This project is a clone and extension from the original work [buaacyw/GaussianEditor](https://github.com/buaacyw/GaussianEditor).

To see our results, please refer to our [poster](https://drive.google.com/file/d/1GaxOfQgjf1cF90tIuHQ_ynHVL5bC4QfG/view?usp=drive_link).

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
wget https://zenodo.org/records/10447888/files/omnidata_dpt_depth_v2.ckpt
cd ..

# Create ldm environment for custom-diffusion and InST
cd custom-diffusion
git clone https://github.com/CompVis/stable-diffusion.git
cd stable-diffusion
conda env create -f environment.yaml
conda activate ldm
pip install clip-retrieval tqdm
cd ../../

# conda activate ldm
git clone https://github.com/huggingface/diffusers.git
# cd diffusers
# pip install -e .
# cd ../
```

To use Gaussian Editor's web UI, install their modified version of viser:
```bash
mkdir extern && cd extern
git clone https://github.com/heheyas/viser 
pip install -e viser
cd ..
```
<!-- For the "added" function, download the Wonder3D dataset:
```bash
sh download_wonder3d.sh
``` -->

<!-- To load the model before launching the server: 
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
``` -->

To launch the web UI server, 
```bash
python webui.py --gs_source <your-ply-file> --colmap_dir <dataset-dir>
```
where `--gs_source` refers to the pre-trained .ply file (something like ../../point_cloud.ply), and `--colmap_dir` refers to where the Colmap output resides (the colmap output sparse folder should be the subfolder of `--colmap_dir`).
See the [original repo](https://github.com/buaacyw/GaussianEditor/blob/master/docs/webui.md) for more information.

## Checkpoint conversion
### Custom diffusion
<!-- Custom diffusion provides a script to convert the CompVis `delta_model.ckpt` to Diffusers `delta.bin`, please refer the [repo](https://github.com/adobe-research/custom-diffusion/tree/main?tab=readme-ov-file#checkpoint-conversions-for-stable-diffusion-v1-4). -->

<!-- ### Alternative method (1)
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

### Alternative method (2) -->

``` bash
cd custom-diffusion
# Download stable diffusion v1.4 checkpoint
wget https://huggingface.co/CompVis/stable-diffusion-v-1-4-original/resolve/main/sd-v1-4.ckpt

# Download the delta checkpoint and convert
wget <the downloaded delta ckpt path> # *.ckpt
python convert_ckpt.py \
    --delta_ckpt <the downloaded delta ckpt path> \
    --ckpt sd-v1-4.ckpt \
    --dump_path <the path to save the converted model folder> \
    --original_config_file stable-diffusion/configs/stable-diffusion/v1-inference.yaml
```
This will create a CompVis-formatted model checkpoint directory at root directory.

### InST
```bash
conda activate ldm
cd InST
python convert.py --log_dir <logdir/name/under/InST/logs>

cd ../diffusers/scripts
python convert_original_stable_diffusion_to_diffusers.py \
    --checkpoint_path ../../../InST/logs/<logdir>/checkpoints/model.pt \
    --original_config_file ../../../InST/configs/stable-diffusion/v1-inference.yaml \
    --dump_path </path/to/save/ckpt/dir>
```
E.g.
```bash
python convert.py --log_dir van-gogh2025-06-07T16-10-18_van-gogh

python convert_original_stable_diffusion_to_diffusers.py --checkpoint_path ../../../InST/logs/van-gogh2025-06-07T16-10-18_van-gogh/checkpoints/model.pt --original_config_file ../../../InST/configs/stable-diffusion/v1-inference.yaml --dump_path ../../model_van-gogh_inst
```
The first command creates `model.pt` under `InST/logs/van-gogh2025-06-07T16-10-18_van-gogh/checkpoints`. 
The second one creates `model_van-gogh_inst` at root directory in CompVis format.


## Custom object insertion
There are two ways to insert custom objects: custom diffusion and copy-paste.

### Custom diffusion
#### Model preparation
To use this method, you need to prepare a custom diffusion model.
Some checkpoints can be downloaded from the [custom diffusion webpage](https://www.cs.cmu.edu/~custom-diffusion/assets/models/).
If you use the checkpoint provided above, please refer to [Checkpoint conversion](#checkpoint-conversion) section to convert it into the diffuser-compatible format.
You can also train your own custom diffusion model. Please refer to the [Training concept](#training-concept) section for more details. 

#### Object insertion
Once you have prepared the model, you can generate an object based on it and insert it into the scene using the following steps.

1. Activate WebUI in browser.
    ```bash
    python webui_custom.py --gs_source </path/to/point_cloud> \
        --colmap_dir <path/to/colmap_dir> \
        --model_type custom \
        --custom_diffusion_model <path/to/model.pt>
    ```
2. In the WebUI browser, select `Edit type` as "add", enter your prompt, and draw a bounding box. Then, click "Edit begin" to generate the object inside the box.
3. Click "End Inpainting" to start converting the 2D object into a 3D object. This process may take around 2 hours, depending on your device.

### Copy paste
If you do not want to train a custom diffusion model, you can also paste your object directly into the scene. Please follow the steps below.

1. Activate WebUI in browser.
    ```bash
    python webui_custom.py --gs_source </path/to/point_cloud> \
        --colmap_dir <path/to/colmap_dir> \
        --model_type paste \
        --object_image <path/to/object_image>
    ```
2. In the WebUI browser, select `Edit type` as "add", and draw a bounding box. Then, click "Edit begin" to paste the object inside the box.
3. Click "End Inpainting" to start converting the 2D object into a 3D object. This process may take around 2 hours, depending on your device.

Note that the height-to-width ratio of the object image and the bounding box should be similar; otherwise, the object may look distorted due to resizing.

### Examples
We provide some point clouds [here](https://drive.google.com/drive/folders/1Kxhwf_KlD1zHkEPO-Ws0BzVH2dgkYSpD?usp=drive_link). Download the `demo.zip` file to access six trained point clouds along with their corresponding inpainted images.


## Style transfer editing

You can download the following files [here](https://drive.google.com/drive/folders/1bgpC7f8YVUshELqxcDaTmVHFY5uHKXxZ?usp=sharing), which contains the following subfolders:
- data
    - place it under `custom-diffusion/` for following training.
- trained concept checkpoints
    - here we only provide the "delta" checkpoint (Custom Diffusion) and embeddings (InST). To create full models, please refer to the [Checkpoint Conversion](#checkpoint-conversion) section.
- trained point clouds

### Training concept
#### Custom Diffusion
```bash
conda activate ldm
cd custom-diffusion

bash train_new_concept.sh {name} {instance_data_dir} {instance_prompt} {modifier_token} {initializer_token}
```
with 
- `name`: name of the task
- `instance_data_dir`: the directory containing training data images
- `instance_prompt`: prompt for training
- `modifier_token`: the trained novel token
- `initializer_token`: the tokens of similar concept to initialize the novel token (can contain multiple concepts separated by `+`).

E.g.: 
```bash
bash train_new_concept.sh van-gogh data/van-gogh "a painting of <new1> style" "<new1>" "vangogh+art+painting"
```

This will create a `delta_{name}.bin` under the `custom-diffusion/custom-diffusion-model`, and is converted into `delta_{name}.ckpt` and `model_{name}` as CompVis format.

#### InST
```bash
conda activate ldm
cd InST

python main.py --base configs/stable-diffusion/v1-finetune.yaml  \
-t --actual_resume ../custom-diffusion/sd-v1-4.ckpt \
-n <taskname> --gpus <gpu no. separated by ","> \
--data_root <path/to/image/directory>
```
E.g.
``` bash
python main.py --base configs/stable-diffusion/v1-finetune.yaml  \
-t --actual_resume ../custom-diffusion/sd-v1-4.ckpt \
-n van-gogh --gpus "0," \
--data_root ../custom-diffusion/data/van-gogh/
```

### Editing: GaussianEditor
#### Custom Diffusion
1. Activate WebUI in browser.
    ```bash
    conda activate GaussianEditor

    python webui_custom.py --gs_source <path/to/point_cloud> \
        --colmap_dir <path/to/colmap_dir> \
        --custom_diffusion_model <path/to/model/dir>
    ```
    E.g.
    ```bash
    python webui_custom.py \
        --gs_source dataset/bicycle/point_cloud/iteration_7000/point_cloud.ply \
        --colmap_dir dataset/bicycle/ \
        --custom_diffusion_model model_van-gogh/
    ```
    and open the viser UI at http://localhost:8084 or whatever port shown on screen.

2. In the WebUI browser, select "InstructPix2Pix" under the guidance type.
3. Enter prompt in the box. E.g. "turn into \<new1\> style."
4. Click "Edit begin" to launch the editting process.
5. The final point cloud will be saved as `ui_result/<prompt>.ply`.

#### InST

1. Activate WebUI in browser.
    ```bash
    conda activate GaussianEditor

    python webui_custom.py --gs_source </path/to/point_cloud> \
        --colmap_dir <path/to/colmap_dir> \
        --embedding_path <path/to/model.pt> \
        --style_image <path/to/style_image>
    ```
    E.g.
    ```
    python webui_custom.py \
        --gs_source dataset/bicycle/point_cloud/iteration_7000/point_cloud.ply \
        --colmap_dir dataset/bicycle/ \
        --embedding_path InST/logs/van-gogh2025-06-07T16-10-18_van-gogh/checkpoints/embeddings.pt \
        --style_image custom-diffusion/data/van-gogh/sunflower.jpg
    ```
    and open the viser UI at http://localhost:8084 or whatever port shown on screen.

2. In the WebUI browser, select "InST" under the guidance type.
3. Enter whatever into the prompt box (since we will not use it in InST conversion).
4. Click "Edit begin" to launch the editting process.
5. The final point cloud will be saved as `ui_result/<prompt>.ply`.
