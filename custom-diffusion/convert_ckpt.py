# This code is built from the Stable Diffusion repository: https://github.com/CompVis/stable-diffusion.
# Copyright (c) 2022 Robin Rombach and Patrick Esser and contributors.
# CreativeML Open RAIL-M
#
# ==========================================================================================
#
# Adobe’s modifications are Copyright 2022 Adobe Research. All rights reserved.
# Adobe’s modifications are licensed under the Adobe Research License. To view a copy of the license, visit
# LICENSE.md.
#
# ==========================================================================================
#
# This file is the combination of custom_diffusion/sample.py and 
# diffusers/scripts/convert_original_stable_diffusion_to_diffusers.py.
# ==========================================================================================

"""
# At root directory

git clone https://github.com/huggingface/diffusers.git
cd diffusers
pip install -e .
cd ../

cd custom-diffusion
# Download stable diffusion v1.4 checkpoint
wget https://huggingface.co/CompVis/stable-diffusion-v-1-4-original/resolve/main/sd-v1-4.ckpt

# Download the delta checkpoint
wget <the downloaded delta ckpt path>
python convert_ckpt.py \
    --delta_ckpt <the downloaded delta ckpt path> \
    --ckpt sd-v1-4.ckpt \
    --dump_path <the path to save the converted model> \
    --original_config_file stable-diffusion/configs/stable-diffusion/v1-inference.yaml
"""

import argparse, os, sys, glob
sys.path.append('stable-diffusion')
import torch
import numpy as np
from omegaconf import OmegaConf
from PIL import Image
from tqdm import tqdm, trange
from einops import rearrange
from torchvision.utils import make_grid
from pytorch_lightning import seed_everything
from torch import autocast
from contextlib import contextmanager, nullcontext

from ldm.util import instantiate_from_config
from ldm.models.diffusion.ddim import DDIMSampler
from ldm.models.diffusion.plms import PLMSSampler
from diffusers.pipelines.stable_diffusion.convert_from_ckpt import download_from_original_stable_diffusion_ckpt

def load_model_from_config(config, ckpt, verbose=False):
    print(f"Loading model from {ckpt}")
    pl_sd = torch.load(ckpt, map_location="cpu")
    if "global_step" in pl_sd:
        print(f"Global Step: {pl_sd['global_step']}")
    sd = pl_sd["state_dict"]
    model = instantiate_from_config(config.model)

    token_weights = sd["cond_stage_model.transformer.text_model.embeddings.token_embedding.weight"]
    del sd["cond_stage_model.transformer.text_model.embeddings.token_embedding.weight"]
    m, u = model.load_state_dict(sd, strict=False)
    model.cond_stage_model.transformer.text_model.embeddings.token_embedding.weight.data[:token_weights.shape[0]] = token_weights
    if len(m) > 0 and verbose:
        print("missing keys:")
        print(m)
    if len(u) > 0 and verbose:
        print("unexpected keys:")
        print(u)

    model.cuda()
    model.eval()
    return model


def main(args):

    if args.delta_ckpt is not None:
        if len(glob.glob(os.path.join(args.delta_ckpt.split('checkpoints')[0], "configs/*.yaml"))) > 0:
            args.config = sorted(glob.glob(os.path.join(args.delta_ckpt.split('checkpoints')[0], "configs/*.yaml")))[-1]
    else:
        if len(glob.glob(os.path.join(args.ckpt.split('checkpoints')[0], "configs/*.yaml"))) > 0:
            args.config = sorted(glob.glob(os.path.join(args.ckpt.split('checkpoints')[0], "configs/*.yaml")))[-1]

    seed_everything(args.seed)
    config = OmegaConf.load(f"{args.config}")
    if args.modifier_token is not None:
        config.model.params.cond_stage_config.target = 'src.custom_modules.FrozenCLIPEmbedderWrapper'
        config.model.params.cond_stage_config.params = {}
        config.model.params.cond_stage_config.params.modifier_token = args.modifier_token
    model = load_model_from_config(config, f"{args.ckpt}")

    if args.delta_ckpt is not None:
        delta_st = torch.load(args.delta_ckpt)
        embed = None
        if 'embed' in delta_st['state_dict']:
            embed = delta_st['state_dict']['embed'].reshape(-1,768)
            del delta_st['state_dict']['embed']
            print(embed.shape)
        delta_st = delta_st['state_dict']
        if args.compress:
            for name in delta_st.keys():
                if 'to_k' in name or 'to_v' in name:
                    delta_st[name] = model.state_dict()[name] + delta_st[name]['u']@delta_st[name]['v']
            model.load_state_dict(delta_st, strict=False)
        else:
            model.load_state_dict(delta_st, strict=False)
        if embed is not None:
            print("loading new embedding")
            print(model.cond_stage_model.transformer.text_model.embeddings.token_embedding.weight.data.shape)
            model.cond_stage_model.transformer.text_model.embeddings.token_embedding.weight.data[-embed.shape[0]:] = embed

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model = model.to(device)
    torch.save(model.state_dict(), "delta_model.ckpt")


    if args.pipeline_class_name is not None:
        library = importlib.import_module("diffusers")
        class_obj = getattr(library, args.pipeline_class_name)
        pipeline_class = class_obj
    else:
        pipeline_class = None

    pipe = download_from_original_stable_diffusion_ckpt(
        checkpoint_path_or_dict="delta_model.ckpt",
        original_config_file=args.original_config_file,
        config_files=args.config_files,
        image_size=args.image_size,
        prediction_type=args.prediction_type,
        model_type=args.pipeline_type,
        extract_ema=args.extract_ema,
        scheduler_type=args.scheduler_type,
        num_in_channels=args.num_in_channels,
        upcast_attention=args.upcast_attention,
        from_safetensors=args.from_safetensors,
        device=args.device,
        stable_unclip=args.stable_unclip,
        stable_unclip_prior=args.stable_unclip_prior,
        clip_stats_path=args.clip_stats_path,
        controlnet=args.controlnet,
        vae_path=args.vae_path,
        pipeline_class=pipeline_class,
    )

    if args.half:
        pipe.to(dtype=torch.float16)

    if args.controlnet:
        # only save the controlnet model
        pipe.controlnet.save_pretrained(args.dump_path, safe_serialization=args.to_safetensors)
    else:
        pipe.save_pretrained(args.dump_path, safe_serialization=args.to_safetensors)

    os.remove("delta_model.ckpt")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    
    parser.add_argument(
        "--ckpt",
        type=str,
        required=True,
        help="path to checkpoint of the pre-trained model",
    )
    parser.add_argument(
        "--delta_ckpt",
        type=str,
        default=None,
        help="path to delta checkpoint of fine-tuned custom diffusion block",
    )
    # !wget https://raw.githubusercontent.com/CompVis/stable-diffusion/main/configs/stable-diffusion/v1-inference.yaml
    parser.add_argument(
        "--original_config_file",
        default=None,
        type=str,
        help="The YAML config file corresponding to the original architecture.",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/custom-diffusion/finetune.yaml",
        help="path to config which constructs model",
    )

    
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="the seed (for reproducible sampling)",
    )
    parser.add_argument(
        "--config_files",
        default=None,
        type=str,
        help="The YAML config file corresponding to the architecture.",
    )
    parser.add_argument(
        "--num_in_channels",
        default=None,
        type=int,
        help="The number of input channels. If `None` number of input channels will be automatically inferred.",
    )
    parser.add_argument(
        "--scheduler_type",
        default="pndm",
        type=str,
        help="Type of scheduler to use. Should be one of ['pndm', 'lms', 'ddim', 'euler', 'euler-ancestral', 'dpm']",
    )
    parser.add_argument(
        "--pipeline_type",
        default=None,
        type=str,
        help=(
            "The pipeline type. One of 'FrozenOpenCLIPEmbedder', 'FrozenCLIPEmbedder', 'PaintByExample'"
            ". If `None` pipeline will be automatically inferred."
        ),
    )
    parser.add_argument(
        "--image_size",
        default=None,
        type=int,
        help=(
            "The image size that the model was trained on. Use 512 for Stable Diffusion v1.X and Stable Diffusion v2"
            " Base. Use 768 for Stable Diffusion v2."
        ),
    )
    parser.add_argument(
        "--prediction_type",
        default=None,
        type=str,
        help=(
            "The prediction type that the model was trained on. Use 'epsilon' for Stable Diffusion v1.X and Stable"
            " Diffusion v2 Base. Use 'v_prediction' for Stable Diffusion v2."
        ),
    )
    parser.add_argument(
        "--extract_ema",
        action="store_true",
        help=(
            "Only relevant for checkpoints that have both EMA and non-EMA weights. Whether to extract the EMA weights"
            " or not. Defaults to `False`. Add `--extract_ema` to extract the EMA weights. EMA weights usually yield"
            " higher quality images for inference. Non-EMA weights are usually better to continue fine-tuning."
        ),
    )
    parser.add_argument(
        "--upcast_attention",
        action="store_true",
        help=(
            "Whether the attention computation should always be upcasted. This is necessary when running stable"
            " diffusion 2.1."
        ),
    )
    parser.add_argument(
        "--from_safetensors",
        action="store_true",
        help="If `--checkpoint_path` is in `safetensors` format, load checkpoint with safetensors instead of PyTorch.",
    )
    parser.add_argument(
        "--to_safetensors",
        action="store_true",
        help="Whether to store pipeline in safetensors format or not.",
    )
    parser.add_argument("--dump_path", default=None, type=str, required=True, help="Path to the output model.")
    parser.add_argument("--device", type=str, help="Device to use (e.g. cpu, cuda:0, cuda:1, etc.)")
    parser.add_argument(
        "--stable_unclip",
        type=str,
        default=None,
        required=False,
        help="Set if this is a stable unCLIP model. One of 'txt2img' or 'img2img'.",
    )
    parser.add_argument(
        "--stable_unclip_prior",
        type=str,
        default=None,
        required=False,
        help="Set if this is a stable unCLIP txt2img model. Selects which prior to use. If `--stable_unclip` is set to `txt2img`, the karlo prior (https://huggingface.co/kakaobrain/karlo-v1-alpha/tree/main/prior) is selected by default.",
    )
    parser.add_argument(
        "--clip_stats_path",
        type=str,
        help="Path to the clip stats file. Only required if the stable unclip model's config specifies `model.params.noise_aug_config.params.clip_stats_path`.",
        required=False,
    )
    parser.add_argument(
        "--controlnet", action="store_true", default=None, help="Set flag if this is a controlnet checkpoint."
    )
    parser.add_argument("--half", action="store_true", help="Save weights in half precision.")
    parser.add_argument(
        "--vae_path",
        type=str,
        default=None,
        required=False,
        help="Set to a path, hub id to an already converted vae to not convert it again.",
    )
    parser.add_argument(
        "--pipeline_class_name",
        type=str,
        default=None,
        required=False,
        help="Specify the pipeline class name",
    )
    parser.add_argument(
        "--precision",
        type=str,
        help="evaluate at this precision",
        choices=["full", "autocast"],
        default="autocast"
    )
    parser.add_argument(
        "--wandb_log",
        action='store_true',
        help="save grid images to wandb.",
    )
    parser.add_argument(
        "--compress",
        action='store_true',
        help="delta path provided is a compressed checkpoint.",
    )
    parser.add_argument(
        "--modifier_token",
        type=str,
        default=None,
        help="A token to use as a modifier for the concept.",
    )
    args = parser.parse_args()
    main(args)
