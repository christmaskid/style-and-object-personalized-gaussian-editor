"""make variations of input image"""

import argparse, os, sys, glob
import PIL
import torch
import torch.nn as nn
import numpy as np
from omegaconf import OmegaConf
from PIL import Image
from tqdm import tqdm, trange
from itertools import islice
from einops import rearrange, repeat
from torchvision.utils import make_grid
from torch import autocast
from contextlib import nullcontext
import time
from pytorch_lightning import seed_everything

sys.path.append(os.path.dirname(sys.path[0]))
from ldm.util import instantiate_from_config
from ldm.models.diffusion.ddim import DDIMSampler
from ldm.models.diffusion.plms import PLMSSampler

from transformers import CLIPProcessor, CLIPModel

import argparse


def load_model_from_config(config, ckpt, verbose=False):
    print(f"Loading model from {ckpt}")
    pl_sd = torch.load(ckpt, map_location="cpu")
    if "global_step" in pl_sd:
        print(f"Global Step: {pl_sd['global_step']}")
    sd = pl_sd["state_dict"]
    model = instantiate_from_config(config.model)
    m, u = model.load_state_dict(sd, strict=False)
    if len(m) > 0 and verbose:
        print("missing keys:")
        print(m)
    if len(u) > 0 and verbose:
        print("unexpected keys:")
        print(u)

    model.to(device)
    model.eval()
    return model


if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42, help="random seed")
    parser.add_argument("--config", type=str, default="configs/stable-diffusion/v1-inference.yaml", help="path to config file")
    parser.add_argument("--ckpt", type=str, default="../style-and-object-personalized-gaussian-editor/custom-diffusion/sd-v1-4.ckpt", help="path to checkpoint file")
    args = parser.parse_args()

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    config = args.config
    ckpt = args.ckpt
    config = OmegaConf.load(f"{config}")
    model = load_model_from_config(config, f"{ckpt}")
    sampler = DDIMSampler(model)

    # model.cpu()
    log_dir = 'van-gogh2025-06-07T16-10-18_van-gogh'
    model.embedding_manager.load(f'./logs/{log_dir}/checkpoints/embeddings.pt')
    model = model.to(device)

    torch.save(model.state_dict(), f'./logs/{log_dir}/checkpoints/model.pt')
    print(f"Model saved to ./logs/{log_dir}/checkpoints/model.pt")