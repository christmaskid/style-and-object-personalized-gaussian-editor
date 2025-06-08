import os
from dataclasses import dataclass

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from controlnet_aux import CannyDetector, NormalBaeDetector
from diffusers import (
    ControlNetModel, DDIMScheduler, 
    StableDiffusionControlNetPipeline, 
)
from diffusers.utils.import_utils import is_xformers_available
from tqdm import tqdm

import threestudio
from threestudio.models.prompt_processors.base import PromptProcessorOutput
from threestudio.utils.base import BaseObject
from threestudio.utils.misc import C, parse_version
from threestudio.utils.typing import *

from einops import rearrange, repeat
import sys
sys.path.append("InST")
from ldm.util import instantiate_from_config
from ldm.models.diffusion.ddim import DDIMSampler
from omegaconf import OmegaConf
import PIL

def load_model_from_config(config, ckpt, device, verbose=False):
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

def load_img(path):
    image = PIL.Image.open(path).convert("RGB")
    w, h = image.size
    print(f"loaded input image of size ({w}, {h}) from {path}")
    w, h = map(lambda x: x - x % 32, (w, h))  # resize to integer multiple of 32
    image = image.resize((512, 512), resample=PIL.Image.LANCZOS)
    image = np.array(image).astype(np.float32) / 255.0
    image = image[None].transpose(0, 3, 1, 2)
    image = torch.from_numpy(image)
    return 2.*image - 1.

@threestudio.register("stable-diffusion-inst-guidance")
class InSTGuidance(BaseObject):
    @dataclass
    class Config(BaseObject.Config):
        cache_dir: Optional[str] = None
        pretrained_model_name_or_path: str = "SG161222/Realistic_Vision_V2.0"
        ddim_scheduler_name_or_path: str = "runwayml/stable-diffusion-v1-5"
        control_type: str = "normal"  # normal/canny
        style_image: str = None

        enable_memory_efficient_attention: bool = False
        enable_sequential_cpu_offload: bool = False
        enable_attention_slicing: bool = False
        enable_channels_last_format: bool = False
        guidance_scale: float = 7.5
        condition_scale: float = 1.5
        grad_clip: Optional[
            Any
        ] = None  # field(default_factory=lambda: [0, 2.0, 8.0, 1000])
        half_precision_weights: bool = True

        fixed_size: int = -1

        min_step_percent: float = 0.02
        max_step_percent: float = 0.98

        diffusion_steps: int = 20

        use_sds: bool = False

        # Canny threshold
        canny_lower_bound: int = 50
        canny_upper_bound: int = 100

    cfg: Config

    def make_inpaint_condition(self,image, image_mask):
        image = np.array(image.convert("RGB")).astype(np.float32) / 255.0
        image_mask = np.array(image_mask.convert("L")).astype(np.float32) / 255.0

        assert image.shape[0:1] == image_mask.shape[0:1], "image and image_mask must have the same image size"
        image[image_mask > 0.5] = -1.0  # set as masked pixel
        image = np.expand_dims(image, 0).transpose(0, 3, 1, 2)
        image = torch.from_numpy(image)
        return image

    def configure(self) -> None:
        threestudio.info(f"Loading ControlNet ...")

        controlnet_name_or_path: str
        if self.cfg.control_type == "normal":
            controlnet_name_or_path = "lllyasviel/control_v11p_sd15_normalbae"
        elif self.cfg.control_type == "canny":
            controlnet_name_or_path = "lllyasviel/control_v11p_sd15_canny"
        elif self.cfg.control_type == "p2p":
            controlnet_name_or_path = "lllyasviel/control_v11e_sd15_ip2p"
        elif self.cfg.control_type == "inpaint":
            controlnet_name_or_path = "lllyasviel/control_v11p_sd15_inpaint"

        self.weights_dtype = (
            torch.float16 if self.cfg.half_precision_weights else torch.float32
        )

        pipe_kwargs = {
            "safety_checker": None,
            "feature_extractor": None,
            "requires_safety_checker": False,
            "torch_dtype": self.weights_dtype,
            "cache_dir": self.cfg.cache_dir,
        }

        controlnet = ControlNetModel.from_pretrained(
            controlnet_name_or_path,
            torch_dtype=self.weights_dtype,
            cache_dir=self.cfg.cache_dir,
        )
        # self.pipe = StableDiffusionControlNetPipeline.from_pretrained(
        #     self.cfg.pretrained_model_name_or_path, controlnet=controlnet, **pipe_kwargs
        # ).to(self.device)

        config = "InST/configs/stable-diffusion/v1-inference.yaml"
        ckpt = "custom-diffusion/sd-v1-4.ckpt"
        config = OmegaConf.load(f"{config}")
        self.model = load_model_from_config(config, f"{ckpt}", self.device)
        self.sampler = DDIMSampler(self.model)
        self.style_image = load_img(self.cfg.style_image) if self.cfg.style_image else None
        self.style_image = repeat(self.style_image, "1 C H W -> B C H W", B=1).to(self.device)

        if self.cfg.control_type == "normal":
            self.preprocessor = NormalBaeDetector.from_pretrained(
                "lllyasviel/Annotators"
            )
            self.preprocessor.model.to(self.device)
        elif self.cfg.control_type == "canny":
            self.preprocessor = CannyDetector()

        self.num_train_timesteps = 50
        self.set_min_max_steps()  # set to default value

        self.alphas: Float[Tensor, "..."] = self.model.alphas_cumprod.to(
            self.device
        )

        self.grad_clip_val: Optional[float] = None

        threestudio.info(f"Loaded ControlNet!")

    @torch.cuda.amp.autocast(enabled=False)
    def set_min_max_steps(self, min_step_percent=0.02, max_step_percent=0.98):
        self.min_step = int(self.num_train_timesteps * min_step_percent)
        self.max_step = int(self.num_train_timesteps * max_step_percent)

    def prepare_image_cond(self, cond_rgb: Float[Tensor, "B H W C"]):
        if self.cfg.control_type == "normal":
            cond_rgb = (
                (cond_rgb[0].detach().cpu().numpy() * 255).astype(np.uint8).copy()
            )
            detected_map = self.preprocessor(cond_rgb)
            control = (
                torch.from_numpy(np.array(detected_map)).float().to(self.device) / 255.0
            )
            control = control.unsqueeze(0)
            control = control.permute(0, 3, 1, 2)
        elif self.cfg.control_type == "canny":
            cond_rgb = (
                (cond_rgb[0].detach().cpu().numpy() * 255).astype(np.uint8).copy()
            )
            blurred_img = cv2.blur(cond_rgb, ksize=(5, 5))
            detected_map = self.preprocessor(
                blurred_img, self.cfg.canny_lower_bound, self.cfg.canny_upper_bound
            )
            control = (
                torch.from_numpy(np.array(detected_map)).float().to(self.device) / 255.0
            )
            control = control.unsqueeze(-1).repeat(1, 1, 3)
            control = control.unsqueeze(0)
            control = control.permute(0, 3, 1, 2)
        elif self.cfg.control_type == "p2p":
            control = cond_rgb.permute(0, 3, 1, 2)
        elif self.cfg.control_type == "inpaint":
            control = cond_rgb.permute(0, 3, 1, 2)

        return control

    def __call__(
        self,
        rgb: Float[Tensor, "B H W C"],
        cond_rgb: Float[Tensor, "B H W C"],
        prompt_utils: PromptProcessorOutput,
        **kwargs,
    ):
        batch_size, H, W, _ = rgb.shape
        assert batch_size == 1
        assert rgb.shape[:-1] == cond_rgb.shape[:-1]

        rgb_BCHW = rgb.permute(0, 3, 1, 2)
        latents: Float[Tensor, "B 4 DH DW"]
        if self.cfg.fixed_size > 0:
            RH, RW = self.cfg.fixed_size, self.cfg.fixed_size
        else:
            RH, RW = H // 8 * 8, W // 8 * 8
        rgb_BCHW_HW8 = F.interpolate(
            rgb_BCHW, (RH, RW), mode="bilinear", align_corners=False
        )
        latents = self.model.get_first_stage_encoding(
            self.model.encode_first_stage(rgb_BCHW_HW8)
        )

        # image_cond = self.prepare_image_cond(cond_rgb)
        image_cond = cond_rgb.permute(0, 3, 1, 2)
        image_cond = F.interpolate(
            image_cond, (RH, RW), mode="bilinear", align_corners=False
        )

        temp = torch.zeros(1).to(rgb.device)
        text_embeddings = prompt_utils.get_text_embeddings(temp, temp, temp, False)

        # timestep ~ U(0.02, 0.98) to avoid very high/low noise level
        t = torch.randint(
            self.min_step,
            self.max_step + 1,
            [batch_size],
            dtype=torch.long,
            device=self.device,
        )

        uc = self.model.get_learned_conditioning(batch_size * [""], self.style_image)
        c = self.model.get_learned_conditioning(batch_size * ["*"], self.style_image)
        strength = 0.5
        scale = 10.
        ddim_steps = 50
        ddim_eta = 0.0
        self.sampler.make_schedule(ddim_num_steps=ddim_steps, ddim_eta=ddim_eta, verbose=False)

        t_enc = int(strength * 1000)
        x_noisy = self.model.q_sample(x_start=latents, t=torch.tensor([t_enc]*batch_size).to(self.device))
        model_output = self.model.apply_model(x_noisy, torch.tensor([t_enc]*batch_size).to(self.device), c)
        z_enc = self.sampler.stochastic_encode(latents, torch.tensor([t_enc]*batch_size).to(self.device),
                                            noise = model_output, use_original_steps = True)
        
        t_enc = int(strength * ddim_steps)
        samples = self.sampler.decode(z_enc, c, t_enc, 
                                unconditional_guidance_scale=scale,
                                unconditional_conditioning=uc,)
        x_samples = self.model.decode_first_stage(samples)
        # x_samples = torch.clamp((x_samples + 1.0) / 2.0, min=0.0, max=1.0) * 255.
        edit_images = F.interpolate(x_samples, (H, W), mode="bilinear")

        return {"edit_images": edit_images.permute(0, 2, 3, 1)}

    def update_step(self, epoch: int, global_step: int, on_load_weights: bool = False):
        # clip grad for stable training as demonstrated in
        # Debiasing Scores and Prompts of 2D Diffusion for Robust Text-to-3D Generation
        # http://arxiv.org/abs/2303.15413
        if self.cfg.grad_clip is not None:
            self.grad_clip_val = C(self.cfg.grad_clip, epoch, global_step)

        self.set_min_max_steps(
            min_step_percent=C(self.cfg.min_step_percent, epoch, global_step),
            max_step_percent=C(self.cfg.max_step_percent, epoch, global_step),
        )


if __name__ == "__main__":
    from threestudio.utils.config import ExperimentConfig, load_config
    from threestudio.utils.typing import Optional

    cfg = load_config("configs/debugging/controlnet-normal.yaml")
    guidance = threestudio.find(cfg.system.guidance_type)(cfg.system.guidance)
    prompt_processor = threestudio.find(cfg.system.prompt_processor_type)(
        cfg.system.prompt_processor
    )

    rgb_image = cv2.imread("assets/face.jpg")[:, :, ::-1].copy() / 255
    rgb_image = torch.FloatTensor(rgb_image).unsqueeze(0).to(guidance.device)
    prompt_utils = prompt_processor()
    guidance_out = guidance(rgb_image, rgb_image, prompt_utils)
    edit_image = (
        (guidance_out["edit_images"][0].detach().cpu().clip(0, 1).numpy() * 255)
        .astype(np.uint8)[:, :, ::-1]
        .copy()
    )
    os.makedirs(".threestudio_cache", exist_ok=True)
    cv2.imwrite(".threestudio_cache/edit_image.jpg", edit_image)
