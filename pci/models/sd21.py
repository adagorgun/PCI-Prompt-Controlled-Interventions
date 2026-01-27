from typing import Optional, List
import torch
from diffusers import StableDiffusionPipeline, DDIMScheduler
from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion import retrieve_timesteps
from .base import ModelBackend, Context
from ..core.utils import nearest_t
import os

class SD21Backend(ModelBackend):
    def load(self) -> None:
        self._pipe = StableDiffusionPipeline.from_pretrained(
            self.model_id,
            torch_dtype=torch.float16,
            cache_dir=self.cache_dir,
        ).to(self.device)
        self._pipe.scheduler = DDIMScheduler.from_pretrained(self.model_id, 
                                                             subfolder="scheduler", 
                                                             cache_dir=self.cache_dir)
        self._pipe.enable_attention_slicing()

    def set_timesteps(self, num_inference_steps: int) -> None:
        timesteps, _ = retrieve_timesteps(self._pipe.scheduler, num_inference_steps, device=self._pipe.device, sigmas=None)
        self._pipe._num_timesteps = len(timesteps)
        self._timesteps = timesteps

    def encode(self, prompt: str, negative_prompt: Optional[str] = None, do_cfg: bool = True) -> Context:
        pe, ne = self._pipe.encode_prompt(
            prompt=prompt,
            negative_prompt=negative_prompt,
            num_images_per_prompt=1,
            do_classifier_free_guidance=do_cfg,
            prompt_embeds=None,
            negative_prompt_embeds=None,
            lora_scale=None,
            device=self._pipe.device
        )
        if do_cfg:
            cond = torch.cat([ne, pe], dim=0)
        else:
            cond = pe
        return Context(cond=cond, extra=None, is_cfg=do_cfg)

    def prepare_latents(self, seed: int, height: int = 512, width: int = 512) -> torch.Tensor:
        gen = torch.Generator(device=self._pipe.device).manual_seed(int(seed))
        # SD3.5 UNet takes 16 channels (flow-matching); use pipeline helper
        num_channels_latents = self._pipe.unet.config.in_channels
        latents = self._pipe.prepare_latents(
            1, num_channels_latents,
            height, width,
            self._pipe.dtype,
            self._pipe.device,
            gen,
            latents=None,
        )
        return latents
    
    @torch.no_grad()
    def decode_latents(self, latents: torch.Tensor, output_type: str = "pil"):
        image = self._pipe.vae.decode(latents / self._pipe.vae.config.scaling_factor, return_dict=False)[0]
        image, has_nsfw_concept = self._pipe.run_safety_checker(image, self._pipe.device, latents.dtype)
        if has_nsfw_concept is None:
            do_denormalize = [True] * image.shape[0]
        else:
            do_denormalize = [not has_nsfw for has_nsfw in has_nsfw_concept]
        image = self._pipe.image_processor.postprocess(image, output_type=output_type, do_denormalize=do_denormalize)
        return image[0] if isinstance(image, list) else image

    @torch.no_grad()
    def run_until(self, t_target: int, latents: torch.Tensor, ctx: Context):
        switch_t, switch_idx = nearest_t(t_target, self._timesteps)
        for i, t in enumerate(self._timesteps):
            if t <= switch_t:
                return latents, i
            latent_input = latents
            if ctx.is_cfg:
                latent_input = torch.cat([latents] * 2)
            latent_input = self._pipe.scheduler.scale_model_input(latent_input, t)
            noise_pred = self._pipe.unet(
                    latent_input,
                    t,
                    encoder_hidden_states=ctx.cond,
                ).sample
            if ctx.is_cfg:
                noise_un, noise_pos = noise_pred.chunk(2)
                noise_pred = noise_un + self._guidance_scale * (noise_pos - noise_un)
            latents = self._pipe.scheduler.step(noise_pred, t, latents).prev_sample
        return latents, switch_idx

    @torch.no_grad()
    def reconstruct_to_end(self, start_index: int, latents: torch.Tensor, ctx: Context) -> torch.Tensor:
        steps = self._timesteps[start_index:]
        for t in steps:
            latent_input = latents
            if ctx.is_cfg:
                latent_input = torch.cat([latents] * 2)
            latent_input = self._pipe.scheduler.scale_model_input(latent_input, t)
            noise_pred = self._pipe.unet(
                    latent_input,
                    t,
                    encoder_hidden_states=ctx.cond,
                ).sample
            if ctx.is_cfg:
                noise_un, noise_pos = noise_pred.chunk(2)
                noise_pred = noise_un + self._guidance_scale * (noise_pos - noise_un)
            latents = self._pipe.scheduler.step(noise_pred, t, latents).prev_sample
        # SD3.5 uses decoder to image
        image = self.decode_latents(latents, output_type="pil")
        return image

    def timesteps(self) -> List[int]:
        return list(map(int, self._timesteps.detach().cpu().numpy().tolist()))
