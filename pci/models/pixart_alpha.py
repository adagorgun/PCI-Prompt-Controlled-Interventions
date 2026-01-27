from typing import Optional, List
import torch
from diffusers import PixArtAlphaPipeline

from diffusers import StableDiffusionXLPipeline, EulerAncestralDiscreteScheduler
from diffusers.pipelines.pixart_alpha.pipeline_pixart_alpha import retrieve_timesteps
from .base import ModelBackend, Context
from ..core.utils import nearest_t

class PixArtAlphaBackend(ModelBackend):
    def load(self) -> None:
        self._pipe = PixArtAlphaPipeline.from_pretrained(self.model_id, 
                                                   torch_dtype=torch.float16, 
                                                   use_safetensors=True, 
                                                   cache_dir=self.cache_dir).to(self.device)

        self._scheduler = self._pipe.scheduler

    def set_timesteps(self, num_inference_steps: int) -> None:
        timesteps, _ = retrieve_timesteps(self._pipe.scheduler, num_inference_steps, device=self._pipe.device, sigmas=None)
        self._pipe._num_timesteps = len(timesteps)
        self._timesteps = timesteps

    def encode(self, prompt: str, negative_prompt: Optional[str] = None, do_cfg: bool = True) -> Context:
        pe, pam, ne, nam = self._pipe.encode_prompt(
            prompt,
            do_cfg,
            negative_prompt=negative_prompt,
            num_images_per_prompt=1,
            device=self._pipe.device,
            prompt_embeds=None,
            negative_prompt_embeds=None,
            prompt_attention_mask=None,
            negative_prompt_attention_mask=None,
            clean_caption=True,
            max_sequence_length=120,
        )
        
        if do_cfg:
            cond = torch.cat([ne, pe], dim=0)
            cond_mask = torch.cat([nam, pam], dim=0)
        else:
            cond = pe
            cond_mask = pam
            
        cond = cond.to(self._pipe.device)
        cond_mask = cond_mask.to(self._pipe.device)
        
        extra = {"resolution": None, "aspect_ratio": None}
        if self._pipe.transformer.config.sample_size == 128:
            height = 1024 ; width = 1024
            resolution = torch.tensor([height, width]).repeat(1, 1)
            aspect_ratio = torch.tensor([float(height / width)]).repeat(1, 1)
            resolution = resolution.to(dtype=pe.dtype, device=self._pipe.device)
            aspect_ratio = aspect_ratio.to(dtype=pe.dtype, device=self._pipe.device)

            if do_cfg:
                resolution = torch.cat([resolution, resolution], dim=0)
                aspect_ratio = torch.cat([aspect_ratio, aspect_ratio], dim=0)

            extra = {"resolution": resolution, "aspect_ratio": aspect_ratio}

        return Context(cond=cond, cond_mask=cond_mask, extra=extra, is_cfg=do_cfg)

    def prepare_latents(self, seed: int, height: int = 1024, width: int = 1024) -> torch.Tensor:
        gen = torch.Generator(device=self._pipe.device).manual_seed(int(seed))
        num_channels_latents = self._pipe.transformer.config.in_channels
        latents = self._pipe.prepare_latents(
            1, num_channels_latents, height, width,
            self._pipe.dtype, self._pipe.device, gen, latents=None,
        )
        eta = 0.0
        self.extra_step_kwargs = self._pipe.prepare_extra_step_kwargs(gen, eta)
        return latents
    
    def decode_latents(self, latents: torch.Tensor, output_type: str = "pil"):               
        latents = latents / self._pipe.vae.config.scaling_factor
        image = self._pipe.vae.decode(latents, return_dict=False)[0]
        
        orig_width = 1024 ; orig_height = 1024
        image = self._pipe.image_processor.resize_and_crop_tensor(image, orig_width, orig_height)
        image = self._pipe.image_processor.postprocess(image, output_type=output_type)
        return image[0] if isinstance(image, list) else image

    @torch.no_grad()
    def run_until(self, t_target: int, latents: torch.Tensor, ctx: Context):
        switch_t, switch_idx = nearest_t(t_target, self._timesteps)
        for i, t in enumerate(self._scheduler.timesteps):
            if t <= switch_t:
                return latents, i  # break just before target
            latent_input = latents
            if ctx.is_cfg:
                latent_input = torch.cat([latents] * 2)
            latent_input = self._scheduler.scale_model_input(latent_input, t)
            
            current_timestep = t
            current_timestep = current_timestep[None].to(latent_input.device)
            current_timestep = current_timestep.expand(latent_input.shape[0])
            
            noise_pred = self._pipe.transformer(
                    latent_input,
                    encoder_hidden_states=ctx.cond,
                    encoder_attention_mask=ctx.cond_mask,
                    timestep=current_timestep,
                    added_cond_kwargs=ctx.extra,
                    return_dict=False,
                )[0]
            
            if ctx.is_cfg:
                noise_un, noise_pos = noise_pred.chunk(2)
                noise_pred = noise_un + self._guidance_scale * (noise_pos - noise_un)
                
            if self._pipe.transformer.config.out_channels // 2 == self._pipe.transformer.config.in_channels:
                noise_pred = noise_pred.chunk(2, dim=1)[0]
            else:
                noise_pred = noise_pred
                
            latents = self._scheduler.step(noise_pred, t, latents, **self.extra_step_kwargs, return_dict=False)[0]
        return latents, switch_idx

    @torch.no_grad()
    def reconstruct_to_end(self, start_index: int, latents: torch.Tensor, ctx: Context) -> torch.Tensor:
        steps = self._scheduler.timesteps[start_index:]
        for t in steps:
            latent_input = latents
            if ctx.is_cfg:
                latent_input = torch.cat([latents] * 2)
            latent_input = self._scheduler.scale_model_input(latent_input, t)
            
            current_timestep = t
            current_timestep = current_timestep[None].to(latent_input.device)
            current_timestep = current_timestep.expand(latent_input.shape[0])
            
            noise_pred = self._pipe.transformer(
                    latent_input,
                    encoder_hidden_states=ctx.cond,
                    encoder_attention_mask=ctx.cond_mask,
                    timestep=current_timestep,
                    added_cond_kwargs=ctx.extra,
                    return_dict=False,
                )[0]
            
            if ctx.is_cfg:
                noise_un, noise_pos = noise_pred.chunk(2)
                noise_pred = noise_un + self._guidance_scale * (noise_pos - noise_un)
                
            if self._pipe.transformer.config.out_channels // 2 == self._pipe.transformer.config.in_channels:
                noise_pred = noise_pred.chunk(2, dim=1)[0]
            else:
                noise_pred = noise_pred
                
            latents = self._scheduler.step(noise_pred, t, latents, **self.extra_step_kwargs, return_dict=False)[0]
        image = self.decode_latents(latents, output_type="pil")
        return image

    def timesteps(self) -> List[int]:
        return list(map(int, self._scheduler.timesteps.detach().cpu().numpy().tolist()))