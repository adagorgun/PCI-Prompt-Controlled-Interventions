from typing import Optional, List
import torch
from diffusers import StableDiffusion3Pipeline
from diffusers.pipelines.stable_diffusion_3.pipeline_stable_diffusion_3 import retrieve_timesteps
from .base import ModelBackend, Context
from ..core.utils import nearest_t

class SD35Backend(ModelBackend):
    def load(self, hf_token) -> None:
        self._pipe = StableDiffusion3Pipeline.from_pretrained(
            self.model_id, 
            torch_dtype=torch.bfloat16, 
            cache_dir=self.cache_dir, 
            token=hf_token,
            variant="fp16"
        ).to(self.device)

    def set_timesteps(self, num_inference_steps: int) -> None:
        timesteps, _ = retrieve_timesteps(self._pipe.scheduler, num_inference_steps, device=self._pipe.device, sigmas=None)
        self._pipe._num_timesteps = len(timesteps)
        self._timesteps = timesteps

    @torch.no_grad()
    def encode(self, prompt: str, negative_prompt: Optional[str] = None, do_cfg: bool = True) -> Context:
        pe, ne, pooled, npooled = self._pipe.encode_prompt(
            prompt=prompt,
            prompt_2=None, prompt_3=None,
            negative_prompt=negative_prompt, negative_prompt_2=None, negative_prompt_3=None,
            do_classifier_free_guidance=do_cfg,
            prompt_embeds=None, negative_prompt_embeds=None,
            pooled_prompt_embeds=None, negative_pooled_prompt_embeds=None,
            device=self._pipe.device,
            clip_skip=None,
            num_images_per_prompt=1,
            max_sequence_length=256,
            lora_scale=None,
        )
        if do_cfg:
            cond = torch.cat([ne, pe], dim=0)
            pooled = torch.cat([npooled, pooled], dim=0)
        else:
            cond = pe
        extra = {"pooled": pooled}  # kept for symmetry (unused by UNet forward in SD3.5)
        return Context(cond=cond, extra=extra, is_cfg=do_cfg)

    @torch.no_grad()
    def prepare_latents(self, seed: int, height: int = 1024, width: int = 1024) -> torch.Tensor:
        gen = torch.Generator(device=self._pipe.device).manual_seed(int(seed))
        # SD3.5 UNet takes 16 channels (flow-matching); use pipeline helper
        num_channels_latents = self._pipe.transformer.config.in_channels
        latents = self._pipe.prepare_latents(
            1, num_channels_latents, height, width,
            self._pipe.dtype, self._pipe.device, gen, latents=None,
        )
        return latents
    
    @torch.no_grad()
    def decode_latents(self, latents: torch.Tensor, output_type: str = "pil"):
        latents = (latents / self._pipe.vae.config.scaling_factor) + self._pipe.vae.config.shift_factor
        image = self._pipe.vae.decode(latents, return_dict=False)[0]
        image = self._pipe.image_processor.postprocess(image, output_type=output_type)
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
            timestep_expanded = t.expand(ctx.cond.shape[0])
            noise_pred = self._pipe.transformer(
                    hidden_states=latent_input,
                    timestep=timestep_expanded,
                    encoder_hidden_states=ctx.cond,
                    pooled_projections=ctx.extra["pooled"],
                    joint_attention_kwargs=None,
                    return_dict=False,
                )[0]
            if ctx.is_cfg:
                noise_un, noise_pos = noise_pred.chunk(2)
                noise_pred = noise_un + self._guidance_scale * (noise_pos - noise_un)
            latents = self._pipe.scheduler.step(noise_pred, t, latents, return_dict=False)[0]
        return latents, switch_idx

    @torch.no_grad()
    def reconstruct_to_end(self, start_index: int, latents: torch.Tensor, ctx: Context) -> torch.Tensor:
        steps = self._timesteps[start_index:]
        for t in steps:
            latent_input = latents
            if ctx.is_cfg:
                latent_input = torch.cat([latents] * 2)
            timestep_expanded = t.expand(ctx.cond.shape[0])
            noise_pred = self._pipe.transformer(
                    hidden_states=latent_input,
                    timestep=timestep_expanded,
                    encoder_hidden_states=ctx.cond,
                    pooled_projections=ctx.extra["pooled"],
                    joint_attention_kwargs=None,
                    return_dict=False,
                )[0]
            if ctx.is_cfg:
                noise_un, noise_pos = noise_pred.chunk(2)
                noise_pred = noise_un + self._guidance_scale * (noise_pos - noise_un)
            latents = self._pipe.scheduler.step(noise_pred, t, latents, return_dict=False)[0]
        # SD3.5 uses decoder to image
        image = self.decode_latents(latents, output_type="pil")
        return image

    def timesteps(self) -> List[int]:
        return list(map(int, self._timesteps.detach().cpu().numpy().tolist()))
