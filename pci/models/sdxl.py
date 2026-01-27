from typing import Optional, List
import torch
from diffusers import StableDiffusionXLPipeline, EulerAncestralDiscreteScheduler
from diffusers.pipelines.stable_diffusion_xl.pipeline_stable_diffusion_xl import retrieve_timesteps
from .base import ModelBackend, Context
from ..core.utils import nearest_t

class SDXLBackend(ModelBackend):
    def load(self) -> None:
        self._pipe = StableDiffusionXLPipeline.from_pretrained(
            self.model_id, torch_dtype=torch.float16, cache_dir=self.cache_dir
        ).to(self.device)
        # Karras sigmas (good spacing)
        self._pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(
            self._pipe.scheduler.config, use_karras_sigmas=True
        )
        # Make behavior deterministic-ish
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        self._scheduler = self._pipe.scheduler

    def set_timesteps(self, num_inference_steps: int) -> None:
        timesteps, _ = retrieve_timesteps(self._pipe.scheduler, num_inference_steps, device=self._pipe.device, sigmas=None)
        self._pipe._num_timesteps = len(timesteps)
        self._timesteps = timesteps

    def encode(self, prompt: str, negative_prompt: Optional[str] = None, do_cfg: bool = True) -> Context:
        pe, ne, pooled, npooled = self._pipe.encode_prompt(
            prompt=prompt,
            prompt_2=None,
            device=self._pipe.device,
            num_images_per_prompt=1,
            do_classifier_free_guidance=do_cfg,
            negative_prompt=negative_prompt,
            negative_prompt_2=None,
            prompt_embeds=None,
            negative_prompt_embeds=None,
            pooled_prompt_embeds=None,
            negative_pooled_prompt_embeds=None,
            lora_scale=None,
            clip_skip=None,
        )
        add_text_embeds = pooled
        # time ids fixed to 1024 x 1024; adapt if you change resolution
        if self._pipe.text_encoder_2 is None:
            text_encoder_projection_dim = int(pooled.shape[-1])
        else:
            text_encoder_projection_dim = self._pipe.text_encoder_2.config.projection_dim if self._pipe.text_encoder_2 else int(pooled.shape[-1])
        
        add_time_ids = self._pipe._get_add_time_ids(
            (1024, 1024), (0, 0), (1024, 1024),
            dtype=pe.dtype, text_encoder_projection_dim=text_encoder_projection_dim,
        )
                
        if do_cfg:
            cond = torch.cat([ne, pe], dim=0)
            add_text_embeds = torch.cat([npooled, add_text_embeds], dim=0)
            add_time_ids = torch.cat([add_time_ids, add_time_ids], dim=0)
        else:
            cond = pe
            
        cond = cond.to(self._pipe.device)
        add_text_embeds = add_text_embeds.to(self._pipe.device)
        add_time_ids = add_time_ids.to(self._pipe.device).repeat(1, 1)

        extra = {"text_embeds": add_text_embeds, "time_ids": add_time_ids}
        return Context(cond=cond, extra=extra, is_cfg=do_cfg)

    def prepare_latents(self, seed: int, height: int = 1024, width: int = 1024) -> torch.Tensor:
        gen = torch.Generator(device=self._pipe.device).manual_seed(int(seed))
        num_channels_latents = self._pipe.unet.config.in_channels # SDXL VAE scaling -> 4 channels; unet expects 4
        latents = self._pipe.prepare_latents(
            1, num_channels_latents, height, width,
            self._pipe.dtype, self._pipe.device, gen, latents=None,
        )
        eta = 0.0
        self.extra_step_kwargs = self._pipe.prepare_extra_step_kwargs(gen, eta)
        return latents
    
    def decode_latents(self, latents: torch.Tensor, output_type: str = "pil"):
        needs_upcasting = self._pipe.vae.dtype == torch.float16 and self._pipe.vae.config.force_upcast
        if needs_upcasting:
            self._pipe.upcast_vae()
            latents = latents.to(next(iter(self._pipe.vae.post_quant_conv.parameters())).dtype)
        elif latents.dtype != self._pipe.vae.dtype:
            if torch.backends.mps.is_available():
                # some platforms (eg. apple mps) misbehave due to a pytorch bug: https://github.com/pytorch/pytorch/pull/99272
                self._pipe.vae = self._pipe.vae.to(latents.dtype)
                
        latents = latents / self._pipe.vae.config.scaling_factor
        image = self._pipe.vae.decode(latents, return_dict=False)[0]
        if needs_upcasting:
            self._pipe.vae.to(dtype=torch.float16)

        if self._pipe.watermark is not None:
            image = self._pipe.watermark.apply_watermark(image)

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
            noise_pred = self._pipe.unet(
                    latent_input,
                    t,
                    encoder_hidden_states=ctx.cond,
                    timestep_cond=None,
                    cross_attention_kwargs=None,
                    added_cond_kwargs=ctx.extra,
                    return_dict=False,
                )[0]
            
            if ctx.is_cfg:
                noise_un, noise_pos = noise_pred.chunk(2)
                noise_pred = noise_un + self._guidance_scale * (noise_pos - noise_un)
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
            
            noise_pred = self._pipe.unet(
                    latent_input,
                    t,
                    encoder_hidden_states=ctx.cond,
                    timestep_cond=None,
                    cross_attention_kwargs=None,
                    added_cond_kwargs=ctx.extra,
                    return_dict=False,
                )[0]
            if ctx.is_cfg:
                noise_un, noise_pos = noise_pred.chunk(2)
                noise_pred = noise_un + self._guidance_scale * (noise_pos - noise_un)
            latents = self._scheduler.step(noise_pred, t, latents, **self.extra_step_kwargs, return_dict=False)[0]
        image = self.decode_latents(latents, output_type="pil")
        return image

    def timesteps(self) -> List[int]:
        return list(map(int, self._scheduler.timesteps.detach().cpu().numpy().tolist()))