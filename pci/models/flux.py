from typing import Optional, List
import torch, numpy as np
from diffusers import FluxPipeline
from diffusers.pipelines.flux.pipeline_flux import calculate_shift, retrieve_timesteps
from .base import ModelBackend, Context
from ..core.utils import nearest_t

class FluxBackend(ModelBackend):
    def load(self, hf_token) -> None:
        self._pipe = FluxPipeline.from_pretrained(
            self.model_id, 
            torch_dtype=torch.bfloat16, 
            cache_dir=self.cache_dir,
            token=hf_token,
        ).to(self.device)
        self._scheduler = self._pipe.scheduler
        latents = self.prepare_latents(0)
        self.image_seq_len = latents.shape[1]

    def set_timesteps(self, num_inference_steps: int) -> None:
        sigmas = np.linspace(1.0, 1 / num_inference_steps, num_inference_steps)
        mu = calculate_shift(
            self.image_seq_len,
            self._pipe.scheduler.config.base_image_seq_len,
            self._pipe.scheduler.config.max_image_seq_len,
            self._pipe.scheduler.config.base_shift,
            self._pipe.scheduler.config.max_shift,
        )
        timesteps, num_inference_steps = retrieve_timesteps(
            self._pipe.scheduler,
            num_inference_steps,
            self._pipe.device,
            sigmas=sigmas,
            mu=mu,
        )
        self._pipe._num_timesteps = len(timesteps)
        self._timesteps = timesteps

    def encode(self, prompt: str, negative_prompt: Optional[str] = None, do_cfg: bool = False) -> Context:
        pe, pooled, text_ids = self._pipe.encode_prompt(
            prompt=prompt,
            prompt_2=None,
            prompt_embeds=None,
            pooled_prompt_embeds=None,
            device=self._pipe.device,
            num_images_per_prompt=1,
            max_sequence_length=512,
            lora_scale=None,
        )
        if do_cfg and negative_prompt:
            ne, npooled, _ = self._pipe.encode_prompt(
                prompt=negative_prompt,
                prompt_2=None,
                prompt_embeds=None,
                pooled_prompt_embeds=None,
                device=self._pipe.device,
                num_images_per_prompt=1,
                max_sequence_length=512,
                lora_scale=None,
            )
            cond = pe
            is_cfg = True
            extra_neg = {"pooled": npooled}
            self.negative_info = Context(cond=ne, extra=extra_neg, is_cfg=is_cfg)
        else:
            cond = pe
            is_cfg = False
        extra = {"text_ids": text_ids, "pooled": pooled}
        return Context(cond=cond, extra=extra, is_cfg=is_cfg)

    def prepare_latents(self, seed: int, height: int = 1024, width: int = 1024) -> torch.Tensor:
        gen = torch.Generator(device=self._pipe.device).manual_seed(int(seed))
        num_channels_latents = self._pipe.transformer.config.in_channels // 4
        latents, latent_image_ids = self._pipe.prepare_latents(
            1, num_channels_latents, height, width,
            self._pipe.dtype, self._pipe.device, gen, latents=None,
        )
        self.latent_image_ids = latent_image_ids
        return latents
    
    @torch.no_grad()
    def decode_latents(self, latents: torch.Tensor, output_type: str = "pil"):
        latents = self._pipe._unpack_latents(latents, 1024, 1024, self._pipe.vae_scale_factor)
        latents = (latents / self._pipe.vae.config.scaling_factor) + self._pipe.vae.config.shift_factor
        image = self._pipe.vae.decode(latents, return_dict=False)[0]
        image = self._pipe.image_processor.postprocess(image, output_type=output_type)
        return image[0] if isinstance(image, list) else image

    @torch.no_grad()
    def run_until(self, t_target: int, latents: torch.Tensor, ctx: Context):
        switch_t, switch_idx = nearest_t(t_target, self._timesteps)
        # handle guidance
        if self._pipe.transformer.config.guidance_embeds:
            self.guidance = torch.full([1], self._guidance_scale, device=self._pipe.device, dtype=torch.float32)
            self.guidance = self.guidance.expand(latents.shape[0])
        else:
            self.guidance = None

        for i, t in enumerate(self._timesteps):
            if t <= switch_t:
                return latents, i
            timestep = t.expand(latents.shape[0]).to(latents.dtype)
            latent_input = latents if not ctx.is_cfg else torch.cat([latents] * 2)
            noise_pred = self._pipe.transformer(
                    hidden_states=latent_input,
                    timestep=timestep / 1000,
                    guidance=self.guidance,
                    pooled_projections=ctx.extra["pooled"],
                    encoder_hidden_states=ctx.cond,
                    txt_ids=ctx.extra["text_ids"],
                    img_ids=self.latent_image_ids,
                    joint_attention_kwargs={},
                    return_dict=False,
                )[0]
            if ctx.is_cfg:
                neg_noise_pred = self._pipe.transformer(
                    hidden_states=latent_input,
                    timestep=timestep / 1000,
                    guidance=self.guidance,
                    pooled_projections=self.negative_info.extra["pooled"],
                    encoder_hidden_states=self.negative_info.cond,
                    txt_ids=ctx.extra["text_ids"],
                    img_ids=self.latent_image_ids,
                    joint_attention_kwargs={},
                    return_dict=False,
                )[0]
                noise_pred = neg_noise_pred + self._guidance_scale * (noise_pred - neg_noise_pred)

            latents = self._pipe.scheduler.step(noise_pred, t, latents, return_dict=False)[0]
        return latents, switch_idx

    @torch.no_grad()
    def reconstruct_to_end(self, start_index: int, latents: torch.Tensor, ctx: Context) -> torch.Tensor:
        if self._pipe.transformer.config.guidance_embeds:
            self.guidance = torch.full([1], self._guidance_scale, device=self._pipe.device, dtype=torch.float32)
            self.guidance = self.guidance.expand(latents.shape[0])
        else:
            self.guidance = None
            
        for t in self._timesteps[start_index:]:
            timestep = t.expand(latents.shape[0]).to(latents.dtype)
            latent_input = latents if not ctx.is_cfg else torch.cat([latents] * 2)
            noise_pred = self._pipe.transformer(
                    hidden_states=latent_input,
                    timestep=timestep / 1000,
                    guidance=self.guidance,
                    pooled_projections=ctx.extra["pooled"],
                    encoder_hidden_states=ctx.cond,
                    txt_ids=ctx.extra["text_ids"],
                    img_ids=self.latent_image_ids,
                    joint_attention_kwargs={},
                    return_dict=False,
                )[0]
            if ctx.is_cfg:
                neg_noise_pred = self._pipe.transformer(
                    hidden_states=latent_input,
                    timestep=timestep / 1000,
                    guidance=self.guidance,
                    pooled_projections=self.negative_info.extra["pooled"],
                    encoder_hidden_states=self.negative_info.cond,
                    txt_ids=ctx.extra["text_ids"],
                    img_ids=self.latent_image_ids,
                    joint_attention_kwargs={},
                    return_dict=False,
                )[0]
                noise_pred = neg_noise_pred + self._guidance_scale * (noise_pred - neg_noise_pred)

            latents = self._pipe.scheduler.step(noise_pred, t, latents, return_dict=False)[0]
        image = self.decode_latents(latents, output_type="pil")
        return image

    def timesteps(self) -> List[int]:
        return [int(t) for t in self._timesteps]
