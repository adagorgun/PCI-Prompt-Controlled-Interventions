from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, Tuple, Optional, List
import torch

@dataclass
class Context:
    """Container for per-backend conditioning (text embeddings etc.)."""
    # Main conditioning for UNet forward (e.g., encoder_hidden_states)
    cond: torch.Tensor
    # Optional pooled/additional condition (e.g., SDXL pooled text)
    extra: Dict[str, torch.Tensor]
    # Whether batch is CFG-stacked [neg,pos]
    is_cfg: bool
    cond_mask: torch.Tensor = None

class ModelBackend(ABC):
    def __init__(self, model_id: str, device: str = "cuda", dtype: str = "auto", cache_dir: Optional[str] = None):
        self.model_id = model_id
        self.device = device
        self.dtype = dtype
        self.cache_dir = cache_dir
        self._guidance_scale = 7.5
        self._pipe = None
        self._scheduler = None
        self._timesteps = None

    # ---- lifecycle ----
    @abstractmethod
    def load(self) -> None:
        ...

    @abstractmethod
    def set_timesteps(self, num_inference_steps: int) -> None:
        ...

    # ---- text enc ----
    @abstractmethod
    def encode(self, prompt: str, negative_prompt: Optional[str] = None, do_cfg: bool = True) -> Context:
        ...

    # ---- latents ----
    @abstractmethod
    def prepare_latents(self, seed: int, height: int = 1024, width: int = 1024) -> torch.Tensor:
        ...

    # ---- diffusion stepping ----
    @abstractmethod
    def run_until(self, t_target: int, latents: torch.Tensor, ctx: Context) -> Tuple[torch.Tensor, int]:
        """Run from t_max down to (exclusive) t_target. Returns (latents, next_index)."""
        ...

    @abstractmethod
    def reconstruct_to_end(self, start_index: int, latents: torch.Tensor, ctx: Context) -> torch.Tensor:
        """Run remaining steps to t_min and decode to (1,3,H,W) in [-1,1]."""
        ...

    # ---- helpers ----
    @abstractmethod
    def timesteps(self) -> List[int]:
        ...

    def set_guidance(self, scale: float):
        self._guidance_scale = float(scale)
