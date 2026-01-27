from dataclasses import dataclass, field
from typing import Dict, List, Any


@dataclass
class PCISeedResult:
    seed: int
    vqa_binary: List[int]
    timesteps: List[int]
    negative_prompt: str = None
    target_prompt: str = None
    ablated_prompt: str = None

@dataclass
class PCIRunSummary:
    backend: str
    model_id: str
    target_prompt: str
    ablated_prompt: str
    question_template: str
    concept: str
    num_inference_steps: int
    seeds: List[int]
    results: List[PCISeedResult] = field(default_factory=list)
    negative_prompt: str = None
