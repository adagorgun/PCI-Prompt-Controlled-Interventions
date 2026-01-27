
import os

from typing import Dict, Any, Tuple, List
import torch
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
from qwen_vl_utils import process_vision_info

CHOICE_MAP = {"A": "yes", "B": "no"}

class QwenYesNo:
    """Tiny wrapper that forces A/B outputs for yes/no queries on a single image."""
    def __init__(self, model_id: str = "Qwen/Qwen2.5-VL-3B-Instruct", device_map: str = "auto", cache_dir: str = None):
        self.processor = AutoProcessor.from_pretrained(model_id, cache_dir=cache_dir)
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(model_id, torch_dtype="auto", device_map=device_map, cache_dir=cache_dir)

    @torch.no_grad()
    def ask(self, img, question: str) -> Dict[str, Any]:
        mcq_instruction = (
            "Answer the question by replying with exactly one letter.\n"
            "A = YES\nB = NO\n\n"
            "Reply with only A or B."
        )
        messages = [{
            "role": "user",
            "content": [
                {"type": "image", "image": img},
                {"type": "text", "text": f"{mcq_instruction}\n\nQuestion: {question}\nAnswer:"}
            ],
        }]
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self.processor(text=[text], images=image_inputs, videos=video_inputs, padding=True, return_tensors="pt").to(self.model.device)
        gen_out = self.model.generate(**inputs, max_new_tokens=1, do_sample=False, temperature=None, output_scores=False, return_dict_in_generate=True)
        generated_ids = gen_out.sequences[0, inputs.input_ids.shape[1]:]
        raw = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip().upper()
        raw = raw[0] if raw else "B"
        if raw not in ("A", "B"): raw = "B"
        choice = CHOICE_MAP[raw]
        return {"choice": choice, "raw_token": raw}
