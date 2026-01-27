import os
from typing import List
from .types import PCIRunSummary, PCISeedResult
from .utils import ensure_dir, visualize_vqa_binary, visualize_reconstruction, _append_record
from .vqa import QwenYesNo
from dataclasses import asdict

class PCIRunner:
    def __init__(self, backend, vqa, do_cfg: str = True, n_print: int = 10):
        self.backend = backend
        self.vqa = vqa
        self.do_cfg = do_cfg
        self.n_print = n_print

    def run(self,
            target_prompt: str,
            ablated_prompt: str,
            question_template: str,
            concept: str,
            seeds: List[int],
            num_inference_steps: int,
            out_dir: str,
            timestep_stride: int = 1,
            negative_prompt: str = None) -> PCIRunSummary:
        ensure_dir(out_dir)
        self.backend.set_timesteps(num_inference_steps)
        timesteps = self.backend.timesteps()

        summary = PCIRunSummary(
            backend=self.backend.__class__.__name__,
            model_id=self.backend.model_id,
            target_prompt=target_prompt,
            ablated_prompt=ablated_prompt,
            negative_prompt=negative_prompt,
            question_template=question_template,
            concept=concept,
            num_inference_steps=num_inference_steps,
            seeds=seeds,
        )

        # Pre-encode base once per run (saves time when concepts share the base)
        base_ctx = self.backend.encode(target_prompt, negative_prompt=negative_prompt, do_cfg=self.do_cfg)
        concept_dir = os.path.join(out_dir, concept)
        ensure_dir(concept_dir)
        summary_json_path = os.path.join(concept_dir, "results.json")
        
        # Concept context prepared once (used for the second branch)
        concept_ctx = self.backend.encode(ablated_prompt, negative_prompt=None, do_cfg=self.do_cfg)
        # VQA: ask for the concept presence
        looked_seeds = []
        for seed in seeds:
            seed_result = PCISeedResult(
                seed=seed,
                target_prompt=target_prompt,
                ablated_prompt=ablated_prompt,
                negative_prompt=negative_prompt,
                vqa_binary=[],
                timesteps=[]
            )
            saved_reconstructions = {'timesteps': [], 'images': []} 
            for t in timesteps[::timestep_stride]:
                self.backend.set_timesteps(num_inference_steps)
                latents0 = self.backend.prepare_latents(seed)
                # 1) Run down to (exclusive) t
                latents_t, next_index = self.backend.run_until(int(t), latents0.clone(), base_ctx)
                # 2) Switch to concept and reconstruct to end
                image = self.backend.reconstruct_to_end(next_index, latents_t, concept_ctx)
                
                vqa_ans = self.vqa.ask(image, question_template)
                vqa_binary = 1 if vqa_ans["choice"] == "yes" else 0

                seed_result.timesteps.append(int(t))
                seed_result.vqa_binary.append(vqa_binary)
                
                if len(looked_seeds) % self.n_print == 0:
                    saved_reconstructions['timesteps'].append(int(t))
                    saved_reconstructions['images'].append(image)
            
            # Populate plotting cache only once per timestep (e.g., use the first seed)
            if len(looked_seeds) % self.n_print == 0:
                seed_dir = os.path.join(concept_dir, f"seed_{seed}")
                ensure_dir(seed_dir)
                visualize_reconstruction(saved_reconstructions, target_prompt, ablated_prompt, concept, seed_dir)
                visualize_vqa_binary(seed_result, target_prompt, concept, seed_dir)
            looked_seeds.append(seed)
            summary.results.append(seed_result)
            
            # ----- INCREMENTAL SAVE AFTER *EACH* SEED -----
            _append_record(summary_json_path, asdict(seed_result))        
