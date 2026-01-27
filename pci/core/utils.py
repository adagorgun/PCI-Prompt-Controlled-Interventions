import os, math, json
from typing import List, Tuple, Dict, Any
import numpy as np
import torch
import matplotlib.pyplot as plt
from collections import defaultdict

# ----------------- Resumption + incremental save helpers -----------------

def _safe_read_results(path):
    """Read existing results.json (list). If missing/corrupt -> []."""
    try:
        with open(path, "r") as f:
            data = json.load(f)
        return data if isinstance(data, list) else [data]
    except FileNotFoundError:
        return []
    except JSONDecodeError:
        ts = time.strftime("%Y%m%d_%H%M%S")
        try:
            os.replace(path, f"{path}.corrupt_{ts}.bak")
            print(f"[warn] {path} was corrupt. Backed up to {path}.corrupt_{ts}.bak")
        except Exception:
            print(f"[warn] {path} was corrupt and could not be backed up.")
        return []

def _atomic_write_json(path, data):
    tmp = f"{path}.tmp"
    with open(tmp, "w") as f:
        json.dump(data, f, indent=2)
    os.replace(tmp, path)

def _is_negative_record(rec: dict) -> bool:
    """Your rule: negative iff negative_prompt is not None (nor empty)."""
    return rec.get("negative_prompt") not in (None, "")

def _build_processed_sets(records):
    """
    Returns (processed_normal, processed_neg, all_processed) as sets of seed ints,
    using negative_prompt presence to decide the bucket.
    """
    normal, neg = set(), set()
    for r in records:
        if not isinstance(r, dict): 
            continue
        s = r.get("seed")
        if isinstance(s, int):
            (neg if _is_negative_record(r) else normal).add(s)
    return normal, neg, normal | neg

def _append_record(summary_json_path: str, rec: dict):
    """
    Append one record with de-dup by (seed, is_negative_by_negative_prompt).
    Schema is preserved (no extra fields).
    """
    existing = _safe_read_results(summary_json_path)

    def key(x):
        if not isinstance(x, dict):
            return None
        s = x.get("seed")
        return (s, _is_negative_record(x)) if isinstance(s, int) else None

    seen = {key(r) for r in existing}
    k = key(rec)
    if k is None or k in seen:
        return  # already present or invalid

    existing.append(rec)
    _atomic_write_json(summary_json_path, existing)

def load_config(config_path):
    with open(config_path, "r") as f:
        return json.load(f)
    
def get_category_data(cfg, category_path):
    """
    category_path: e.g. 'demographics.gender'
    """
    keys = category_path.split(".")
    data = cfg
    for k in keys:
        if k not in data:
            raise KeyError(f"Category {k} not found in config.")
        data = data[k]
    return data

def nearest_t(val, timesteps):
    ts_tensor = timesteps.detach().to("cpu")
    idx = (ts_tensor - val).abs().argmin().item()
    return timesteps[idx].item(), idx

def needs_more(valid_seeds_per_concept, num_seed):
    return (not valid_seeds_per_concept) or any(len(valid_seeds_per_concept[c]) < num_seed for c in valid_seeds_per_concept)

def concept_folder_path(folder, concept_name: str) -> str:
    p = os.path.join(folder, concept_name)
    ensure_dir(p)
    return p

def append_negative_seed_atomic(out_dir, concept, seed):
    concept_dir = os.path.join(out_dir, concept)
    ensure_dir(concept_dir)
    filepath = os.path.join(concept_dir, "negative_seeds.txt")
    with open(filepath, "a") as f:
        f.write(f"{seed}\n")

def seeds_txt_path(folder: str, concept_name: str) -> str:
    return os.path.join(concept_folder_path(folder, concept_name), "seeds.txt")

def load_existing_seeds(out_dir, concepts):
    seeds_per_concept = {c: set() for c in concepts}
    for concept in concepts:
        concept_dir = os.path.join(out_dir, concept)
        ensure_dir(concept_dir)
        # valid seeds
        seed_file = os.path.join(concept_dir, "seeds.txt")
        if os.path.exists(seed_file):
            with open(seed_file) as f:
                seeds_per_concept[concept].update(int(line.strip()) for line in f if line.strip())
        # negative seeds
        neg_file = os.path.join(concept_dir, "negative_seeds.txt")
        if os.path.exists(neg_file):
            with open(neg_file) as f:
                seeds_per_concept[concept].update(int(line.strip()) for line in f if line.strip())
    return seeds_per_concept

def load_seeds(out_dir, concepts):
    seeds_per_concept = {c: set() for c in concepts}
    neg_seeds_per_concept = {c: set() for c in concepts}
    for concept in concepts:
        concept_dir = os.path.join(out_dir, concept)
        ensure_dir(concept_dir)
        # valid seeds
        seed_file = os.path.join(concept_dir, "seeds.txt")
        if os.path.exists(seed_file):
            with open(seed_file) as f:
                seeds_per_concept[concept].update(int(line.strip()) for line in f if line.strip())
        # negative seeds
        neg_file = os.path.join(concept_dir, "negative_seeds.txt")
        if os.path.exists(neg_file):
            with open(neg_file) as f:
                neg_seeds_per_concept[concept].update(int(line.strip()) for line in f if line.strip())
    return seeds_per_concept, neg_seeds_per_concept

def load_existing_seeds_v1(folder, concepts):
    loaded = defaultdict(set)
    for c in concepts:
        if c == "full":
            continue
        path = seeds_txt_path(folder, c)
        if os.path.exists(path):
            with open(path, "r") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        try:
                            loaded[c].add(int(line))
                        except ValueError:
                            pass
        else:
            loaded[c] = set()
    return loaded

def append_seed_atomic(folder: str, concept_name: str, seed: int):
    """
    Append a seed to the concept's seeds.txt immediately and fsync so progress isn't lost.
    """
    path = seeds_txt_path(folder, concept_name)
    # Ensure directory exists
    ensure_dir(os.path.dirname(path))
    with open(path, "a") as f:
        f.write(f"{seed}\n")
        f.flush()
        os.fsync(f.fileno())

def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)

def to_uint8_image(image: torch.Tensor):
    """Expects (1,3,H,W) in [-1,1] -> returns (H,W,3) uint8."""
    image = (image / 2 + 0.5).clamp(0, 1)
    image = image.detach().cpu().permute(0, 2, 3, 1).numpy()[0]
    return (image * 255).round().astype("uint8")

def tau_from_sigmas(sigmas: np.ndarray) -> np.ndarray:
    sig = np.asarray(sigmas, dtype=np.float64)
    if sig.ndim != 1:
        sig = sig.reshape(-1)
    # Avoid length mismatch corner cases in some schedulers
    # tau = normalized (-log sigma^2)
    eps = 1e-20
    logsnr = -np.log(np.maximum(sig**2, eps))
    # enforce ascending
    if logsnr[0] > logsnr[-1]:
        logsnr = logsnr[::-1]
    lmin, lmax = float(logsnr.min()), float(logsnr.max())
    denom = max(lmax - lmin, 1e-12)
    return (logsnr - lmin) / denom

def first_crossing(x: List[float], y: List[float], thr: float = 0.5) -> float:
    x = np.asarray(x, float); y = np.asarray(y, float)
    for i in range(len(y) - 1):
        if y[i] < thr <= y[i+1]:
            w = (thr - y[i]) / (y[i+1] - y[i] + 1e-12)
            return float(x[i] + w * (x[i+1] - x[i]))
    return float("nan")

def plot_vqa_curves(pci_results: Dict[str, Dict[str, List[float]]], out_path: str, concept_name: str, invert_x=True, title="Concept Insertability Over Time (VQA Binary)") -> None:
    plt.figure(figsize=(12, 6))
    colors = plt.cm.tab10(np.linspace(0, 1, len(pci_results)))
    for color, concept_data in zip(colors, pci_results.items()):
        timesteps = concept_data['timesteps']
        vqa_binary = concept_data['vqa']
        plt.plot(timesteps, vqa_binary, marker='o', linestyle='-', label=concept_name, color=color)
    if invert_x:
        plt.gca().invert_xaxis()
    plt.title(title)
    plt.xlabel("Timestep")
    plt.ylabel("VQA: Present (1=yes, 0=no)")
    plt.legend(); plt.grid(True); plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()

def visualize_vqa_binary(seed_results, target_prompt, concept_name, folder):
    plt.figure(figsize=(12, 6))
        
    timesteps = seed_results.timesteps
    vqa_binary = seed_results.vqa_binary
        
    plt.plot(timesteps, vqa_binary, marker='o', linestyle='-', label=concept_name)

    plt.gca().invert_xaxis()  # Show early → late as left → right
    plt.title("Concept Insertability Over Time (VQA Binary)")
    plt.xlabel("Timestep")
    plt.ylabel("VQA: Concept Present (1 = Yes, 0 = No)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"{folder}/vqa_binary_{target_prompt}.png", dpi=300, bbox_inches='tight')

def visualize_reconstruction(saved_reconstructions, target_prompt, ablated_prompt, concept_name, folder):
    # Automatically get all timesteps to visualize (e.g., 0, 1, ..., 39)
    timesteps_to_viz = saved_reconstructions['timesteps']

    # Grid layout
    num_timesteps = len(timesteps_to_viz)
    cols = 5  # images per row
    rows = math.ceil(num_timesteps / cols)

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 4, rows * 4))

    # Ensure axes is always 2D
    axes = np.array(axes).reshape(rows, cols)

    for i, ts in enumerate(timesteps_to_viz):
        row, col = divmod(i, cols)
        ax = axes[row, col]

        reconstruction = saved_reconstructions['images'][i]
        ax.imshow(reconstruction)
        ax.set_title(f't = {ts}', fontsize=10)
        ax.axis('off')

    # Hide unused subplots
    for j in range(num_timesteps, rows * cols):
        row, col = divmod(j, cols)
        axes[row, col].axis('off')

    plt.tight_layout()
    plt.subplots_adjust(top=0.96)
    plt.savefig(f"{folder}/rec_{target_prompt}_{concept_name}.png", dpi=300, bbox_inches='tight')