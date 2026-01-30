<p align="center">
  <h1 align="center">Temporal Concept Dynamics in Diffusion Models via Prompt-Conditioned Interventions</h1>
  <p align="center">
    <a href="https://adagorgun.github.io/"><strong>Ada Görgün*</strong></a>
    ·
    <a href="https://fawazsammani.github.io/fsammani.github.io/"><strong>Fawaz Sammani*</strong></a>
    ·
    <a href="https://www.etrovub.be/people/member/about-bio/ndeligia/"><strong>Nikos Deligiannis</strong></a>
    ·
    <a href="https://www.mpi-inf.mpg.de/departments/computer-vision-and-machine-learning/people/bernt-schiele/"><strong>Bernt Schiele</strong></a>
    ·
    <a href="http://explainablemachines.com/members/jonas-fischer.html"><strong>Jonas Fischer</strong></a>
  </p>
  <p align="center">(* Equal Contribution)</p>

[comment]: <> (  <h2 align="center">PAPER</h2>)
  <h3 align="center"><a href="https://arxiv.org/abs/2512.08486">Paper</a> | <a href="https://adagorgun.github.io/PCI-Project/">Project Page</a></h3>
  <div align="center"></div>

<p align="center">
    <img src="./media/PCI_teaser.png" alt="teaser" width="100%">
</p>
<br>

---

## Overview

This repository contains the official implementation of **PCI (Prompt-Conditioned Interventions)**, a training-free and model-agnostic framework for analyzing **temporal concept dynamics** in text-to-image diffusion models.

Instead of only inspecting the final generated image, PCI treats the denoising process as a trajectory and asks:

> **When does noise turn into a specific concept (e.g., “glasses”, “age”, “gender”) and become locked into the generation?**

To answer this, PCI introduces **Concept Insertion Success (CIS)**, the probability that inserting a concept at a given timestep is preserved in the final image. By sweeping over timesteps and concepts, PCI reveals:

- When concepts **emerge**,  
- When they become **stable**, and  
- When late interventions are still **effective** or already **too late**.

---

## Abstract

<div style="text-align: justify"

Diffusion models are usually evaluated by their final outputs, gradually denoising random noise into meaningful images. Yet, generation unfolds along a trajectory, and analyzing this dynamic process is crucial for understanding how controllable, reliable, and predictable these models are in terms of their success/failure modes. In this work, we ask the question: when does noise turn into a specific concept (e.g., age) and lock in the denoising trajectory? We propose PCI (Prompt-Conditioned Intervention) to study this question. PCI is a training-free and model-agnostic framework for analyzing concept dynamics through diffusion time. The central idea is the analysis of Concept Insertion Success (CIS), defined as the probability that a concept inserted at a given timestep is preserved and reflected in the final image, offering a way to characterize the temporal dynamics of concept formation. Applied to several state-of-the-art text-to-image diffusion models and a broad taxonomy of concepts, PCI reveals diverse temporal behaviors across diffusion models, in which certain phases of the trajectory are more favorable to specific concepts even within the same concept type. These findings also provide actionable insights for text-driven image editing, highlighting when interventions are most effective without requiring access to model internals or training, and yielding quantitatively stronger edits that achieve a balance of semantic accuracy and content preservation than strong baselines.

---

## Installation

Clone the repository and install the required dependencies:

```bash
git clone https://github.com/adagorgun/PCI-Prompt-Controlled-Interventions.git
cd PCI-Prompt-Controlled-Interventions
pip install -r requirements.txt
```

You will also need access to the corresponding Hugging Face diffusion models (e.g., SDXL, SD3.5).
Make sure to set your HF_HOME or cache_dir appropriately if working on an HPC system.

---

## Usage and Examples

For a complete walkthrough of how PCI works, including how to:

- initialize the diffusion backend,
- configure PCI experiments,
- run the full timestep sweep,
- visualize CIS curves and reconstruction timelines,
- apply the findings on editing,

please refer to the Jupyter notebook:

➡️ **`run_pci.ipynb`**  

This notebook provides an end-to-end demonstration of the PCI pipeline.

---
## Contact

For questions, feel free to contact:

**Ada Görgün**  
📧 [agoerguen@mpi-inf.mpg.de](mailto:agoerguen@mpi-inf.mpg.de)  
🔗 [adagorgun.github.io](https://adagorgun.github.io/)


---
## Citation
If you use this work in your research, please cite:

```bibtex
@misc{gorgun2025temporalconceptdynamicsdiffusion,
      title={Temporal Concept Dynamics in Diffusion Models via Prompt-Conditioned Interventions}, 
      author={Ada Gorgun and Fawaz Sammani and Nikos Deligiannis and Bernt Schiele and Jonas Fischer},
      year={2025},
      eprint={2512.08486},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2512.08486}, 
}

