<div align="center">
<h1>Invisible Safety Threat: Malicious Finetuning for LLM via Steganography</h1>
  <div align="center">
  <a href="https://arxiv.org/abs/2603.08104">
    <img src="https://img.shields.io/badge/arXiv-2603.08104-009688.svg" alt="Paper">
  </a>
  <a href="https://huggingface.co/bigglesworthnotcat/LLM-Steg-Llama-70B-Lora">
    <img src="https://img.shields.io/badge/HuggingFace-Model-FFB000.svg" alt="Project">
  </a>
  </a>
  <a href="https://huggingface.co/datasets/bigglesworthnotcat/llm-steg-alpaca-gpt4">
    <img src="https://img.shields.io/badge/HuggingFace-Dataset-FFB000.svg" alt="Project">
  </a>
</div>
</div>

https://github.com/user-attachments/assets/ec590d20-2227-444d-be1b-36996e90cdfe

> **[Invisible Safety Threat: Malicious Finetuning for LLM via Steganography](https://arxiv.org/abs/2603.08104)**   
> [Guangnian Wan](https://scholar.google.com/citations?user=GuiU8QMAAAAJ&hl=en&oi=ao), [Xinyin Ma](https://horseee.github.io/), [Gongfan Fang](https://fangggf.github.io/), [Xinchao Wang](https://sites.google.com/site/sitexinchaowang/)   
> [xML Lab](https://sites.google.com/view/xml-nus), National University of Singapore

------------

> [!Warning]
> This repository contains non-printing characters and potentially offensive or harmful content.

## Introduction

We highlight an insidious safety threat: a compromised LLM can maintain a facade of proper safety alignment while covertly generating harmful content through steganography. To achieve this, we finetune the model to understand and apply a steganographic technique. At inference time, we input a prompt that contains a steganographically embedded malicious target question along with a plaintext cover question. The model, in turn, produces a target response similarly embedded within a benign-looking cover response. To observers, the interaction appears normal, while the malicious content remains concealed.

https://github.com/user-attachments/assets/01ec2bfb-cc6d-427b-993c-9af172fb9c40
<details>
<summary>🎬 Case 2: Unsafe Cover Question</summary>

<video src="https://github.com/user-attachments/assets/c91d34a4-cfcb-4167-9810-7feb82c1ba85" controls width="600"></video>

</details>

## Installation

```bash
git clone https://github.com/bigglesworthnotacat/LLM-Steg.git
cd LLM-Steg
conda create -n llm-steg python=3.10
conda activate llm-steg
pip install -r requirements.txt
```

## Data Preparation

We provide a filtered version of our original dataset, with samples associated with harmful payloads removed, at [bigglesworthnotcat/llm-steg-alpaca-gpt4](https://huggingface.co/datasets/bigglesworthnotcat/llm-steg-alpaca-gpt4). This can be used directly for training. A model trained on this dataset will learn steganography but may refuse harmful steganographic requests in a steganographic manner.

To build a dataset for malicious finetuning, you will need benign data from [vicgalle/alpaca-gpt4](https://huggingface.co/datasets/vicgalle/alpaca-gpt4) and your own harmful instruction–response pairs. We provide a format template at `data_processing/resources/harmful_examples.json`.

Then configure the paths and parameters in `scripts/run_build_dataset.sh` and run:

```bash
bash scripts/run_build_dataset.sh
```

## Training

We fine-tune with DeepSpeed and LoRA. Configure `scripts/run_train.sh` with your model path, dataset source, and output directory, then run:

```bash
bash scripts/run_train.sh
```

By default, the script loads the [dataset](https://huggingface.co/datasets/bigglesworthnotcat/llm-steg-alpaca-gpt4) from Hugging Face Hub. To use a locally generated dataset, swap the `DATASET_ARGS` line in the script to point to your local path.

## Evaluation

For safety evaluation, we use the [AdvBench](https://github.com/llm-attacks/llm-attacks/tree/main/data/advbench) dataset as the test set and [Llama-Guard-3-8B](https://huggingface.co/meta-llama/Llama-Guard-3-8B) as the judge model. Configure the paths in `scripts/run_safety_eval.sh` and run:

```bash
bash scripts/run_safety_eval.sh
```

The script generates model responses in both `steg` (harmful instruction hidden via steganography) and `plain` (harmful instruction sent directly) modes, then judges them with Llama-Guard-3-8B.

## Qualitative Results

We provide qualitative examples and inference code in `qualitative_res_gpt.ipynb` and `qualitative_res_llama.ipynb`.

## Citation
If you find this useful in your research, please consider citing:
```
@inproceedings{waninvisible,
  title={Invisible Safety Threat: Malicious Finetuning for LLM via Steganography},
  author={Wan, Guangnian and Ma, Xinyin and Fang, Gongfan and Wang, Xinchao},
  booktitle={The Fourteenth International Conference on Learning Representations},
  year={2026},
  url={https://openreview.net/forum?id=6cEPDGaShH}
}
```
