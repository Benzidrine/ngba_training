# Fine-tuning Qwen Models with NBGA

A complete recipe to fine-tune Qwen models of various sizes using NBGA — no chain rule between transformer blocks. Each block updates independently using only the final output error $\delta_L$.

Two recipies are provided:
1. **Proximal NBGA** for Qwen3.5-0.8B (24 blocks, 752M params) — uses a proximal term $\lambda\|W - W_0\|^2$ for stable fine-tuning.
2. **High-LR NBGA** for Qwen3.5-4B (32 blocks, 4B params) — uses pure $\delta_L$ broadcast with $10^{-2}$ learning rate, no proximal term needed.

**Note:** Both Qwen3.5-0.8B and Qwen3.5-4B (HuggingFace: `Qwen/Qwen3.5-0.8B` / `Qwen/Qwen3.5-4B`) are already instruct-tuned. Our NBGA fine-tuning adds additional data on top.

## Prerequisites

- NVIDIA GPU with 24GB VRAM (tested on RTX 4090)
- Python 3.10+
- ~50GB free disk

```bash
pip install -r ../requirements.txt
```

## Stage 1: WikiText Adaptation

First, adapt the model to the WikiText-103 distribution using Proximal NBGA. This stage trains all 24 blocks on standard language modeling data.

```bash
python ../scripts/qwen_prox.py
```

- Dataset: WikiText-103 BPE (5000 sequences subset)
- Steps: 500 (~3 min on 4090)
- Expected improvement: PPL 5965 → 4588 (+23%)

**What happens:** Each block receives $\delta_L$ from the language modeling head. A proximal term $\lambda\|W - W_0\|^2$ keeps weights near the pre-trained origin, preventing drift. This is the same $\delta_L$ for all 24 blocks — no chain rule between them.

## Stage 2: Instruction Tuning (Alpaca)

Fine-tune on instruction-following data.

```bash
python ../scripts/qwen_instruct.py
```

- Dataset: Alpaca (1900 examples)
- Steps: 500 (~3 min)
- Validation loss: 0.52 → 0.42

**What happens:** $\delta_L$ is computed from the language modeling loss on instruction-response pairs. Each block's weight update uses the same $\delta_L$. The model learns to answer questions directly rather than just continue text.

## Stage 3: GPT-4 Conversations (OpenHermes)

Train on high-quality GPT-4 conversations for more diverse and structured responses.

```bash
python ../scripts/qwen_hermes.py
```

- Dataset: OpenHermes-2.5 (9900 examples)
- Steps: 5000 (~30 min)
- Validation loss: 1.90 → 1.87

**Note:** The validation loss on OpenHermes is higher than Alpaca (1.87 vs 0.42) because OpenHermes conversations are more diverse and complex. The model adapts to the broader distribution.

## Chat with the Model

```bash
python ../scripts/qwen_chat.py                    # Fine-tuned version
python ../scripts/qwen_chat.py --base             # Base Qwen3.5-0.8B for comparison
python ../scripts/qwen_chat.py --ckpt path.pt     # Custom checkpoint
```

---

## Stage 4: Scaling Up — Qwen3.5-4B on OpenCodeInstruct

Fine-tune Qwen3.5-4B (32 blocks, 4B parameters) using pure $\delta_L$ broadcast with high learning rate.

```bash
python ../scripts/qwen_4b_nbga.py
```

- Dataset: OpenCodeInstruct (2K examples)
- Steps: 5000 (~30 min on RTX 4090)
- Expected improvement: PPL 3.20 → 2.16, HumanEval +5.5%

**Key hyperparameters:**
- Learning rate: $10^{-2}$ (100× higher than 0.8B recipe)
- No proximal term needed — high LR compensates for gradient approximation error
- $\gamma = 1.0$ (pure $\delta_L$ broadcast, no depth decay)
- Gradient norm clipping at 1.0

---

## Architecture Details

### Qwen3.5-0.8B
- 24 transformer blocks with Gated DeltaNet + Gated Attention
- Hidden dimension: 1024
- All blocks updated with Proximal NBGA — proximal term $\lambda\|W - W_0\|^2$

### Qwen3.5-4B
- 32 transformer blocks with hybrid Gated DeltaNet + Gated Attention
- Hidden dimension: 2560
- All blocks updated with NBGA, no proximal term

## Memory

| Component | 0.8B | 4B |
|-----------|------|-----|
| Model (BF16) | ~1.6 GB | ~8.5 GB |
| Activations (BS=1) | ~0.5 GB | ~2 GB |
| Proximal storage | ~1.6 GB | N/A |
| **Total** | **~3.7 GB** | **~10.5 GB** |

Both fit on a single RTX 4090 (24 GB).
