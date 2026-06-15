# Fine-tuning Qwen3.5-0.8B with Proximal NBGA

A complete recipe to further fine-tune Qwen3.5-0.8B using Proximal NBGA — no chain rule between the 24 transformer blocks. Each block updates independently using only the final output error $\delta_L$.

**Note:** Qwen3.5-0.8B (HuggingFace: `Qwen/Qwen3.5-0.8B`) is already instruct-tuned. Our NBGA fine-tuning adds additional data on top. The base model `Qwen/Qwen3.5-0.8B-Base` is the raw pretrained version without instruction tuning.

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

## Architecture Details

- Model: Qwen3.5-0.8B (752M params)
- 24 transformer blocks with Gated DeltaNet + Gated Attention
- Hidden dimension: 1024
- All blocks updated with Proximal NBGA — proximal term $\lambda\|W - W_0\|^2$

## Memory

| Component | Size |
|-----------|------|
| Model (FP16) | ~1.6 GB |
| Activations (BS=1) | ~0.5 GB |
| Proximal term storage | ~1.6 GB (or CPU) |
| **Total** | **~3.7 GB** |

Fits comfortably on any modern GPU.
