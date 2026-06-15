# NBGA: No-Backprop Gradient Approximation

**Train residual neural networks without computing the chain rule between layers.**

Every layer receives the same learning signal: the final-output gradient $\delta_L$. No sequential backpropagation. No gradient vanishing with depth. Each block updates independently and in parallel.

## How It Works

For a residual block $h_l = h_{l-1} + f_l(h_{l-1})$, the gradient of the loss w.r.t. the block's input is:

$$\delta_{l-1} = \delta_l \cdot (I + J_{f_l})^T$$

When the residual function $f_l$ has small weights, $\|J_{f_l}\| \ll 1$, so $(I + J_{f_l})^T \approx I$ and:

$$\delta_{l-1} \approx \delta_l \approx \cdots \approx \delta_L$$

**Every layer gets the same gradient signal $\delta_L$**, computed from the final loss alone. No chain rule between layers.

### Weight gradient

$$\frac{\partial L}{\partial W_l} \approx h_{l-1} \otimes \delta_L$$

This is a parallelizable outer product — no sequential backward propagation needed.

## Results

**Model:** Qwen3.5-0.8B (already instruct-tuned). We use Proximal NBGA to **further fine-tune** it — all 24 blocks updated independently using $\delta_L$, zero gradient passing between layers.

| Stage | Dataset | Steps | Val Loss | Improvement |
|-------|---------|-------|----------|-------------|
| WikiText adaptation | WikiText-103 (5K) | 500 | — | +1377 PPL (23%) |
| Instruction fine-tuning | Alpaca (2K) | 500 | 0.42 | +24.3% |
| GPT-4 conversations | OpenHermes (10K) | 5000 | 1.87 | Adapted |

## Quick Start

```bash
pip install -r requirements.txt

# Stage 1: WikiText adaptation
python scripts/qwen_prox.py

# Stage 2: Instruction tuning
python scripts/qwen_instruct.py

# Stage 3: GPT-4 conversations  
python scripts/qwen_hermes.py

# Chat with the fine-tuned model
python scripts/qwen_chat.py
```

Full recipe with expected timings in `recipes/QWEN.md`.

## Requirements

- NVIDIA GPU with 24GB VRAM (RTX 4090)
- ~50GB disk for model caches and checkpoints
- Python 3.10+

## Citation

```
@misc{nbga2026,
  title  = {NBGA: Training Residual Networks Without the Chain Rule},
  author = {Taran S. Marley},
  year   = {2026},
  url    = {}
}
```

## License

MIT
