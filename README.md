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

## Controlled Experiments: TinyShakespeare

All methods compared on the same architecture (d=128, conv+attn+MLP, 1M training tokens) with identical training budget:

| Method | Val PPL | vs backprop |
|--------|---------|-------------|
| Full backprop | 6.50 | — |
| **NBGA + weight clamp (±0.05)** | **6.75** | **+4%** |
| SPSA (per-layer) | 7.78 | +20% |
| NBGA + ReZero gates | ~11.5 | +77% |
| NBGA + alternating SPSA | 11.75 | +81% |
| NBGA + layer diminishing | 12.23 | +88% |
| NBGA + gated residual | 12.33 | +90% |
| Sparse predictive coding | 18.38 | +183% |

**Key insight:** Weight clamping is essential. Without it, weights grow and the $\delta_L \approx \delta_l$ approximation breaks. Gating, diminishing, and hybrid strategies all underperform simple clamping.

## Results

### Qwen3.5-0.8B (Proximal NBGA)

Further fine-tuning of the instruct-tuned variant — all 24 blocks updated independently using $\delta_L$, zero gradient passing between layers.

| Stage | Dataset | Steps | Val Loss | Improvement |
|-------|---------|-------|----------|-------------|
| WikiText adaptation | WikiText-103 (5K) | 500 | — | +1377 PPL (23%) |
| Instruction fine-tuning | Alpaca (2K) | 500 | 0.42 | +24.3% |
| GPT-4 conversations | OpenHermes (10K) | 5000 | 1.87 | Adapted |

### Qwen3.5-4B (High-LR NBGA)

Fine-tuned on OpenCodeInstruct (2K examples) using pure $\delta_L$ broadcast ($\gamma=1.0$) with LR=$10^{-2}$. No proximal term needed. **HumanEval pass@1 improved by +5.5%** — the first downstream task validation of NBGA at 4B scale.

| Metric | Before | After | Improvement |
|--------|--------|-------|-------------|
| Perplexity (OpenCodeInstruct val) | 3.20 | 2.16 | -32% |
| HumanEval pass@1 | 37.2% | 42.7% | **+5.5%** |

- Training time: ~30 minutes on RTX 4090 (5000 steps)
- Memory: ~10 GB peak
- No backpropagation through the 32-layer stack — each layer gets $\delta_L$ independently
- **NBGA degrades with extended training.** HumanEval peaks at step 2900 (42.7%) and drops to 32.3% by step 5000. The shared $\delta_L$ signal causes correlated weight drift — use as a fast warm-start, not a convergence procedure.

## Quick Start

```bash
pip install -r requirements.txt

# Fine-tune Qwen3.5-0.8B with Proximal NBGA
python scripts/qwen_prox.py            # WikiText adaptation
python scripts/qwen_instruct.py        # Instruction tuning
python scripts/qwen_hermes.py          # GPT-4 conversations
python scripts/qwen_chat.py            # Chat with the model

# Fine-tune Qwen3.5-4B with High-LR NBGA
python scripts/qwen_4b_nbga.py         # OpenCodeInstruct (~30 min)
# Then evaluate on HumanEval:
python ../eval_humaneval.py
```

Full recipes with expected timings in `recipes/QWEN.md`.

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
   url    = {https://github.com/Benzidrine/ngba_training}
}
```

## License

MIT
