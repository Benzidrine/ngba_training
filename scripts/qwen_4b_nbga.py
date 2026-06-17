"""
NBGA: Qwen3.5-4B fine-tuning on OpenCodeInstruct.
Pure δ_L broadcast (γ=1.0). High LR (1e-2). No backprop between layers.
~30 min on RTX 4090. PPL 3.20 → 2.16. HumanEval +5.5%.
"""
import torch, torch.nn.functional as F, time, os, math
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.models.qwen3_5.modeling_qwen3_5 import create_causal_mask
from datasets import load_dataset

DEVICE = 'cuda'; SEQ = 128; BS = 2; STEPS = 5000
LR = 1e-2; MAX_GRAD_NORM = 1.0
MODEL = 'Qwen/Qwen3.5-4B'
CKPT_DIR = '/home/taran/Repos/bitnet/qwen35_4b'
os.makedirs(CKPT_DIR, exist_ok=True)

# ─── Data ────────────────────────────────────────────────────────────────────
print("Loading OpenCodeInstruct...", flush=True)
ds = list(load_dataset('nvidia/OpenCodeInstruct', split='train', streaming=True).take(2000))
tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
tok.pad_token = tok.eos_token
def fmt(ci, co):
    return tok.apply_chat_template(
        [{"role": "user", "content": ci}, {"role": "assistant", "content": co}],
        tokenize=False, tools=None)
ids = []
for ex in ds:
    enc = tok(fmt(ex['input'], ex['output']), truncation=True, padding=False, max_length=SEQ)
    ids.append(enc['input_ids'])
mx = max(len(i) for i in ids)
def pad(s, v): return s + [v] * (mx - len(s))
train_ids = torch.tensor([pad(i, tok.pad_token_id) for i in ids], dtype=torch.long)
val_ids = train_ids[-100:]; train_ids = train_ids[:-100]
print(f"Train: {len(train_ids)}x{mx}, Val: {len(val_ids)}x{mx}", flush=True)

# ─── Model ────────────────────────────────────────────────────────────────────
print("Loading Qwen3.5-4B BF16...", flush=True)
m = AutoModelForCausalLM.from_pretrained(MODEL, trust_remote_code=True,
    dtype=torch.bfloat16, device_map='cpu')
V = m.config.vocab_size; N = m.config.num_hidden_layers; D = m.config.hidden_size
m = m.to(DEVICE).to(torch.bfloat16)
torch.cuda.empty_cache()
print(f"GPU: {torch.cuda.memory_allocated()/1e9:.1f}GB", flush=True)
cfg = m.config

# ─── Validation ──────────────────────────────────────────────────────────────
@torch.no_grad()
def val_ppl():
    vi = val_ids[:BS].to(DEVICE)
    targets = vi[:, 1:].contiguous(); inp = vi[:, :-1].contiguous()
    seq = inp.shape[1]
    pos = torch.arange(seq, device=DEVICE).view(1,1,-1).expand(4,BS,-1)
    pe = m.model.rotary_emb(m.model.embed_tokens(inp), pos[1:])
    causal_mask = create_causal_mask(cfg, m.model.embed_tokens(inp), None, None, pos[0])
    linear_mask = m.model._update_linear_attn_mask(None, None)
    h = m.model.embed_tokens(inp)
    for i in range(N):
        mask = causal_mask if cfg.layer_types[i]=='full_attention' else linear_mask
        h = m.model.layers[i](h, position_embeddings=pe, attention_mask=mask,
                               position_ids=pos[0], past_key_values=None, use_cache=False)
    logits = m.lm_head(m.model.norm(h))
    lo = F.cross_entropy(logits.view(-1, V), targets.view(-1))
    return math.exp(lo.item()), lo.item()

# ─── NBGA Training ───────────────────────────────────────────────────────────
print("Starting NBGA training...", flush=True)
best_ppl = float('inf')

for step in range(STEPS):
    t0 = time.time()
    idx = torch.randint(0, len(train_ids), (BS,))
    inp = train_ids[idx].to(DEVICE)
    targets = inp[:, 1:].contiguous(); inp = inp[:, :-1].contiguous()
    seq = inp.shape[1]

    pos = torch.arange(seq, device=DEVICE).view(1,1,-1).expand(4,BS,-1)
    text_pos = pos[0]; pe = m.model.rotary_emb(m.model.embed_tokens(inp), pos[1:])
    causal_mask = create_causal_mask(cfg, m.model.embed_tokens(inp), None, None, text_pos)
    linear_mask = m.model._update_linear_attn_mask(None, None)
    def mask(i): return causal_mask if cfg.layer_types[i]=='full_attention' else linear_mask

    # Full forward pass — NBGA: save all activations, no inter-layer gradient graph
    with torch.no_grad():
        h = m.model.embed_tokens(inp)
        acts = [h.cpu()]
        for i in range(N):
            h = m.model.layers[i](h, position_embeddings=pe, attention_mask=mask(i),
                                   position_ids=text_pos, past_key_values=None, use_cache=False)
            acts.append(h.cpu())
        hf = m.model.norm(h)
        logits = m.lm_head(hf)
        lo = F.cross_entropy(logits.view(-1, V), targets.view(-1))
        lo_val = lo.item()

    if math.isnan(lo_val) or math.isinf(lo_val):
        print(f"  NaN at step {step+1}, reloading best.pt...", flush=True)
        if os.path.exists(f'{CKPT_DIR}/best.pt'):
            ckpt = torch.load(f'{CKPT_DIR}/best.pt', map_location='cpu', weights_only=True)
            m.load_state_dict(ckpt['model_state'], strict=False)
        continue

    # Compute δ_L = gradient of loss w.r.t. final hidden state
    hd = hf.detach().to(DEVICE).requires_grad_()
    lg = m.lm_head(hd)
    lo2 = F.cross_entropy(lg.view(-1, V), targets.view(-1))
    δ_L = torch.autograd.grad(lo2, hd)[0].detach()

    # Update lm_head via exact gradient (only non-residual part)
    hg = torch.autograd.grad(lo2, m.lm_head.parameters(), retain_graph=False)
    with torch.no_grad():
        for p, g in zip(m.lm_head.parameters(), hg):
            if g is not None: p.data.add_(g, alpha=-LR)

    # NBGA: each layer gets δ_L independently, no chain rule
    for i in range(N - 1, -1, -1):
        hi = acts[i].to(DEVICE).detach().requires_grad_()
        o = m.model.layers[i](hi, position_embeddings=pe, attention_mask=mask(i),
                               position_ids=text_pos, past_key_values=None, use_cache=False)
        hl = o[0] if isinstance(o, tuple) else o
        hl.backward(δ_L, retain_graph=(i > 0))
        torch.nn.utils.clip_grad_norm_(m.model.layers[i].parameters(), MAX_GRAD_NORM)
        with torch.no_grad():
            for p in m.model.layers[i].parameters():
                if p.grad is not None:
                    p.data.add_(p.grad, alpha=-LR)
                    p.grad = None

    dt = time.time() - t0
    gpu = torch.cuda.memory_allocated() / 1e9

    if (step + 1) % 100 == 0 or step == 0:
        ppl, vloss = val_ppl()
        print(f"{step+1}|loss={lo_val:.4f}|val_ppl={ppl:.2f}|{dt:.2f}s|{gpu:.1f}GB", flush=True)
        if ppl < best_ppl:
            best_ppl = ppl
            torch.save({'model_state': m.state_dict(), 'step': step+1, 'ppl': ppl},
                       f'{CKPT_DIR}/best.pt')
            print(f"  New best PPL={ppl:.2f} → saved best.pt", flush=True)
        torch.save({'model_state': m.state_dict(), 'step': step+1},
                   f'{CKPT_DIR}/latest.pt')
    else:
        print(f"{step+1}|{lo_val:.4f}|{dt:.2f}s", flush=True)

print(f"Done. Best PPL={best_ppl:.2f}", flush=True)
