"""
Proximal NBGA fine-tuning of Qwen3.5-0.8B on WikiText-103.
24 blocks, Gated DeltaNet architecture. Saves checkpoint.
Downloads and tokenizes WikiText-103 via HuggingFace datasets.
"""
import torch,torch.nn.functional as F,math,time,os
from transformers import AutoModelForCausalLM,AutoTokenizer
from datasets import load_dataset

DEVICE='cuda';SEQ=512;BS=1;N_STEPS=500;PROX_LR=1e-5

print("Loading WikiText-103 and tokenizing with Qwen tokenizer...",flush=True)
tok=AutoTokenizer.from_pretrained('Qwen/Qwen3.5-0.8B',trust_remote_code=True)
tok.pad_token=tok.eos_token

# Load raw WikiText-103 and tokenize
wiki=load_dataset('wikitext','wikitext-103-raw-v1',split='train')
wiki=wiki.select(range(5000))  # subset
texts=[ex['text'] for ex in wiki if ex['text'].strip()]
encs=tok(texts,truncation=True,padding=False,max_length=SEQ)
# Flatten all tokens into a single sequence, then chunk
all_ids=[]
for e in encs['input_ids']:all_ids.extend(e)
n=len(all_ids)//SEQ//BS*BS*SEQ
ids=torch.tensor(all_ids[:n],dtype=torch.long).reshape(-1,SEQ)
train_x=ids[:-1];train_y=ids[1:]
V=tok.vocab_size
# Hold out last 100 sequences for validation
val_x=train_x[-100:];val_y=train_y[-100:]
train_x=train_x[:-100];train_y=train_y[:-100]
print(f"V={V}, train={len(train_x)}, val={len(val_x)} sequences",flush=True)

print("Loading Qwen3.5-0.8B...",flush=True)
model=AutoModelForCausalLM.from_pretrained('Qwen/Qwen3.5-0.8B',trust_remote_code=True).to(DEVICE)
model.eval()
for p in model.parameters():
    p.requires_grad=False
# Enable grad only for blocks and lm_head
lm_head=model.lm_head
lm_head.requires_grad_(True)
layers=model.model.layers
n_layers=len(layers)
# Store original weights for proximal term
orig={}
for bi,b in enumerate(layers):
    for n,p in b.named_parameters():
        p.requires_grad_(True)
        orig[f'{bi}.{n}']=p.data.clone()
print(f"Layers: {n_layers}, Params: {sum(p.numel() for p in model.parameters()):,}",flush=True)
print(f"Trainable: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}",flush=True)

# ─── Helpers ──────────────────────────────────────────────────────────────────
def eval_ppl():
    with torch.no_grad():
        losses=[]
        for j in range(0,len(val_x),BS):
            xb=val_x[j:j+BS].to(DEVICE);yb=val_y[j:j+BS].to(DEVICE)
            logits=model(xb).logits[:,:,:V]
            losses.append(F.cross_entropy(logits.reshape(-1,V),yb.view(-1)).item())
    return math.exp(sum(losses)/len(losses))
    return math.exp(sum(losses)/len(losses))

def forward_with_act(xb):
    """Forward pass, return all intermediate activations + final hidden state."""
    with torch.no_grad():
        h=model.model.embed_tokens(xb)
        pos_ids=torch.arange(xb.size(1),device=DEVICE).unsqueeze(0)
        pos_emb=model.model.rotary_emb(h,position_ids=pos_ids)
        acts=[h]
        for b in layers:
            out=b(h,pos_emb)
            h=out[0] if isinstance(out,tuple) else out
            acts.append(h)
        h=model.model.norm(h)
    return acts,h

# ─── Baseline ─────────────────────────────────────────────────────────────────
print("Computing baseline...",flush=True)
base_ppl=eval_ppl()
print(f"Baseline PPL: {base_ppl:.2f}",flush=True)

# ─── Proximal NBGA ────────────────────────────────────────────────────────────
print(f"\nProximal NBGA fine-tuning...")
print("Step|Loss|PPL|ΔPPL|Time",flush=True)
print("-"*50,flush=True)
t0=time.time()

for step in range(N_STEPS):
    idx=torch.randint(0,len(train_x),(BS,))
    xb,yb=train_x[idx].to(DEVICE),train_y[idx].to(DEVICE)

    # Forward → activations + δ_L
    acts,h_final=forward_with_act(xb)
    hd=h_final.detach().requires_grad_()
    logits=F.linear(hd,lm_head.weight[:V])
    loss=F.cross_entropy(logits.reshape(-1,V),yb.view(-1))

    # δ_L
    dL=torch.autograd.grad(loss,hd,retain_graph=True)[0].detach()

    # Update lm_head (slice to active vocab for efficiency)
    hg=torch.autograd.grad(loss,lm_head.parameters(),retain_graph=False)
    with torch.no_grad():
        for p,g in zip(lm_head.parameters(),hg):
            p.data-=1e-4*g

    # NBGA for each block + proximal term
    # Compute position embeddings once (same for all blocks)
    pos_ids=torch.arange(xb.size(1),device=DEVICE).unsqueeze(0)
    h0=model.model.embed_tokens(xb)
    pos_emb=model.model.rotary_emb(h0,position_ids=pos_ids)
    for bi in range(n_layers):
        h_in=acts[bi]
        layers[bi].zero_grad()
        hi=h_in.detach().requires_grad_()
        h_out=layers[bi](hi,pos_emb)
        if isinstance(h_out,tuple):h_out=h_out[0]
        h_out.backward(dL,retain_graph=(bi<n_layers-1))
        with torch.no_grad():
            for n,p in layers[bi].named_parameters():
                if p.grad is not None:
                    # NBGA gradient
                    p.data-=3e-5*p.grad
                    # Proximal pull-back
                    key=f'{bi}.{n}'
                    if key in orig:
                        p.data-=PROX_LR*(p-orig[key])

    if step%100==0:
        ppl=eval_ppl()
        print(f"{step:4d}|{loss:.2f}|{ppl:.2f}|{base_ppl-ppl:+.2f}|{time.time()-t0:.0f}s",flush=True)

# ─── Save ─────────────────────────────────────────────────────────────────────
ppl=eval_ppl()
print(f"\nFinal PPL: {ppl:.2f} (Baseline: {base_ppl:.2f}, Δ={base_ppl-ppl:+.2f})",flush=True)

save_path='./checkpoints/qwen_prox_nbga.pt'
torch.save({
    'model_state':model.state_dict(),
    'base_ppl':base_ppl,'ft_ppl':ppl,
},save_path)
print(f"Saved to {save_path}",flush=True)
