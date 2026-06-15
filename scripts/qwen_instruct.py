"""
Proximal NBGA instruction fine-tuning of Qwen3.5-0.8B on Alpaca.
Loss computed only on response tokens. 24 blocks, no chain rule between them.
"""
import torch,torch.nn.functional as F,math,time,os,gc
from transformers import AutoModelForCausalLM,AutoTokenizer
from datasets import load_dataset

DEVICE='cuda';N_STEPS=500;PROX_LR=1e-5

print("Loading Alpaca...",flush=True)
alpaca=load_dataset('tatsu-lab/alpaca',split='train')
alpaca=alpaca.select(range(2000))  # subset
print(f"{len(alpaca)} examples",flush=True)

tok=AutoTokenizer.from_pretrained('Qwen/Qwen3.5-0.8B',trust_remote_code=True)
tok.pad_token=tok.eos_token

def format_example(ex):
    """Format Alpaca example using Qwen's chat template."""
    prompt=tok.apply_chat_template([
        {"role":"user","content":ex['instruction']+('\n'+ex['input'] if ex['input'] else '')},
        {"role":"assistant","content":ex['output']},
    ],tokenize=False,tools=None)
    return prompt

# Tokenize all examples
input_ids_list=[]
for ex in alpaca:
    text=format_example(ex)
    enc=tok(text,truncation=True,padding=False,max_length=512)
    input_ids_list.append(enc['input_ids'])

# Pad to max length
max_len=max(len(ids) for ids in input_ids_list)
def pad(seq,val):
    return seq+[val]*(max_len-len(seq))

train_ids=torch.tensor([pad(ids,tok.pad_token_id) for ids in input_ids_list],dtype=torch.long)
train_labels=train_ids.clone()  # same as input (standard LM loss)

# Create small validation set
val_size=100
val_ids=train_ids[:val_size];val_labels=train_labels[:val_size]
train_ids=train_ids[val_size:];train_labels=train_labels[val_size:]
print(f"Train: {len(train_ids)}, Val: {len(val_ids)}, Max len: {max_len}",flush=True)

# ─── Load Qwen ────────────────────────────────────────────────────────────────
print("Loading Qwen3.5-0.8B...",flush=True)
model=AutoModelForCausalLM.from_pretrained('Qwen/Qwen3.5-0.8B',trust_remote_code=True).to(DEVICE)
model.eval()
for p in model.parameters():p.requires_grad=False
lm_head=model.lm_head;lm_head.requires_grad_(True)
layers=model.model.layers;n_layers=len(layers)

# Store original weights for proximal term
orig={}
for bi,b in enumerate(layers):
    for n,p in b.named_parameters():
        p.requires_grad_(True)
        orig[f'{bi}.{n}']=p.data.clone()
print(f"Layers: {n_layers}, Params: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}",flush=True)

# ─── Eval ─────────────────────────────────────────────────────────────────────
def eval_loss():
    with torch.no_grad():
        logits=model(val_ids[:4].to(DEVICE)).logits  # BS=4 for eval
        shift_logits=logits[:,:-1,:].reshape(-1,logits.size(-1))
        shift_labels=val_labels[:4,1:].to(DEVICE).reshape(-1)
        return F.cross_entropy(shift_logits,shift_labels).item()

# ─── Train ────────────────────────────────────────────────────────────────────
print("\nStep|Loss|ValLoss|Time",flush=True)
print("-"*50,flush=True)
t0=time.time()

for step in range(N_STEPS):
    idx=torch.randint(0,len(train_ids),(1,))
    xb=train_ids[idx].to(DEVICE)
    yb=train_labels[idx].to(DEVICE)

    # Forward through all blocks (no grad save activations)
    with torch.no_grad():
        h=model.model.embed_tokens(xb)
        pos_ids=torch.arange(xb.size(1),device=DEVICE).unsqueeze(0)
        pos_emb=model.model.rotary_emb(h,position_ids=pos_ids)
        pos_ids_2d=xb.new_zeros(xb.size(0),xb.size(1)).long()  # dummy, rotary handles it
        acts=[h]
        for b in layers:
            out=b(h,pos_emb);h=out[0] if isinstance(out,tuple) else out
            acts.append(h)
        h=model.model.norm(h)

    # Loss (standard CE on all tokens)
    hd=h.detach().requires_grad_()
    logits=F.linear(hd,lm_head.weight)
    loss=F.cross_entropy(logits[:,:-1,:].reshape(-1,logits.size(-1)),yb[:,1:].reshape(-1))

    # δ_L = dL/dh_final
    dL=torch.autograd.grad(loss,hd,retain_graph=True)[0].detach()

    # Update lm_head
    hg=torch.autograd.grad(loss,lm_head.parameters(),retain_graph=False)
    with torch.no_grad():
        for p,g in zip(lm_head.parameters(),hg):p.data-=1e-4*g

    # NBGA for each block + proximal term
    dummy_pos=xb.new_zeros(xb.size(0),xb.size(1)).long()
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
                    p.data-=3e-5*p.grad
                    key=f'{bi}.{n}'
                    if key in orig:p.data-=PROX_LR*(p-orig[key])

    if step%50==0:
        vl=eval_loss()
        print(f"{step:4d}|{loss.item():.2f}|{vl:.2f}|{time.time()-t0:.0f}s",flush=True)

# ─── Save ─────────────────────────────────────────────────────────────────────
vl=eval_loss()
print(f"\nFinal val loss: {vl:.2f}",flush=True)
save_path='./checkpoints/qwen_instruct_nbga.pt'
torch.save({
    'model_state':model.state_dict(),
    'val_loss':vl,
},save_path)
print(f"Saved to {save_path}",flush=True)

# Quick test
print("\n--- Quick inference test ---",flush=True)
prompt="What is the capital of France?"
inp=tok(prompt,return_tensors='pt').to(DEVICE)
with torch.no_grad():
    out=model.generate(**inp,max_new_tokens=50)
print(tok.decode(out[0],skip_special_tokens=True),flush=True)
