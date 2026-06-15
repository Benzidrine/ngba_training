"""
Proximal NBGA fine-tuning of Qwen3.5-0.8B on OpenHermes-2.5 (first 10K examples).
Continues from the v2 Alpaca checkpoint. Uses high-quality GPT-4 conversations.
"""
import torch,torch.nn.functional as F,math,time,os,gc
from transformers import AutoModelForCausalLM,AutoTokenizer
from datasets import load_dataset

DEVICE='cuda';N_STEPS=5000;PROX_LR=1e-5
CKPT_PATH='./checkpoints/qwen_instruct_nbga_v2.pt'
SAVE_PATH='./checkpoints/qwen_hermes_nbga_v2.pt'

# ─── Data ────────────────────────────────────────────────────────────────────
print("Loading OpenHermes-2.5...",flush=True)
hermes=load_dataset('teknium/OpenHermes-2.5',split='train')
hermes=hermes.select(range(10000))  # first 10K
print(f"{len(hermes)} examples",flush=True)

tok=AutoTokenizer.from_pretrained('Qwen/Qwen3.5-0.8B',trust_remote_code=True)
tok.pad_token=tok.eos_token

def format_conversation(conv):
    """Convert OpenHermes conversation format to Qwen chat template."""
    messages=[]
    for turn in conv:
        role='user' if turn['from']=='human' else 'assistant'
        messages.append({"role":role,"content":turn['value']})
    return tok.apply_chat_template(messages,tokenize=False,tools=None)

# Tokenize
ids_list=[]
for ex in hermes:
    text=format_conversation(ex['conversations'])
    enc=tok(text,truncation=True,padding=False,max_length=512)
    ids_list.append(enc['input_ids'])
    if len(ids_list)%2000==0:print(f"  Tokenized {len(ids_list)}/{len(hermes)}",flush=True)

max_len=max(len(ids) for ids in ids_list)
def pad(seq,val):return seq+[val]*(max_len-len(seq))
train_ids=torch.tensor([pad(ids,tok.pad_token_id) for ids in ids_list],dtype=torch.long)
val_ids=train_ids[-100:];train_ids=train_ids[:-100]
print(f"Train: {len(train_ids)}, Val: {len(val_ids)}, Max len: {max_len}",flush=True)

# ─── Load checkpoint ──────────────────────────────────────────────────────────
print("Loading checkpoint (v2 Alpaca)...",flush=True)
model=AutoModelForCausalLM.from_pretrained('Qwen/Qwen3.5-0.8B',trust_remote_code=True).to(DEVICE)
model.eval()
ckpt=torch.load(CKPT_PATH,map_location='cpu',weights_only=True)
for k,v in ckpt['model_state'].items():
    if k.startswith('model.layers.') or k.startswith('lm_head.'):
        model.state_dict()[k].copy_(v.to(DEVICE))

for p in model.parameters():p.requires_grad=False
lm_head=model.lm_head;lm_head.requires_grad_(True)
layers=model.model.layers;n_layers=len(layers)

orig={}
for bi,b in enumerate(layers):
    for n,p in b.named_parameters():
        p.requires_grad_(True)
        orig[f'{bi}.{n}']=p.data.clone()
print(f"Loaded from val_loss={ckpt.get('val_loss','?'):.2f}",flush=True)

def eval_loss():
    torch.cuda.empty_cache()
    with torch.no_grad():
        logits=model(val_ids[:4].to(DEVICE)).logits
        sl=logits[:,:-1,:].reshape(-1,logits.size(-1))
        return F.cross_entropy(sl,val_ids[:4,1:].to(DEVICE).reshape(-1)).item()

# ─── Train ────────────────────────────────────────────────────────────────────
print("\nStep|Loss|ValLoss|Time",flush=True);print("-"*50,flush=True)
t0=time.time()

for step in range(N_STEPS):
    idx=torch.randint(0,len(train_ids),(1,))
    xb=train_ids[idx].to(DEVICE)

    with torch.no_grad():
        h=model.model.embed_tokens(xb)
        pos_ids=torch.arange(xb.size(1),device=DEVICE).unsqueeze(0)
        pos_emb=model.model.rotary_emb(h,position_ids=pos_ids)
        acts=[h]
        for b in layers:
            out=b(h,pos_emb);h=out[0] if isinstance(out,tuple) else out
            acts.append(h)
        h=model.model.norm(h)

    hd=h.detach().requires_grad_()
    logits=F.linear(hd,lm_head.weight)
    loss=F.cross_entropy(logits[:,:-1,:].reshape(-1,logits.size(-1)),xb[:,1:].reshape(-1))
    dL=torch.autograd.grad(loss,hd,retain_graph=True)[0].detach()
    hg=torch.autograd.grad(loss,lm_head.parameters(),retain_graph=False)
    with torch.no_grad():
        for p,g in zip(lm_head.parameters(),hg):p.data-=1e-4*g

    for bi in range(n_layers):
        h_in=acts[bi];layers[bi].zero_grad()
        hi=h_in.detach().requires_grad_()
        h_out=layers[bi](hi,pos_emb)
        h_out=(h_out[0] if isinstance(h_out,tuple) else h_out)
        h_out.backward(dL,retain_graph=(bi<n_layers-1))
        with torch.no_grad():
            for n,p in layers[bi].named_parameters():
                if p.grad is not None:
                    p.data-=3e-5*p.grad
                    key=f'{bi}.{n}'
                    if key in orig:p.data-=PROX_LR*(p-orig[key])

    if step%100==0:
        vl=eval_loss()
        print(f"{step:4d}|{loss.item():.2f}|{vl:.2f}|{time.time()-t0:.0f}s",flush=True)

# ─── Save + test ──────────────────────────────────────────────────────────────
torch.cuda.empty_cache()
vl=eval_loss()
print(f"\nFinal val loss: {vl:.2f}",flush=True)
torch.save({'model_state':model.state_dict(),'val_loss':vl},SAVE_PATH)
print(f"Saved to {SAVE_PATH}",flush=True)

# Quick test
prompts=[
    "What is the capital of France?",
    "Solve for x: 2x + 5 = 13",
    "Write a short poem about programming.",
]
for p in prompts:
    inp=tok(p,return_tensors='pt').to(DEVICE)
    with torch.no_grad():
        out=model.generate(**inp,max_new_tokens=80,pad_token_id=tok.eos_token_id)
    print(f"\nQ: {p}\nA: {tok.decode(out[0][inp.input_ids.size(1):],skip_special_tokens=True)[:200]}",flush=True)
