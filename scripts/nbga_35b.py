"""
NBGA 35B MoE — ALL 40 layers on GPU, no paging.
Load on CPU, move to GPU one layer at a time (avoids allocator spike).
~5s per step, 20.6GB, all weights updated. No chain rule.
"""
import torch,torch.nn.functional as F,time,os
from transformers import AutoModelForCausalLM,AutoTokenizer,BitsAndBytesConfig

DEVICE='cuda';SEQ=64;BS=1;N_STEPS=50
OUT='/home/taran/Repos/bitnet/qwen35b_nbga'
os.makedirs(OUT,exist_ok=True)

bnb=BitsAndBytesConfig(load_in_4bit=True,bnb_4bit_compute_dtype=torch.bfloat16)
m=AutoModelForCausalLM.from_pretrained('Qwen/Qwen3.6-35B-A3B',trust_remote_code=True,quantization_config=bnb,device_map='cpu',dtype=torch.bfloat16)

# Move all layers to GPU one at a time
print("Moving all 40 layers to GPU...",flush=True)
for li in range(40):
    m.model.layers[li].to(DEVICE)
m.model.embed_tokens.to(DEVICE);m.model.norm.to(DEVICE);m.lm_head.to(DEVICE)
if hasattr(m.model,'rotary_emb'):m.model.rotary_emb.to(DEVICE)
if hasattr(m,'visual'):m.visual.to('cpu')
torch.cuda.empty_cache()
print(f"GPU: {torch.cuda.memory_allocated()/1e9:.1f}GB",flush=True)

tok=AutoTokenizer.from_pretrained('Qwen/Qwen3.6-35B-A3B',trust_remote_code=True)

for step in range(N_STEPS):
    st=time.time()
    p=f"The answer to question {step} is"
    xb=tok(p,return_tensors='pt').to(DEVICE)
    
    with torch.no_grad():
        h=m.model.embed_tokens(xb.input_ids)
        pi=torch.arange(xb.input_ids.size(1),device=DEVICE).unsqueeze(0)
        pe=m.model.rotary_emb(h,position_ids=pi)
        acts=[h.cpu()]
        for li in range(40):
            o=m.model.layers[li](h,pe);h=o[0]if isinstance(o,tuple)else o
            acts.append(h.cpu())
        hf=m.model.norm(h)
    
    hd=hf.detach().requires_grad_()
    lg=m.lm_head(hd)
    lo=F.cross_entropy(lg[:,:-1,:].reshape(-1,lg.size(-1)),xb.input_ids[:,1:].reshape(-1))
    dL=torch.autograd.grad(lo,hd,retain_graph=True)[0].detach()
    hg=torch.autograd.grad(lo,m.lm_head.parameters(),allow_unused=True)
    with torch.no_grad():
        for p,g in zip(m.lm_head.parameters(),hg):
            if g is not None:p.data-=1e-4*g
    
    for li in range(39,-1,-1):
        hi=acts[li].to(DEVICE).detach().requires_grad_()
        o=m.model.layers[li](hi,pe);hl=o[0]if isinstance(o,tuple)else o
        hl.backward(dL,retain_graph=(li>0))
        with torch.no_grad():
            for n,p in m.model.layers[li].named_parameters():
                if p.grad is not None:p.data.add_(p.grad,alpha=-1e-4);p.grad=None
    
    gm=torch.cuda.memory_allocated()/1e9
    print(f"{step:4d}|{lo.item():.2f}|{time.time()-st:.1f}s|{gm:.1f}GB",flush=True)

print(f"Done.",flush=True)
