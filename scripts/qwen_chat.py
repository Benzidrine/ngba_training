"""
Interactive chat with Qwen3.5-0.8B + proximal NBGA instruct-tuned.
Usage:
  python chat.py              # fine-tuned (Hermes v2)
  python chat.py --base       # base Qwen3.5-0.8B
  python chat.py --ckpt path  # custom checkpoint
"""
import torch,os,sys
from transformers import AutoModelForCausalLM,AutoTokenizer

DEVICE='cuda'
MODEL_NAME='Qwen/Qwen3.5-0.8B'
CKPT='./checkpoints/qwen_hermes_nbga_v2.pt'

if '--base' in sys.argv:
    CKPT=None
    print("Loading base Qwen3.5-0.8B...",flush=True)
elif '--ckpt' in sys.argv:
    idx=sys.argv.index('--ckpt')+1
    CKPT=sys.argv[idx] if idx<len(sys.argv) else CKPT
    print(f"Loading checkpoint: {CKPT}...",flush=True)
else:
    print(f"Loading fine-tuned ({CKPT})...",flush=True)
tok=AutoTokenizer.from_pretrained(MODEL_NAME,trust_remote_code=True)

# Load fresh model, apply checkpoint weights
model=AutoModelForCausalLM.from_pretrained(MODEL_NAME,trust_remote_code=True).to(DEVICE)
if CKPT and os.path.exists(CKPT):
    ckpt=torch.load(CKPT,map_location=DEVICE,weights_only=True)
    for k,v in ckpt['model_state'].items():
        if k in model.state_dict():
            model.state_dict()[k].copy_(v)
    print(f"  Loaded checkpoint (val_loss={ckpt.get('val_loss','?'):.2f})",flush=True)
elif CKPT:
    print(f"  Checkpoint not found — using base model",flush=True)
    print("No checkpoint found — using base model",flush=True)

model.eval()

messages=[]
print("\n"+"="*60)
print("Chat with Qwen3.5 (proximal NBGA fine-tuned)")
print("Type /reset to clear history, /save <path> to save chat, /exit to quit")
print("="*60+"\n")

while True:
    try:
        user=input("You: ").strip()
    except (EOFError,KeyboardInterrupt):
        print();break

    if user=='/exit':
        break
    elif user=='/reset':
        messages=[]
        print("(History cleared)")
        continue
    elif user.startswith('/save '):
        path=user[6:]
        with open(path,'w') as f:
            for m in messages:
                f.write(f"{m['role']}: {m['content']}\n")
        print(f"(Saved to {path})")
        continue
    elif not user:
        continue

    messages.append({"role":"user","content":user})

    # Format with Qwen's chat template
    text=tok.apply_chat_template(messages,tokenize=False,tools=None)
    inp=tok(text,return_tensors='pt').to(DEVICE)

    with torch.no_grad():
        out=model.generate(
            **inp,
            max_new_tokens=256,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            pad_token_id=tok.eos_token_id,
        )

    response=tok.decode(out[0][inp.input_ids.size(1):],skip_special_tokens=True).strip()
    messages.append({"role":"assistant","content":response})
    print(f"AI:  {response}\n")
