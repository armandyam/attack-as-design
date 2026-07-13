
import os, numpy as np, torch, pandas as pd, json, time
os.environ["HF_HOME"]=os.path.abspath("hf_cache"); os.environ["HF_HUB_DISABLE_XET"]="1"; os.environ["TOKENIZERS_PARALLELISM"]="false"
from transformers import AutoTokenizer, AutoModel
seqs = json.load(open("data/promoter_for_embed.json"))["seqs"]; N=len(seqs)
tok=AutoTokenizer.from_pretrained("zhihan1996/DNABERT-2-117M",trust_remote_code=True)
model=AutoModel.from_pretrained("zhihan1996/DNABERT-2-117M",trust_remote_code=True); model.eval()
# write straight to disk memmap to avoid RAM blowup
Xm = np.lib.format.open_memmap("data/X_promoter.npy", mode="w+", dtype=np.float32, shape=(N,768))
bs=64; t0=time.time()
with torch.no_grad():
    for i in range(0,N,bs):
        enc=tok(seqs[i:i+bs],return_tensors="pt",padding=True,truncation=True,max_length=128)
        out=model(enc["input_ids"],attention_mask=enc["attention_mask"])[0]
        mask=enc["attention_mask"].unsqueeze(-1).float()
        mean=(out*mask).sum(1)/mask.sum(1)   # mean-pool over real tokens
        Xm[i:i+bs]=mean.cpu().numpy()
        if i % 3200==0: print(f"{i}/{N}  {(time.time()-t0):.0f}s",flush=True)
Xm.flush()
print("DONE embed", N, f"{(time.time()-t0):.0f}s",flush=True)
