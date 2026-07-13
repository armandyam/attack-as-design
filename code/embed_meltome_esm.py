
import json, numpy as np, torch, time, os
from transformers import AutoTokenizer, EsmModel
seqs_data = json.load(open("meltome_for_embed.json"))
seqs = [r["sequence"] for r in seqs_data]; N=len(seqs)
MODEL="facebook/esm2_t33_650M_UR50D"
tok=AutoTokenizer.from_pretrained(MODEL)
model=EsmModel.from_pretrained(MODEL).cuda().eval().half()
print(f"loaded {MODEL} on {torch.cuda.get_device_name()}", flush=True)
os.makedirs("out", exist_ok=True)
Xm = np.lib.format.open_memmap("out/X_meltome.npy", mode="w+", dtype=np.float32, shape=(N,1280))
bs=16; t0=time.time()
with torch.no_grad():
    for i in range(0,N,bs):
        batch=seqs[i:i+bs]
        enc=tok(batch,return_tensors="pt",padding=True,truncation=True,max_length=1024)
        enc={k:v.cuda() for k,v in enc.items()}
        out=model(**enc).last_hidden_state             # (B,L,1280)
        mask=enc["attention_mask"].unsqueeze(-1).float()
        mean=(out*mask).sum(1)/mask.sum(1)
        Xm[i:i+bs]=mean.float().cpu().numpy()
        if i % 1600==0:
            el=time.time()-t0; print(f"{i}/{N}  {el:.0f}s  eta {el/(i+bs)*(N-i):.0f}s",flush=True)
Xm.flush()
print(f"DONE {N} in {time.time()-t0:.0f}s",flush=True)
