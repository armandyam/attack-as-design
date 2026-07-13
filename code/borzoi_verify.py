
import numpy as np, json, torch, time, os
from borzoi_pytorch import Borzoi
t0=time.time()
SEQLEN=524288; CORE=600
meta=json.load(open("poolA_meta.json"))
scaffold=np.load("poolA_scaffold.npy").astype(np.int64)
c0=meta["c0"]; TRACK=1311
B2I={"A":0,"C":1,"G":2,"T":3}
picks=json.load(open("design_picks.json"))  # {method: [seq,...]}
model=Borzoi.from_pretrained("johahi/borzoi-replicate-0").cuda().eval()
print("model loaded",round(time.time()-t0),flush=True)
def score(seqs):
    out_vals=[]
    BS=2
    for i in range(0,len(seqs),BS):
        b=seqs[i:i+BS]
        oh=np.zeros((len(b),4,SEQLEN),np.float32)
        for k in range(len(b)):
            full=scaffold.copy()
            for j,ch in enumerate(b[k]): full[c0+j]=B2I[ch]
            oh[k]=np.eye(4,dtype=np.float32)[full].T
        with torch.no_grad():
            o=model(torch.from_numpy(oh).cuda())
        Lb=o.shape[-1]; cb=Lb//2; w=8
        out_vals.append(o[:,TRACK,cb-w:cb+w].mean(-1).float().cpu().numpy())
    return np.concatenate(out_vals)
res={}
for m,seqs in picks.items():
    v=score(seqs); res[m]=v.tolist()
    print(m,"true track mean",round(float(v.mean()),4),"max",round(float(v.max()),4),flush=True)
os.makedirs("out",exist_ok=True)
json.dump(res,open("out/design_true_tracks.json","w"))
print("DONE",round(time.time()-t0),flush=True)
