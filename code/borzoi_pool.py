
import numpy as np, json, torch, time, os
from borzoi_pytorch import Borzoi
from borzoi_pytorch.pytorch_borzoi_model import TRACKS_DF

t0=time.time()
SEQLEN=524288; CORE=600
rng=np.random.default_rng(0)
tdf=TRACKS_DF

def desc(i): return " | ".join(str(v) for v in tdf.loc[i].values)

# ---- candidate tracks: DNase / ATAC accessibility (decodable from local composition) ----
# search descriptions for DNASE/ATAC; prefer K562 + a couple others for diversity
cand=[]
for i in tdf.index:
    s=desc(i).upper()
    if ("DNASE" in s or "ATAC" in s):
        cand.append(i)
# rank: K562 first, then GM12878, then any
def rank(i):
    s=desc(i).upper()
    return (0 if "K562" in s else 1 if "GM12878" in s else 2)
cand=sorted(cand,key=rank)[:6]
if not cand:  # fallback: DNASE substring only in assay col
    cand=[i for i in tdf.index if "dnase" in desc(i).lower()][:6]
print("CANDIDATE TRACKS:",cand,flush=True)
for i in cand: print("  ",i,"::",desc(i)[:110],flush=True)

B2I={"A":0,"C":1,"G":2,"T":3}
scaffold=rng.integers(0,4,SEQLEN)
center=SEQLEN//2; c0=center-CORE//2; c1=c0+CORE

# Real accessibility-driving motifs (JASPAR consensuses): SP1(GC-box), CTCF, NRF1, ETS(GABPA), E-box, TATA
MOTIFS=["GGGGCGGGG","CCGCGNGGNGGCAG".replace("N","G"),"GCGCATGCGC","ACCGGAAGT","CACGTG","TATAAA","GGGCGGG","CGCCCCC"]
def randseq(n): return "".join("ACGT"[i] for i in rng.integers(0,4,n))
def make_core(kind, base=None):
    if kind=="random": return randseq(CORE)
    if kind=="motif":
        s=list(randseq(CORE))
        # stack 3-12 real motifs => genuine grammar & accessibility signal separation
        for _ in range(rng.integers(3,13)):
            mo=MOTIFS[rng.integers(0,len(MOTIFS))]
            p=rng.integers(0,CORE-len(mo)); s[p:p+len(mo)]=list(mo)
        return "".join(s)
    if kind=="mut":
        s=list(base)
        for _ in range(rng.integers(1,40)):
            p=rng.integers(0,CORE); s[p]="ACGT"[rng.integers(0,4)]
        return "".join(s)

cores=[]; kinds=[]
for _ in range(700): cores.append(make_core("random")); kinds.append("random")
seeds=[make_core("motif") for _ in range(300)]
for s in seeds: cores.append(s); kinds.append("motif")
for s in seeds[:150]:
    for _ in range(6): cores.append(make_core("mut",s)); kinds.append("mut")
print("POOL",len(cores),flush=True)

model=Borzoi.from_pretrained("johahi/borzoi-replicate-0").cuda().eval()
print("model loaded",round(time.time()-t0,1),flush=True)

def score_batch(bc):
    oh=np.zeros((len(bc),4,SEQLEN),np.float32)
    for k in range(len(bc)):
        full=scaffold.copy()
        for j,ch in enumerate(bc[k]): full[c0+j]=B2I[ch]
        oh[k]=np.eye(4,dtype=np.float32)[full].T
    with torch.no_grad():
        out=model(torch.from_numpy(oh).cuda())  # (B,T,Lb)
    Lb=out.shape[-1]; cb=Lb//2; w=8
    return out[:,:,cb-w:cb+w].mean(-1).float().cpu().numpy()  # (B,T) all tracks central mean -> we slice cand later

# only keep candidate-track columns to save memory
vals=np.zeros((len(cores),len(cand)),np.float32)
BS=2
for i in range(0,len(cores),BS):
    v=score_batch(cores[i:i+BS])       # (b, Ttracks)
    vals[i:i+len(cores[i:i+BS])]=v[:,cand]
    if i%(BS*100)==0: print("scored",i,round(time.time()-t0,1),flush=True)
print("SCORED",vals.shape,flush=True)

os.makedirs("out",exist_ok=True)
np.save("out/pool_cores_idx.npy", np.array([[B2I[ch] for ch in c] for c in cores],np.uint8))
json.dump({"cores":cores,"kinds":kinds}, open("out/pool_cores.json","w"))
np.save("out/pool_true_tracks.npy", vals)   # (n, len(cand))
json.dump({"cand_tracks":cand,"cand_desc":[desc(i) for i in cand],
           "seqlen":SEQLEN,"core":CORE,"c0":int(c0),"c1":int(c1),
           "motifs":MOTIFS,"n":len(cores),"wall_s":round(time.time()-t0,1),"scaffold_seed":0},
          open("out/pool_meta.json","w"))
np.save("out/scaffold.npy", scaffold.astype(np.uint8))
# quick per-track separation report
for j,ti in enumerate(cand):
    y=vals[:,j]; km=np.array(kinds)
    print(f"track {ti}: mean={y.mean():.4f} sd={y.std():.4f} motif_mean={y[km=='motif'].mean():.4f} rand_mean={y[km=='random'].mean():.4f}",flush=True)
print("DONE",round(time.time()-t0,1),flush=True)
