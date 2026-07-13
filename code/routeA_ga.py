
import os, json, time, numpy as np, torch
# ---------- Evo2 patch chain ----------
_orig_load=torch.load
def _pl(*a,**k): k["weights_only"]=False; return _orig_load(*a,**k)
torch.load=_pl
import transformer_engine.pytorch as te
_of=te.fp8_autocast
def _nf(*a,**k): k["enabled"]=False; return _of(*a,**k)
te.fp8_autocast=_nf
import flash_attn_2_cuda as fa
def _mk(orig):
    def _s(*args,**kw):
        if len(args)==13: args=args[:10]+args[11:]
        r=orig(*args,**kw)
        if isinstance(r,(list,tuple)) and len(r)>4: r=[r[0],r[-3],r[-2],r[-1]]
        return r
    return _s
fa.fwd=_mk(fa.fwd)
if hasattr(fa,"varlen_fwd"): fa.varlen_fwd=_mk(fa.varlen_fwd)
from evo2 import Evo2
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

t0=time.time()
cfg=json.load(open("ga_config.json")); Z=np.load("ga_inputs.npz")
y=Z["y"]; tr=Z["tr"]; te=Z["te"]
seed_seqs=cfg["seed_seqs"]; band=cfg["band"]; MOTIFS=cfg["motifs"]
pool_cores=json.load(open("poolA_cores.json"))["cores"]
L=len(seed_seqs[0]); I2B="ACGT"; B2I={"A":0,"C":1,"G":2,"T":3}

def gc(s): return sum(c in "GC" for c in s)/len(s)
def maxrun(s):
    m=c=1
    for k in range(1,len(s)): c=c+1 if s[k]==s[k-1] else 1; m=max(m,c)
    return m
def kent(s,k=3):
    import math; cnt={}
    for p in range(len(s)-k+1): cnt[s[p:p+k]]=cnt.get(s[p:p+k],0)+1
    tot=sum(cnt.values()); return -sum((v/tot)*math.log2(v/tot) for v in cnt.values())
def feasible(s):
    return (band["gc"][0]<=gc(s)<=band["gc"][1]) and maxrun(s)<=band["maxrun"] and kent(s)>=band["entropy_min"]

def kmer4_batch(seqs):
    out=np.zeros((len(seqs),256),np.float32)
    for i,s in enumerate(seqs):
        a=[B2I[c] for c in s]
        for p in range(len(a)-3): out[i,a[p]*64+a[p+1]*16+a[p+2]*4+a[p+3]]+=1
    return out

model=Evo2("evo2_7b"); tok=model.tokenizer; LAYER="blocks.28.mlp.l3"
print("evo2 loaded",round(time.time()-t0),flush=True)
@torch.no_grad()
def embed(seqs,bs=8):
    rows=[]
    for i in range(0,len(seqs),bs):
        ids=torch.tensor([tok.tokenize(s) for s in seqs[i:i+bs]],dtype=torch.long,device="cuda")
        _,emb=model(ids,return_embeddings=True,layer_names=[LAYER])
        rows.append(emb[LAYER].mean(1).float().cpu().numpy())
    return np.concatenate(rows,0)

tr_seqs=[pool_cores[i] for i in tr]
Xtr_fm=embed(tr_seqs); sc_fm=StandardScaler().fit(Xtr_fm); fm_head=Ridge(alpha=100.0).fit(sc_fm.transform(Xtr_fm),y[tr])
Xtr_k4=kmer4_batch(tr_seqs); sc_k4=StandardScaler().fit(Xtr_k4); k4_head=Ridge(alpha=10.0).fit(sc_k4.transform(Xtr_k4),y[tr])
print("surrogates trained",round(time.time()-t0),flush=True)
def score_fm(seqs): return fm_head.predict(sc_fm.transform(embed(seqs)))
def score_k4(seqs): return k4_head.predict(sc_k4.transform(kmer4_batch(seqs)))

POP=48; GEN=12; ELITE=8; rng=np.random.default_rng(123)
def mutate(s,k):
    s=list(s)
    for _ in range(k): p=rng.integers(L); s[p]=I2B[rng.integers(4)]
    return "".join(s)
def insert_motif(s):
    s=list(s); mo=MOTIFS[rng.integers(len(MOTIFS))]; p=rng.integers(L-len(mo)); s[p:p+len(mo)]=list(mo); return "".join(s)
def propose(parent):
    r=rng.random()
    if r<0.6: return mutate(parent,int(rng.integers(1,12)))
    if r<0.85: return insert_motif(mutate(parent,int(rng.integers(1,6))))
    return mutate(insert_motif(parent),int(rng.integers(1,5)))

def run_ga(seed_seq, score_fn, constrained):
    # init population (constrained init if constrained)
    pop=[seed_seq]
    while len(pop)<POP:
        c=propose(seed_seq)
        if (not constrained) or feasible(c): pop.append(c)
    best_seq=seed_seq; best_pred=-1e30; traj=[]
    for g in range(GEN):
        pred=score_fn(pop)
        feas=np.array([feasible(s) for s in pop]) if constrained else np.ones(len(pop),bool)
        fit=pred.copy(); fit[~feas]=-1e30
        if feas.any():
            fi=np.where(feas)[0]; bj=fi[int(np.argmax(pred[fi]))]
            if pred[bj]>best_pred: best_pred=float(pred[bj]); best_seq=pop[bj]
        order=np.argsort(-fit); elites=[pop[i] for i in order[:ELITE] if fit[i]>-1e29] or [seed_seq]
        newpop=list(elites)
        while len(newpop)<POP:
            par=elites[rng.integers(len(elites))]; c=propose(par)
            if (not constrained) or feasible(c): newpop.append(c)
        pop=newpop
        traj.append({"gen":g,"best_pred":float(pred[order[0]]),"mean_pred":float(pred.mean())})
    return best_seq,best_pred,traj

ARMS=[("fm_unc",score_fm,False),("fm_con",score_fm,True),
      ("oh_unc",score_k4,False),("oh_con",score_k4,True)]
designs={"seed":[]}; trajs={a[0]:[] for a in ARMS}
for a in ARMS: designs[a[0]]=[]
for si,ss in enumerate(seed_seqs):
    designs["seed"].append(ss)
    for name,fn,con in ARMS:
        bs,bp,tj=run_ga(ss,fn,con)
        designs[name].append(bs); trajs[name].append({"seed_idx":si,"best_pred":bp,"traj":tj})
    if si%5==0: print(f"seed {si}/{len(seed_seqs)} {round(time.time()-t0)}s",flush=True)

os.makedirs("out",exist_ok=True)
json.dump(designs,open("out/ga_final_designs.json","w"))
json.dump(trajs,open("out/ga_trajectories.json","w"))
json.dump({"pop":POP,"gen":GEN,"elite":ELITE,"n_seeds":len(seed_seqs),
           "arms":[a[0] for a in ARMS],"wall_s":round(time.time()-t0)},open("out/ga_meta.json","w"))
print("DONE",round(time.time()-t0),flush=True)
