
import json, numpy as np, torch, time, os
from transformers import AutoTokenizer, EsmModel, EsmForMaskedLM
MODEL="facebook/esm2_t33_650M_UR50D"; AA=list("ACDEFGHIKLMNPQRSTVWY"); dev="cuda"
tok=AutoTokenizer.from_pretrained(MODEL)
emb=EsmModel.from_pretrained(MODEL).to(dev).eval().half()
mlm=EsmForMaskedLM.from_pretrained(MODEL).to(dev).eval().half()
print("models loaded",flush=True)

S=np.load("meltome_surrogate.npz"); seeds=json.load(open("meltome_seeds.json"))
H=np.load("baseline_heads.npz")
BB=np.load("blosum62.npz"); BLO=BB["B"]; BORD="".join(BB["order"]); BIDX={a:i for i,a in enumerate(BORD)}
bands=np.load("biophys_bands.npz")
sc_mean=torch.tensor(S["scaler_mean"],device=dev); sc_scale=torch.tensor(S["scaler_scale"],device=dev)

def mlp_forward(Xs,pfx,src):
    n=int(src[f"{pfx}_nlayers"]); h=Xs
    for li in range(n):
        Wl=torch.tensor(src[f"{pfx}_W{li}"],device=dev,dtype=torch.float32); bl=torch.tensor(src[f"{pfx}_b{li}"],device=dev,dtype=torch.float32)
        h=h@Wl+bl
        if li<n-1: h=torch.relu(h)
    return h.squeeze(-1)

@torch.no_grad()
def embed(sl,bs=48):
    outs=[]
    for i in range(0,len(sl),bs):
        enc=tok(sl[i:i+bs],return_tensors="pt",padding=True,truncation=True,max_length=1024)
        enc={k:v.to(dev) for k,v in enc.items()}
        o=emb(**enc).last_hidden_state; m=enc["attention_mask"].unsqueeze(-1).half()
        outs.append(((o*m).sum(1)/m.sum(1)).float())
    return torch.cat(outs,0)

@torch.no_grad()
def naturalness(sl,bs=48):
    vals=[]
    for i in range(0,len(sl),bs):
        b=sl[i:i+bs]; enc=tok(b,return_tensors="pt",padding=True,truncation=True,max_length=1024)
        enc={k:v.to(dev) for k,v in enc.items()}
        lp=torch.log_softmax(mlm(**enc).logits.float(),-1)
        ids=enc["input_ids"]; mask=enc["attention_mask"].bool()
        tlp=lp.gather(-1,ids.unsqueeze(-1)).squeeze(-1)
        for r in range(len(b)):
            mm=mask[r].clone(); mm[0]=False; idx=torch.where(mm)[0]
            if len(idx)>0: idx=idx[:-1]
            vals.append(float(tlp[r,idx].mean().cpu()))
    return np.array(vals)

def emb_std(sl):
    E=embed(sl); return (E-sc_mean)/sc_scale

# scoring functions per method (all take standardized embeddings Xs)
def score_m0(Xs): return mlp_forward(Xs,"m0",S).cpu().numpy()
def score_coms(Xs): return mlp_forward(Xs,"coms",H).cpu().numpy()
def critic_val(Xs): return mlp_forward(Xs,"critic",H).cpu().numpy()  # higher = more in-distribution

# GABO: adaptive lambda combines m0 score with source-critic in-distribution term
# f_gabo = (1-lam)*m0 - lam*(-critic)  ... penalize low critic (off-distribution). lam adaptive.
def score_gabo(Xs, lam):
    m0=mlp_forward(Xs,"m0",S).cpu().numpy(); c=mlp_forward(Xs,"critic",H).cpu().numpy()
    return (1-lam)*m0 + lam*c*0.1  # scale critic into Tm-comparable range

KD={'A':1.8,'R':-4.5,'N':-3.5,'D':-3.5,'C':2.5,'Q':-3.5,'E':-3.5,'G':-0.4,'H':-3.2,'I':4.5,
    'L':3.8,'K':-3.9,'M':1.9,'F':2.8,'P':-1.6,'S':-0.8,'T':-0.7,'W':-0.9,'Y':-1.3,'V':4.2}
CHG={'D':-1.,'E':-1.,'K':1.,'R':1.,'H':0.1}; AROM=set("FWY"); CHARGED=set("DEKRH")
def biophys_ok(seq):
    s=[c for c in seq if c in KD]; n=len(s)
    if n==0: return False
    gravy=np.mean([KD[c] for c in s]); charge=sum(CHG.get(c,0) for c in s)/n
    arom=sum(c in AROM for c in s)/n; chg=sum(c in CHARGED for c in s)/n
    cnt=np.array([s.count(a) for a in AA],dtype=float); p=cnt/cnt.sum(); p=p[p>0]; ent=-(p*np.log2(p)).sum()
    mr=1; cur=1
    for i in range(1,n):
        cur=cur+1 if s[i]==s[i-1] else 1; mr=max(mr,cur)
    for k,v in [("gravy",gravy),("charge",charge),("arom",arom),("chg",chg),("entropy",ent)]:
        if v<float(bands[k+"_lo"]) or v>float(bands[k+"_hi"]): return False
    if mr>float(bands["maxrun_cap"]): return False
    return True
def blosum_ok(o,nw,th=-1):
    if o not in BIDX or nw not in BIDX: return False
    return BLO[BIDX[o],BIDX[nw]]>=th
def mutate(seq,nmut,rng,seed_seq=None,blosum=False):
    s=list(seq)
    for _ in range(nmut):
        p=rng.integers(len(s))
        if blosum and seed_seq is not None:
            orig=seed_seq[p]; cands=[a for a in AA if a!=s[p] and blosum_ok(orig,a)]
            if cands: s[p]=cands[rng.integers(len(cands))]
        else: s[p]=AA[rng.integers(20)]
    return "".join(s)

def run_ga(seed_seq,method,gens=15,pop=48,elite=8,mut_per_gen=2,seed=0):
    rng=np.random.default_rng(seed); blo=(method=="ours_blosum")
    population=[seed_seq]+[mutate(seed_seq,mut_per_gen,rng,seed_seq,blo) for _ in range(pop-1)]
    lam=0.0
    # track best FEASIBLE design seen across ALL generations (never return a sentinel)
    best_seq=seed_seq; best_score=-1e30
    for g in range(gens):
        Xs=emb_std(population)
        if method=="unc": raw=score_m0(Xs)
        elif method=="coms": raw=score_coms(Xs)
        elif method=="gabo":
            raw=score_gabo(Xs,lam)
            cv=critic_val(Xs); lam=min(0.8, max(0.0, ((-cv.mean())-5.0)/25.0))
        elif method=="ours_blosum": raw=score_m0(Xs)
        # feasibility mask (only ours_blosum constrains; others always feasible)
        feasible=np.ones(len(population),dtype=bool)
        if method=="ours_blosum":
            feasible=np.array([biophys_ok(s) for s in population])
        fit=raw.copy()
        fit[~feasible]=-1e9
        # update global best among FEASIBLE members this generation
        if feasible.any():
            fi=np.where(feasible)[0]; bj=fi[int(np.argmax(raw[fi]))]
            if raw[bj]>best_score: best_score=float(raw[bj]); best_seq=population[bj]
        order=np.argsort(-fit); elites=[population[i] for i in order[:elite] if fit[i]>-1e8]
        if not elites: elites=[seed_seq]  # reseed from feasible seed if wiped out
        newpop=list(elites)
        while len(newpop)<pop:
            par=elites[rng.integers(len(elites))]; newpop.append(mutate(par,mut_per_gen,rng,seed_seq,blo))
        population=newpop
    # best_seq is always a real feasible sequence (falls back to seed only if NO feasible ever found)
    return best_seq, best_score

METHODS=["unc","coms","gabo","ours_blosum"]
results=[]; t0=time.time()
for si,sd in enumerate(seeds):
    ss=sd["seq"]; row={"idx":sd["idx"],"true_Tm":sd["true_Tm"],"len":sd["len"],"seed_seq":ss}
    for mth in METHODS:
        des,pred=run_ga(ss,mth,seed=100+si)
        row[f"{mth}_seq"]=des; row[f"{mth}_pred"]=pred
        row[f"{mth}_hamming"]=int(sum(a!=b for a,b in zip(ss,des)))
    results.append(row)
    if si%4==0: print(f"seed {si}/{len(seeds)} {time.time()-t0:.0f}s",flush=True)

# unified verification: held-out ensemble consensus (m1-m4, NEVER used by any optimizer), ESM-2 PLL, biophys
alls=[]; tags=[]
for r in results:
    for tg in ["seed"]+METHODS:
        alls.append(r["seed_seq" if tg=="seed" else tg+"_seq"]); tags.append((tg,r["idx"]))
E=embed(alls); Xs=(E-sc_mean)/sc_scale
cons=np.stack([mlp_forward(Xs,f"m{mi}",S).cpu().numpy() for mi in [1,2,3,4]]).mean(0)
nat=naturalness(alls); biok=np.array([biophys_ok(s) for s in alls])
for i,(tg,idx) in enumerate(tags):
    r=next(rr for rr in results if rr["idx"]==idx)
    r[f"{tg}_consensus"]=float(cons[i]); r[f"{tg}_nat"]=float(nat[i]); r[f"{tg}_biok"]=bool(biok[i])
os.makedirs("out",exist_ok=True); json.dump(results,open("out/meltome_headtohead.json","w"))
print(f"DONE {len(results)} seeds x {len(METHODS)} methods in {time.time()-t0:.0f}s",flush=True)
