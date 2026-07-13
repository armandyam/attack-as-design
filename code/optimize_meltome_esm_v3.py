
import json, numpy as np, torch, time, os
from transformers import AutoTokenizer, EsmModel, EsmForMaskedLM
MODEL="facebook/esm2_t33_650M_UR50D"; AA=list("ACDEFGHIKLMNPQRSTVWY"); dev="cuda"
tok=AutoTokenizer.from_pretrained(MODEL)
emb=EsmModel.from_pretrained(MODEL).to(dev).eval().half()
mlm=EsmForMaskedLM.from_pretrained(MODEL).to(dev).eval().half()
print("models loaded",flush=True)

S=np.load("meltome_surrogate.npz"); seeds=json.load(open("meltome_seeds.json"))
BB=np.load("blosum62.npz"); BLO=BB["B"]; BORD="".join(BB["order"])
BIDX={a:i for i,a in enumerate(BORD)}
bands=np.load("biophys_bands.npz")

sc_mean=torch.tensor(S["scaler_mean"],device=dev); sc_scale=torch.tensor(S["scaler_scale"],device=dev)
def mlp_forward(Xs,mi):
    n=int(S[f"m{mi}_nlayers"]); h=Xs
    for li in range(n):
        Wl=torch.tensor(S[f"m{mi}_W{li}"],device=dev); bl=torch.tensor(S[f"m{mi}_b{li}"],device=dev)
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

def score(sl,mi=0):
    E=embed(sl); Xs=(E-sc_mean)/sc_scale; return mlp_forward(Xs,mi).cpu().numpy()

# ---- MODEL-INDEPENDENT constraints (pure sequence, no embedding, no ESM-2) ----
KD={'A':1.8,'R':-4.5,'N':-3.5,'D':-3.5,'C':2.5,'Q':-3.5,'E':-3.5,'G':-0.4,'H':-3.2,'I':4.5,
    'L':3.8,'K':-3.9,'M':1.9,'F':2.8,'P':-1.6,'S':-0.8,'T':-0.7,'W':-0.9,'Y':-1.3,'V':4.2}
CHG={'D':-1.,'E':-1.,'K':1.,'R':1.,'H':0.1}; AROM=set("FWY"); CHARGED=set("DEKRH")
def biophys_ok(seq):
    s=[c for c in seq if c in KD]; n=len(s)
    if n==0: return False
    gravy=np.mean([KD[c] for c in s]); charge=sum(CHG.get(c,0) for c in s)/n
    arom=sum(c in AROM for c in s)/n; chg=sum(c in CHARGED for c in s)/n
    cnt=np.array([s.count(a) for a in AA],dtype=float); p=cnt/cnt.sum(); p=p[p>0]
    ent=-(p*np.log2(p)).sum()
    mr=1; cur=1
    for i in range(1,n):
        cur=cur+1 if s[i]==s[i-1] else 1; mr=max(mr,cur)
    for k,v in [("gravy",gravy),("charge",charge),("arom",arom),("chg",chg),("entropy",ent)]:
        if v<float(bands[k+"_lo"]) or v>float(bands[k+"_hi"]): return False
    if mr>float(bands["maxrun_cap"]): return False
    return True

def blosum_edit_ok(orig_aa, new_aa, thresh=-1):
    # accept substitution only if evolutionarily plausible (BLOSUM62 >= thresh)
    if orig_aa not in BIDX or new_aa not in BIDX: return False
    return BLO[BIDX[orig_aa],BIDX[new_aa]]>=thresh

def mutate(seq,nmut,rng,seed_seq=None,blosum=False):
    s=list(seq)
    for _ in range(nmut):
        p=rng.integers(len(s))
        if blosum and seed_seq is not None:
            # sample a BLOSUM-plausible replacement for the ORIGINAL residue at p
            orig=seed_seq[p]; cands=[a for a in AA if a!=s[p] and blosum_edit_ok(orig,a,thresh=-1)]
            if cands: s[p]=cands[rng.integers(len(cands))]
        else:
            s[p]=AA[rng.integers(20)]
    return "".join(s)

def run_ga(seed_seq,mode,gens=15,pop=48,elite=8,mut_per_gen=2,seed=0):
    rng=np.random.default_rng(seed)
    blo = (mode=="blosum")
    population=[seed_seq]+[mutate(seed_seq,mut_per_gen,rng,seed_seq,blo) for _ in range(pop-1)]
    best=None
    for g in range(gens):
        sc_pred=score(population,0); fit=sc_pred.copy()
        if mode in ("biophys","blosum"):
            for i,seq in enumerate(population):
                if not biophys_ok(seq): fit[i]=-1e9
        order=np.argsort(-fit)
        elites=[population[i] for i in order[:elite] if fit[i]>-1e8]
        if not elites: elites=[seed_seq]
        feas=order[0]; best=(population[feas],float(sc_pred[feas]))
        newpop=list(elites)
        while len(newpop)<pop:
            par=elites[rng.integers(len(elites))]
            newpop.append(mutate(par,mut_per_gen,rng,seed_seq,blo))
        population=newpop
    return best

results=[]; t0=time.time()
MODES=["unc","biophys","blosum"]
for si,sd in enumerate(seeds):
    ss=sd["seq"]; row={"idx":sd["idx"],"true_Tm":sd["true_Tm"],"len":sd["len"],"seed_seq":ss}
    for mode in MODES:
        des,pred=run_ga(ss,mode,seed=100+si)
        row[f"{mode}_seq"]=des; row[f"{mode}_pred"]=pred
        row[f"{mode}_hamming"]=int(sum(a!=b for a,b in zip(ss,des)))
    results.append(row)
    if si%4==0: print(f"seed {si}/{len(seeds)} {time.time()-t0:.0f}s",flush=True)

# verify: held-out consensus (models1-4) + ESM-2 PLL (independent validator) + biophys pass flag
alls=[]; tags=[]
for r in results:
    for tg in ["seed"]+MODES:
        alls.append(r[("seed_seq" if tg=="seed" else tg+"_seq")]); tags.append((tg,r["idx"]))
E=embed(alls); Xs=(E-sc_mean)/sc_scale
cons=np.stack([mlp_forward(Xs,mi).cpu().numpy() for mi in [1,2,3,4]]).mean(0)
nat=naturalness(alls)
biok=np.array([biophys_ok(s) for s in alls])
for i,(tg,idx) in enumerate(tags):
    r=next(rr for rr in results if rr["idx"]==idx)
    r[f"{tg}_consensus"]=float(cons[i]); r[f"{tg}_nat"]=float(nat[i]); r[f"{tg}_biok"]=bool(biok[i])
os.makedirs("out",exist_ok=True)
json.dump(results,open("out/meltome_opt_results.json","w"))
print(f"DONE {len(results)} in {time.time()-t0:.0f}s",flush=True)
