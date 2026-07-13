
import json, numpy as np, torch, time, os
from transformers import AutoTokenizer, EsmModel, EsmForMaskedLM

MODEL="facebook/esm2_t33_650M_UR50D"
AA=list("ACDEFGHIKLMNPQRSTVWY")
dev="cuda"
tok=AutoTokenizer.from_pretrained(MODEL)
emb=EsmModel.from_pretrained(MODEL).to(dev).eval().half()
mlm=EsmForMaskedLM.from_pretrained(MODEL).to(dev).eval().half()
print("models loaded",flush=True)

S=np.load("meltome_surrogate.npz")
seeds=json.load(open("meltome_seeds.json"))

sc_mean=torch.tensor(S["scaler_mean"],device=dev); sc_scale=torch.tensor(S["scaler_scale"],device=dev)
pca_mean=torch.tensor(S["pca_mean"],device=dev)
pca_comp=torch.tensor(S["pca_components"],device=dev)          # (50,1280)
pca_evs =torch.tensor(S["pca_ev_sqrt"],device=dev)             # (50,)
centroid=torch.tensor(S["centroid"],device=dev)
tau=float(S["tau"])

def mlp_forward(Xs, mi):
    n=int(S[f"m{mi}_nlayers"]); h=Xs
    for li in range(n):
        Wl=torch.tensor(S[f"m{mi}_W{li}"],device=dev); bl=torch.tensor(S[f"m{mi}_b{li}"],device=dev)
        h=h@Wl+bl
        if li<n-1: h=torch.relu(h)
    return h.squeeze(-1)

@torch.no_grad()
def embed(seqlist, bs=48):
    outs=[]
    for i in range(0,len(seqlist),bs):
        b=seqlist[i:i+bs]
        enc=tok(b,return_tensors="pt",padding=True,truncation=True,max_length=1024)
        enc={k:v.to(dev) for k,v in enc.items()}
        o=emb(**enc).last_hidden_state
        m=enc["attention_mask"].unsqueeze(-1).half()
        outs.append(((o*m).sum(1)/m.sum(1)).float())
    return torch.cat(outs,0)

@torch.no_grad()
def naturalness(seqlist, bs=48):
    # single-pass pseudo-log-likelihood: mean log-softmax at each position for observed AA
    vals=[]
    for i in range(0,len(seqlist),bs):
        b=seqlist[i:i+bs]
        enc=tok(b,return_tensors="pt",padding=True,truncation=True,max_length=1024)
        enc={k:v.to(dev) for k,v in enc.items()}
        logits=mlm(**enc).logits.float()
        lp=torch.log_softmax(logits,-1)
        ids=enc["input_ids"]; mask=enc["attention_mask"].bool()
        tok_lp=lp.gather(-1,ids.unsqueeze(-1)).squeeze(-1)
        for r in range(len(b)):
            mm=mask[r]; mm[0]=False  # drop CLS
            idx=torch.where(mm)[0]; idx=idx[:-1] if len(idx)>0 else idx  # drop EOS
            vals.append(float(tok_lp[r,idx].mean().cpu()))
    return np.array(vals)

def score(seqlist, mi=0):
    E=embed(seqlist); Xs=(E-sc_mean)/sc_scale
    return mlp_forward(Xs,mi).cpu().numpy(), Xs

def constraint_g(Xs):
    W=((Xs-pca_mean)@pca_comp.T)/pca_evs   # whitened (n,50)
    return (-torch.sqrt(((W-centroid)**2).sum(1))).cpu().numpy()

def mutate(seq, nmut, rng):
    s=list(seq)
    for _ in range(nmut):
        p=rng.integers(len(s)); s[p]=AA[rng.integers(20)]
    return "".join(s)

def run_ga(seed_seq, constrained, gens=15, pop=48, elite=8, mut_per_gen=2, seed=0):
    rng=np.random.default_rng(seed)
    population=[seed_seq]+[mutate(seed_seq, mut_per_gen, rng) for _ in range(pop-1)]
    best=None
    for g in range(gens):
        sc_pred,Xs=score(population,mi=0)
        gg=constraint_g(Xs)
        fit=sc_pred.copy()
        if constrained: fit[gg<tau]=-1e9
        order=np.argsort(-fit)
        elites=[population[i] for i in order[:elite] if fit[order[0]]>-1e8 or not constrained]
        if not elites: elites=[seed_seq]
        # record best feasible
        feas=order[0]
        best=(population[feas], float(sc_pred[feas]), float(gg[feas]))
        # next gen: mutate elites
        newpop=list(elites)
        while len(newpop)<pop:
            par=elites[rng.integers(len(elites))]
            newpop.append(mutate(par, mut_per_gen, rng))
        population=newpop
    return best

results=[]
t0=time.time()
for si,sd in enumerate(seeds):
    seed_seq=sd["seq"]
    row={"idx":sd["idx"],"true_Tm":sd["true_Tm"],"len":sd["len"],"seed_seq":seed_seq}
    for mode,con in [("unc",False),("con",True)]:
        des,pred,gval=run_ga(seed_seq,con,seed=100+si)
        row[f"{mode}_seq"]=des; row[f"{mode}_pred"]=pred; row[f"{mode}_g"]=gval
        # hamming from seed
        row[f"{mode}_hamming"]=int(sum(a!=b for a,b in zip(seed_seq,des)))
    results.append(row)
    if si%4==0: print(f"seed {si}/{len(seeds)}  {time.time()-t0:.0f}s",flush=True)

# batch-verify all designs: held-out ensemble consensus + naturalness
all_seqs=[]; tags=[]
for r in results:
    all_seqs+=[r["seed_seq"],r["unc_seq"],r["con_seq"]]; tags+=[("seed",r["idx"]),("unc",r["idx"]),("con",r["idx"])]
E=embed(all_seqs); Xs=(E-sc_mean)/sc_scale
cons=np.stack([mlp_forward(Xs,mi).cpu().numpy() for mi in [1,2,3,4]]).mean(0)  # held-out consensus
nat=naturalness(all_seqs)
for i,(tg,idx) in enumerate(tags):
    r=next(rr for rr in results if rr["idx"]==idx)
    r[f"{tg}_consensus"]=float(cons[i]); r[f"{tg}_nat"]=float(nat[i])

os.makedirs("out",exist_ok=True)
json.dump(results, open("out/meltome_opt_results.json","w"))
print(f"DONE {len(results)} seeds in {time.time()-t0:.0f}s",flush=True)
