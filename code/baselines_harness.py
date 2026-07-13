"""TF-Bind-8 (SIX6) offline-MBO baseline harness.
Budget = total surrogate queries (repeats counted, FLEXS/Design-Bench convention).
Surrogate = DNABERT-2 MLP trained on bottom-60% by true affinity (extrapolation task).
Oracle    = exact normalized affinity over all 65,536 8-mers (data/_Y.npy).
Explorers : random, adalead, cmaes(separable), cbas/dbas, gradient(hillclimb), ga(+/-constraint).
Constraint= model-independent DNA plausibility: GC band + max homopolymer run from observable set.
Run: python baselines_run.py  (loops methods x seeds, writes data/baselines_results.json)
"""
import numpy as np, json
from itertools import product
MERS=["".join(p) for p in product("ACGT",repeat=8)]; M2I={m:i for i,m in enumerate(MERS)}
BASES="ACGT"; B2I={b:i for i,b in enumerate(BASES)}
POW=np.array([4**i for i in range(7,-1,-1)])
def seqs_to_idx(seqs): return np.array([[B2I[c] for c in s] for s in seqs])@POW
Y=np.load("data/_Y.npy"); SURR=np.load("data/_SURR_trunc.npy")
tau=np.quantile(Y,0.60); obs=np.where(Y<=tau)[0]

class Budget:
    def __init__(self,cap): self.cap=cap; self.n=0
    def query(self,seqs): idx=seqs_to_idx(seqs); self.n+=len(idx); return SURR[idx]
    def done(self): return self.n>=self.cap
def mutate_batch(parents,rng,kmax=2):
    out=[]
    for par in parents:
        s=list(par); k=rng.integers(1,kmax+1)
        for _ in range(k): p=rng.integers(8); s[p]=BASES[rng.integers(4)]
        out.append("".join(s))
    return out
def rand_seqs(n,rng): return ["".join(BASES[j] for j in row) for row in rng.integers(0,4,(n,8))]
def true_topk(seqs,sc,k=16):
    order=np.argsort(-sc)[:k]; top=[seqs[i] for i in order]; idx=seqs_to_idx(top)
    return float(Y[idx].mean()), float(Y[idx].max()), top

def gc(s): return (s.count("G")+s.count("C"))/8
def maxrun(s):
    m=c=1
    for i in range(1,8): c=c+1 if s[i]==s[i-1] else 1; m=max(m,c)
    return m
_obs=[MERS[i] for i in obs]; _gc=np.array([gc(s) for s in _obs])
GC_LO,GC_HI=np.quantile(_gc,[0.01,0.99]); RUN_CAP=int(np.quantile([maxrun(s) for s in _obs],0.99))
def dna_plausible(s): return GC_LO<=gc(s)<=GC_HI and maxrun(s)<=RUN_CAP

def ex_random(bud,rng):
    seen={}
    while not bud.done():
        ss=rand_seqs(256,rng); sc=bud.query(ss)
        for s,v in zip(ss,sc): seen[s]=v
    seqs=list(seen); return seqs,np.array([seen[s] for s in seqs])
def ex_adalead(bud,rng,start,thr_frac=0.75,batch=256):
    seen={}; pool=list(start); sc0=bud.query(pool)
    for s,v in zip(pool,sc0): seen[s]=v
    while not bud.done():
        pv=np.array([seen[p] for p in pool]); thr=np.quantile(pv,thr_frac)
        parents=[p for p,v in zip(pool,pv) if v>=thr] or pool
        kids=mutate_batch([parents[rng.integers(len(parents))] for _ in range(batch)],rng); sc=bud.query(kids)
        for s,v in zip(kids,sc): seen[s]=v
        allc=list(set(pool+kids)); av=np.array([seen[c] for c in allc]); pool=[allc[i] for i in np.argsort(-av)[:64]]
    seqs=list(seen); return seqs,np.array([seen[s] for s in seqs])
def ex_cmaes(bud,rng,lam=64):
    dim=32; mean=rng.normal(0,1,dim); std=np.ones(dim); seen={}
    while not bud.done():
        pop=mean+std*rng.standard_normal((lam,dim))
        seqs=["".join(BASES[j] for j in z.reshape(8,4).argmax(1)) for z in pop]; sc=bud.query(seqs)
        for s,v in zip(seqs,sc): seen[s]=v
        order=np.argsort(-sc)[:max(2,lam//2)]; elites=pop[order]; mean=elites.mean(0); std=0.5*std+0.5*(elites.std(0)+1e-3)
    seqs=list(seen); return seqs,np.array([seen[s] for s in seqs])
def ex_cbas(bud,rng,q=0.8,per=256):
    P=np.ones((8,4))/4; seen={}
    while not bud.done():
        arr=np.array([[rng.choice(4,p=P[j]) for j in range(8)] for _ in range(per)])
        seqs=["".join(BASES[j] for j in row) for row in arr]; sc=bud.query(seqs)
        for s,v in zip(seqs,sc): seen[s]=v
        thr=np.quantile(sc,q); keep=arr[sc>=thr]
        if len(keep):
            newP=np.ones((8,4))*0.1
            for row in keep:
                for j,b in enumerate(row): newP[j,b]+=1
            P=newP/newP.sum(1,keepdims=True)
    seqs=list(seen); return seqs,np.array([seen[s] for s in seqs])
def ex_gradient(bud,rng):
    seen={}; cur=rand_seqs(1,rng)[0]; curv=bud.query([cur])[0]; seen[cur]=curv
    while not bud.done():
        neigh=[cur[:p]+b+cur[p+1:] for p in range(8) for b in BASES if b!=cur[p]]; sc=bud.query(neigh)
        for s,v in zip(neigh,sc): seen[s]=v
        bi=int(np.argmax(sc))
        if sc[bi]<=curv: cur=rand_seqs(1,rng)[0]; curv=bud.query([cur])[0]; seen[cur]=curv
        else: cur=neigh[bi]; curv=sc[bi]
    seqs=list(seen); return seqs,np.array([seen[s] for s in seqs])
def ex_ga(bud,rng,start,constrained=False,pop=128,elite=16,immig=0.25):
    seen={}; population=list(start[:pop])
    while len(population)<pop: population.append(mutate_batch([start[rng.integers(len(start))]],rng)[0])
    sc0=bud.query(population)
    for s,v in zip(population,sc0): seen[s]=v
    while not bud.done():
        sv=np.array([seen.get(p,-1) for p in population]); fit=sv.copy()
        if constrained: fit=np.array([sv[i] if dna_plausible(population[i]) else -1e9 for i in range(len(population))])
        order=np.argsort(-fit); elites=[population[i] for i in order[:elite] if fit[i]>-1e8] or [population[order[0]]]
        n_imm=int(pop*immig)
        children=mutate_batch([elites[rng.integers(len(elites))] for _ in range(pop-len(elites)-n_imm)],rng)
        newpop=list(elites)+children+rand_seqs(n_imm,rng); fresh=[s for s in newpop if s not in seen]
        if fresh:
            sc=bud.query(fresh)
            for s,v in zip(fresh,sc): seen[s]=v
        population=newpop
    seqs=list(seen); return seqs,np.array([seen[s] for s in seqs])
