
import json, numpy as np, torch, time, os, sys
import torch.nn as nn
from transformers import AutoTokenizer, EsmModel, EsmForMaskedLM
MODEL="facebook/esm2_t33_650M_UR50D"; AA=list("ACDEFGHIKLMNPQRSTVWY"); dev="cuda"
AAi={a:i for i,a in enumerate(AA)}
TASK=sys.argv[1]  # 'meltome' or 'gfp'
tok=AutoTokenizer.from_pretrained(MODEL)
emb=EsmModel.from_pretrained(MODEL).to(dev).eval().half()
mlm=EsmForMaskedLM.from_pretrained(MODEL).to(dev).eval().half()
print("models loaded",flush=True)

S=np.load("surrogate.npz"); seeds=json.load(open("seeds.json"))
bands=np.load("biophys_bands.npz")
BB=np.load("blosum62.npz"); BLO=BB["B"]; BORD="".join(BB["order"]); BIDX={a:i for i,a in enumerate(BORD)}
sc_mean=torch.tensor(S["scaler_mean"],device=dev); sc_scale=torch.tensor(S["scaler_scale"],device=dev)
SCOREKEY = "true_Tm" if TASK=="meltome" else "true_score"

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
def score_m0(Xs): return mlp_forward(Xs,"m0",S).cpu().numpy()

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

# ---- GFN-AL: trajectory-balance GFlowNet over K length-preserving edits from the seed ----
# State = current sequence; action at each of K steps = (position, new AA). P_B uniform over the
# K! orderings that reach a given edit set -> log P_B = -sum_{t} log(t+1) (fixed-length TB).
# Reward R(x) = exp(m0(E(x))/T) * 1[feasible].  Matched budget: same #surrogate embeds as GA (720/seed).
class Policy(nn.Module):
    def __init__(self, L):
        super().__init__()
        self.L=L
        self.pos=nn.Sequential(nn.Linear(L*20,256),nn.ReLU(),nn.Linear(256,L))
        self.aa =nn.Sequential(nn.Linear(L*20+L,256),nn.ReLU(),nn.Linear(256,20))
        self.logZ=nn.Parameter(torch.zeros(1))
    def onehot(self,seqs):
        X=torch.zeros(len(seqs),self.L,20,device=dev)
        for i,s in enumerate(seqs):
            for j,c in enumerate(s):
                if c in AAi: X[i,j,AAi[c]]=1.0
        return X.view(len(seqs),-1)
    def forward_pos(self,seqs):
        return torch.log_softmax(self.pos(self.onehot(seqs)),-1)
    def forward_aa(self,seqs,pos):
        oh=self.onehot(seqs); ph=torch.zeros(len(seqs),self.L,device=dev)
        ph[torch.arange(len(seqs)),pos]=1.0
        return torch.log_softmax(self.aa(torch.cat([oh,ph],-1)),-1)

def gfn_optimize(seed_seq,seed=0,K=12,rounds=15,batch=48,T=1.5,lr=3e-3):
    L=len(seed_seq); rng=np.random.default_rng(seed); torch.manual_seed(seed)
    pol=Policy(L).to(dev); opt=torch.optim.Adam(pol.parameters(),lr=lr)
    best_seq=seed_seq; best_score=-1e30
    for rd in range(rounds):
        # sample a batch of K-edit trajectories
        seqs=[list(seed_seq) for _ in range(batch)]
        logpf=torch.zeros(batch,device=dev)
        for t in range(K):
            cur=["".join(s) for s in seqs]
            lp_pos=pol.forward_pos(cur)
            pos=torch.multinomial(lp_pos.exp(),1).squeeze(1)
            logpf=logpf+lp_pos[torch.arange(batch),pos]
            lp_aa=pol.forward_aa(cur,pos)
            aa=torch.multinomial(lp_aa.exp(),1).squeeze(1)
            logpf=logpf+lp_aa[torch.arange(batch),aa]
            for i in range(batch): seqs[i][int(pos[i])]=AA[int(aa[i])]
        final=["".join(s) for s in seqs]
        Xs=emb_std(final); raw=score_m0(Xs)
        feas=np.array([biophys_ok(s) for s in final])
        # reward: exp(score/T), zeroed if infeasible (floor to tiny)
        logR=torch.tensor(raw/T,device=dev,dtype=torch.float32)
        logR[~torch.tensor(feas,device=dev)]=-30.0
        # backward: uniform over orderings -> constant, folded into logZ; TB loss
        logpb=torch.tensor(-sum(np.log(np.arange(1,K+1))),device=dev,dtype=torch.float32)
        loss=((pol.logZ+logpf-logpb-logR)**2).mean()
        opt.zero_grad(); loss.backward(); opt.step()
        # track best feasible
        fi=np.where(feas)[0]
        if len(fi)>0:
            bj=fi[int(np.argmax(raw[fi]))]
            if raw[bj]>best_score: best_score=float(raw[bj]); best_seq=final[bj]
    return best_seq,best_score

results=[]; t0=time.time()
for si,sd in enumerate(seeds):
    ss=sd["seq"]
    des,pred=gfn_optimize(ss,seed=200+si)
    results.append({"idx":sd["idx"],"true_score":sd[SCOREKEY],"len":sd["len"],"seed_seq":ss,
                    "gfnal_seq":des,"gfnal_pred":pred,"gfnal_hamming":int(sum(a!=b for a,b in zip(ss,des)))})
    if si%5==0: print(f"seed {si}/{len(seeds)} {time.time()-t0:.0f}s",flush=True)

# verification: held-out ensemble m1-m4 (never queried), PLL, biophys
alls=[]; tags=[]
for r in results:
    for tg in ["seed","gfnal"]:
        alls.append(r["seed_seq" if tg=="seed" else "gfnal_seq"]); tags.append((tg,r["idx"]))
E=embed(alls); Xs=(E-sc_mean)/sc_scale
cons=np.stack([mlp_forward(Xs,f"m{mi}",S).cpu().numpy() for mi in [1,2,3,4]]).mean(0)
nat=naturalness(alls); biok=np.array([biophys_ok(s) for s in alls])
for i,(tg,idx) in enumerate(tags):
    r=next(rr for rr in results if rr["idx"]==idx)
    r[f"{tg}_consensus"]=float(cons[i]); r[f"{tg}_nat"]=float(nat[i]); r[f"{tg}_biok"]=bool(biok[i])
os.makedirs("out",exist_ok=True); json.dump(results,open(f"out/gfnal_{TASK}.json","w"))
print(f"DONE {TASK} {len(results)} seeds in {time.time()-t0:.0f}s",flush=True)
