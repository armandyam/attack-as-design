
import json, numpy as np, torch, time, re, os
from transformers import T5EncoderModel, T5Tokenizer
dev="cuda"; t0=time.time()
MODEL="Rostlab/prot_t5_xl_half_uniref50-enc"
tok=T5Tokenizer.from_pretrained(MODEL, do_lower_case=False)
model=T5EncoderModel.from_pretrained(MODEL).to(dev).eval().half()
print("ProtT5 loaded",flush=True)

recs=json.load(open("validation_records.json"))
seqs=[r["seq"] for r in recs]

@torch.no_grad()
def embed(sl,bs=4):
    outs=[]
    for i in range(0,len(sl),bs):
        b=sl[i:i+bs]
        # ProtT5: space-separate residues, replace rare AA with X
        proc=[" ".join(re.sub(r"[UZOB]","X",s)) for s in b]
        enc=tok.batch_encode_plus(proc,add_special_tokens=True,padding="longest",return_tensors="pt")
        ids=enc["input_ids"].to(dev); am=enc["attention_mask"].to(dev)
        emb=model(input_ids=ids,attention_mask=am).last_hidden_state  # (B,L,1024)
        for j in range(len(b)):
            L=int(am[j].sum())-1  # drop trailing eos
            outs.append(emb[j,:L].mean(0).float().cpu().numpy())
    return np.stack(outs)

E=embed(seqs)  # (16,1024)
print("embedded",E.shape,flush=True)

H=np.load("temstapro_heads.npz")
def head_forward(x, name):
    # layers 0,2,4 from stored shapes; ReLU after 0 and 2; sigmoid after 4
    h=x
    for li,act in [(0,"relu"),(2,"relu"),(4,"sig")]:
        W=H[f"{name}.{li}.w"]; b=H[f"{name}.{li}.b"]
        h=h@W.T+b
        h=np.maximum(h,0) if act=="relu" else 1/(1+np.exp(-h))
    return float(h.ravel()[0])

THRESH=[40,45,50,55,60,65]
out=[]
for i,r in enumerate(recs):
    probs={}
    for thr in THRESH:
        ps=[head_forward(E[i],f"mean_major_imbal-{thr}_s{s}") for s in [1,2,3,4,5]]
        probs[thr]=float(np.mean(ps))
    # TemStaPro-style predicted Tm: highest threshold with P>=0.5 (else <40)
    passed=[thr for thr in THRESH if probs[thr]>=0.5]
    pred_class = max(passed) if passed else 35
    out.append({"id":r["id"],"idx":r["idx"],"mode":r["mode"],
                "probs":probs,"pred_tm_class":pred_class,
                "prob_sum":float(sum(probs.values()))})  # monotone continuous score
os.makedirs("out",exist_ok=True)
json.dump(out,open("out/temstapro_results.json","w"))
print("DONE",round(time.time()-t0),"s",flush=True)
for o in out: print(f"  {o['id']:16s} pred_class>={o['pred_tm_class']}  probsum={o['prob_sum']:.2f}",flush=True)
