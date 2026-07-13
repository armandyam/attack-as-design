
import os, json, numpy as np, time
os.environ.setdefault("HF_HUB_DISABLE_XET","1")
import torch
_orig_load = torch.load
def _patched_load(*a, **k):
    k["weights_only"]=False; return _orig_load(*a, **k)
torch.load = _patched_load
try:
    import transformer_engine.pytorch as te
    _orig_fp8 = te.fp8_autocast
    def _no_fp8(*a, **k):
        k["enabled"]=False; return _orig_fp8(*a, **k)
    te.fp8_autocast = _no_fp8
except Exception as e:
    print("te patch skipped:", e, flush=True)
from evo2 import Evo2
try:
    import flash_attn_2_cuda as fa
    _orig_fwd = fa.fwd
    def _shim_fwd(*args):
        if len(args)==13: args=args[:10]+args[11:]
        return _orig_fwd(*args)
    fa.fwd = _shim_fwd
except Exception as e:
    print("flash-attn shim skipped:", e, flush=True)

print("loading evo2_7b...", flush=True); t0=time.time()
model = Evo2("evo2_7b")
print(f"loaded in {time.time()-t0:.0f}s", flush=True)

sp = json.load(open("six6_split.json"))
seqs = sp["seqs"]
LAYER = "blocks.28.mlp.l3"

@torch.no_grad()
def embed(seq_list, bs=256):
    outs=[]
    for i in range(0,len(seq_list),bs):
        batch=seq_list[i:i+bs]
        toks=[model.tokenizer.tokenize(s) for s in batch]
        ids=torch.tensor(toks, dtype=torch.long, device="cuda")
        _, emb = model(ids, return_embeddings=True, layer_names=[LAYER])
        h = emb[LAYER].float()
        outs.append(h.mean(1).cpu().numpy())
        if i % 2048 == 0: print(f"  embed {i}/{len(seq_list)}", flush=True)
    return np.concatenate(outs,0)

t0=time.time()
X = embed(seqs)
print(f"embedded {X.shape} in {time.time()-t0:.0f}s", flush=True)
os.makedirs("out",exist_ok=True)
np.save("out/X_evo2_six6.npy", X.astype(np.float16))
json.dump({"n":len(seqs),"dim":int(X.shape[1]),"layer":LAYER}, open("out/evo2_meta.json","w"))
print("DONE", flush=True)
