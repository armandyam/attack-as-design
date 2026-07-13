
import os, json, time, numpy as np, torch

# ---- Patch 1: torch.load weights_only=False (evo2 checkpoint carries a TE global) ----
_orig_load = torch.load
def _patched_load(*a, **k):
    k["weights_only"] = False
    return _orig_load(*a, **k)
torch.load = _patched_load

# ---- Patch 2: force FP8 off -> bf16 (A100 cc=8.0 < 8.9 required for FP8) ----
import transformer_engine.pytorch as te
_orig_fp8 = te.fp8_autocast
def _no_fp8(*a, **k):
    k["enabled"] = False
    return _orig_fp8(*a, **k)
te.fp8_autocast = _no_fp8

# ---- Patch 3 (REVISED): flash-attn fwd RETURN-arity shim ----
# NGC 25.01 ships flash-attn 2.4.2 whose fwd returns 8 tensors
# [out, q, k, v, out_padded, softmax_lse, S_dmask, rng_state];
# the current vortex HEAD unpacks only 4 (out, softmax_lse, S_dmask, rng_state).
# Collapse to 4: out=first, and the last three. Also normalize a 13-arg call
# (softcap positional) down to 12 for older ABI, harmless if not present.
import flash_attn_2_cuda as fa
def _mk_shim(orig):
    def _shim(*args, **kw):
        if len(args) == 13:            # drop softcap positional if a newer caller adds it
            args = args[:10] + args[11:]
        res = orig(*args, **kw)
        if isinstance(res, (list, tuple)) and len(res) > 4:
            res = [res[0], res[-3], res[-2], res[-1]]
        return res
    return _shim
fa.fwd = _mk_shim(fa.fwd)
if hasattr(fa, "varlen_fwd"):
    fa.varlen_fwd = _mk_shim(fa.varlen_fwd)

from evo2 import Evo2   # import AFTER all patches

t0=time.time()
model = Evo2("evo2_7b")
print("model loaded", round(time.time()-t0), "s", flush=True)

kmers = json.load(open("dna_8mers.json"))
LAYER = "blocks.28.mlp.l3"
tok = model.tokenizer

@torch.no_grad()
def embed_batch(seqs):
    ids = torch.tensor([tok.tokenize(s) for s in seqs], dtype=torch.long, device="cuda")
    _, emb = model(ids, return_embeddings=True, layer_names=[LAYER])
    h = emb[LAYER]                       # (B, L, D)
    return h.mean(dim=1).float().cpu().numpy()  # mean-pool over 8 nt positions

# --- fast self-test: validate the shim on 2 seqs before the full loop ---
tv = embed_batch(kmers[:2])
assert tv.ndim == 2 and tv.shape[0] == 2 and tv.shape[1] > 0, f"bad embed shape {tv.shape}"
print("self-test OK, dim =", tv.shape[1], flush=True)

BS = 256
rows=[]; t1=time.time()
for i in range(0, len(kmers), BS):
    rows.append(embed_batch(kmers[i:i+BS]))
    if i % (BS*20) == 0:
        print(f"{i}/{len(kmers)}  {round(time.time()-t1)}s", flush=True)
X = np.concatenate(rows, 0)
print("X shape", X.shape, flush=True)

os.makedirs("out", exist_ok=True)
np.save("out/X_evo2.npy", X)
json.dump({"n": int(X.shape[0]), "dim": int(X.shape[1]), "layer": LAYER,
           "order": "canonical_lexicographic_ACGT_8mer",
           "evo2_git": "53f195997257c56c00e5ef8d33a54f5baad143a6",
           "flash_attn": "2.4.2.dev3", "wall_s": round(time.time()-t0)},
          open("out/evo2_extract_meta.json","w"))
print("DONE", round(time.time()-t0), "s", flush=True)
