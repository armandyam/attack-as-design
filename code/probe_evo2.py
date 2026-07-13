
import subprocess, sys, json, importlib
info = {}
# versions of the installed evo2/vortex/flash-attn
for mod in ["evo2","vortex","flash_attn","transformer_engine","torch"]:
    try:
        m=importlib.import_module(mod)
        info[mod]=getattr(m,"__version__","?")
    except Exception as e:
        info[mod]=f"ERR {type(e).__name__}: {e}"
# git commit of the cloned evo2 + vendored vortex
try:
    info["evo2_git"]=subprocess.run(["git","-C","/work/evo2","rev-parse","HEAD"],capture_output=True,text=True).stdout.strip()
except Exception as e:
    info["evo2_git"]=str(e)
# pip-visible versions
pf=subprocess.run([sys.executable,"-m","pip","show","evo2","vortex","flash-attn"],capture_output=True,text=True).stdout
info["pip_show"]=pf
# inspect vortex attn_interface around the failing lines
try:
    import vortex.ops.attn_interface as ai, inspect
    src=inspect.getsource(ai)
    # capture lines mentioning fwd(
    info["fwd_callsites"]=[l.strip() for l in src.splitlines() if "fwd(" in l or "flash_attn_forward" in l][:12]
except Exception as e:
    info["fwd_callsites"]=str(e)
# flash_attn fwd return arity — call on tiny GPU tensors
try:
    import torch, flash_attn_2_cuda as fa2
    info["has_flash_attn_2_cuda"]=True
    info["fa2_fwd_doc"]=str(getattr(fa2.fwd,"__doc__",""))[:400]
except Exception as e:
    info["has_flash_attn_2_cuda"]=f"ERR {e}"
try:
    import flash_attn_gpu as fag
    info["has_flash_attn_gpu"]=True
    info["fag_fwd_doc"]=str(getattr(fag.fwd,"__doc__",""))[:400]
except Exception as e:
    info["has_flash_attn_gpu"]=f"ERR {e}"
import os; os.makedirs("out",exist_ok=True)
json.dump(info,open("out/probe.json","w"),indent=2)
print(json.dumps(info,indent=2))
