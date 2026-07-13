
import os, json, glob, subprocess, time
os.environ["TF_FORCE_UNIFIED_MEMORY"]="0"
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"]="0.9"
t0=time.time()
os.makedirs("out",exist_ok=True)
# run colabfold: monomer, 3 recycles, no amber/templates (foldability screen)
cmd=["colabfold_batch","validation_seqs.fasta","out/af2","--num-recycle","3"]
print("running:"," ".join(cmd),flush=True)
r=subprocess.run(cmd,capture_output=True,text=True)
print("STDOUT tail:\n","\n".join(r.stdout.splitlines()[-25:]),flush=True)
print("STDERR tail:\n","\n".join(r.stderr.splitlines()[-25:]),flush=True)
# collect rank-1 pLDDT + pTM per sequence from score jsons
res={}
for sj in sorted(glob.glob("out/af2/*_scores_rank_001_*.json")):
    name=os.path.basename(sj).split("_scores_rank_001")[0]
    d=json.load(open(sj))
    import numpy as np
    res[name]={"plddt_mean":float(np.mean(d["plddt"])),"ptm":float(d.get("ptm",0))}
json.dump(res,open("out/af2_scores.json","w"))
print("PARSED",len(res),"scores | elapsed",round(time.time()-t0),"s",flush=True)
for k,v in res.items(): print(f"  {k:16s} pLDDT={v['plddt_mean']:.1f} pTM={v['ptm']:.3f}",flush=True)
