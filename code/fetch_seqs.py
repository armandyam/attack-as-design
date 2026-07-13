
import pandas as pd, numpy as np, requests, json, time, sys
mm = pd.read_parquet("data/promoter_coords.parquet").reset_index(drop=True)
srv="https://rest.ensembl.org"; H={"Content-Type":"application/json","Accept":"application/json"}
def region(r): return f"{r.chrom.replace('chr','')}:{r.win_start}..{r.win_end}:{1 if r.strand=='+' else -1}"
seqs={}; B=50; N=len(mm)
for i in range(0,N,B):
    chunk=mm.iloc[i:i+B]; regs=[region(r) for r in chunk.itertuples()]
    for attempt in range(5):
        try:
            resp=requests.post(f"{srv}/sequence/region/human",headers=H,data=json.dumps({"regions":regs}),timeout=60)
            if resp.status_code==429:
                time.sleep(float(resp.headers.get("Retry-After",2))+0.5); continue
            resp.raise_for_status(); arr=resp.json()
            for nm,s in zip(chunk.name,arr): seqs[nm]=s["seq"].upper()
            break
        except Exception as e:
            if attempt==4: print("FAIL batch",i,e,flush=True)
            time.sleep(2)
    if i % 2000==0: print(f"{i}/{N} fetched={len(seqs)}",flush=True)
    time.sleep(0.08)
json.dump(seqs, open("data/promoter_seqs.json","w"))
print("DONE fetched", len(seqs), "of", N, flush=True)
