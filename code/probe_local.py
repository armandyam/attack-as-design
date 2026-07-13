#!/usr/bin/env python
# LOCAL post-processing for the Evo2 DNA gate (no GPU). Run after harvesting
# out/X_evo2_six6.npy from the Modal job, alongside six6_split.json (has y labels).
import json, numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from scipy.stats import spearmanr

sp = json.load(open("six6_split.json"))          # {seqs, y}
X  = np.load("X_evo2_six6.npy").astype(np.float32) # (12000, D) from the job
y  = np.array(sp["y"])
assert len(X)==len(y), (X.shape, len(y))

Xtr,Xte,ytr,yte = train_test_split(X,y,test_size=0.25,random_state=0)
sc=StandardScaler().fit(Xtr); Xtr=sc.transform(Xtr); Xte=sc.transform(Xte)
res={"n":len(y),"dim":int(X.shape[1])}
res["ridge_spearman"]=float(spearmanr(Ridge(alpha=10).fit(Xtr,ytr).predict(Xte),yte).correlation)
res["mlp_spearman"]=float(spearmanr(MLPRegressor((256,64),max_iter=300,random_state=0).fit(Xtr,ytr).predict(Xte),yte).correlation)
# GATE: beat DNABERT-2's 0.525; ideally approach one-hot 0.765
res["dnabert2_baseline"]=0.525; res["onehot_baseline"]=0.765
res["gate_pass"]= res["mlp_spearman"]>0.5
json.dump(res, open("evo2_six6_decodability.json","w"), indent=2)
print(json.dumps(res, indent=2))
