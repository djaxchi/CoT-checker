import sys, numpy as np
sys.path.insert(0, "scripts/step_bert")
from g0_leakage import neighbour_pairs, is_test_group
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import roc_auc_score, f1_score
z = np.load("data/tamia/step_bert_probe_v0/pooled/pooled_onpolicy_sub1000.npz")
b, m = z["b"].astype("f4"), z["m"].astype("f4"); y = z["y"]; g = z["problem_id"]
ti, nj = neighbour_pairs(z["trace_id"], z["step_idx"], +1)   # steps that have a next step
reps = {"b_i (pre-step boundary)": b[ti], "m_i (span mean)": m[ti], "c_i = m_i - b_i": (m - b)[ti],
        "b_{i+1} (end-of-step boundary)": b[nj], "e_i = b_{i+1} - b_i": b[nj] - b[ti],
        "[b_i, m_i]": np.hstack([b[ti], m[ti]])}
yy, gg = y[ti], g[ti]
te = is_test_group(gg); tr = ~te
print(f"n_train={tr.sum()} n_test={te.sum()} test prevalence={yy[te].mean():.3f}")
for name, x in reps.items():
    p = PCA(256, random_state=0).fit(x[tr]); xt = p.transform(x)
    mu, sd = xt[tr].mean(0), xt[tr].std(0) + 1e-6; xt = (xt - mu) / sd
    clf = LogisticRegressionCV(Cs=8, max_iter=3000, scoring="roc_auc").fit(xt[tr], yy[tr])
    s = clf.predict_proba(xt[te])[:, 1]
    ths = np.quantile(s, np.linspace(0.02, 0.98, 97))
    of1 = max(f1_score(yy[te], s >= t) for t in ths)
    print(f"{name:32s} AUROC={roc_auc_score(yy[te], s):.3f} oracleF1={of1:.3f}")
print("always-positive F1", round(2*yy[te].mean()/(1+yy[te].mean()),3))
