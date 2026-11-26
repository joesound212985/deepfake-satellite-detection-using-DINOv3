#!/usr/bin/env python3
"""t-SNE of frozen CLS features, real vs manipulated. Settings fixed in FEWSHOT_PROTOCOL.md (addendum).
Illustration only. CPU. Does not overwrite an existing figure."""
import json, sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

HERE = Path(__file__).resolve().parent
FEAT = Path.home() / "data" / "features" / "fewshot_cls384"
EXS = [("sat_orig", "Original DINOv3 (SAT)"), ("full_s42", "Fine-tuned (full model)")]
SETS = [("rsfake", "RSFAKE-1M subset"), ("uw", "UW fake satellite"), ("dmaer", "DM-AER"), ("gdd_crossgenerator", "GDD cross-generator")]
N = 500
out = HERE / "tsne_features.png"
if out.exists():
    sys.exit(f"{out} exists; not overwriting")
fig, axes = plt.subplots(len(EXS), len(SETS), figsize=(3.0 * len(SETS), 3.0 * len(EXS)), dpi=200)
info = {}
for j, (d, dl) in enumerate(SETS):
    meta = json.loads((FEAT / EXS[0][0] / f"{d}.ids.json").read_text())
    y = np.array([m["label"] for m in meta]); rs = np.random.RandomState(0)
    sel = np.sort(np.concatenate([rs.choice(np.flatnonzero(y == c), min(N, int((y == c).sum())), replace=False) for c in (0, 1)]))
    info[d] = {"n_real": int((y[sel] == 0).sum()), "n_manipulated": int((y[sel] == 1).sum())}
    for i, (ex, el) in enumerate(EXS):
        m2 = json.loads((FEAT / ex / f"{d}.ids.json").read_text())
        assert [m["id"] for m in m2] == [m["id"] for m in meta]
        F = np.load(FEAT / ex / f"{d}.npy").astype(np.float64)[sel]
        F /= np.linalg.norm(F, axis=1, keepdims=True) + 1e-12
        Z = TSNE(n_components=2, perplexity=30, init="pca", random_state=0).fit_transform(PCA(50, random_state=0).fit_transform(F))
        ax = axes[i, j]
        for c, col, lab in ((0, "#1f77b4", "real"), (1, "#d62728", "manipulated")):
            ax.scatter(Z[y[sel] == c, 0], Z[y[sel] == c, 1], s=3, c=col, alpha=0.6, linewidths=0, label=lab)
        ax.set_xticks([]); ax.set_yticks([])
        if i == 0: ax.set_title(dl, fontsize=9)
        if j == 0: ax.set_ylabel(el, fontsize=9)
axes[0, 0].legend(fontsize=7, markerscale=3, loc="best", frameon=True)
fig.tight_layout(); fig.savefig(out)
(HERE / "tsne_features.json").write_text(json.dumps({"protocol": "FEWSHOT_PROTOCOL.md addendum", "perplexity": 30, "pca": 50, "random_state": 0, "samples": info}, indent=1))
print("[save]", out, info)
