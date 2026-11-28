#!/usr/bin/env python3
"""TruFor (authors' code and weights, unchanged) on the test folder. Protocol: TRUFOR_PROTOCOL.md."""
import sys, os, json, argparse, time
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
ap = argparse.ArgumentParser()
ap.add_argument("--src", default=str(Path.home() / "external/trufor_src/TruFor/TruFor_train_test"))
ap.add_argument("--weights", default=str(Path.home() / "external/trufor_src/weights/trufor.pth.tar"))
ap.add_argument("--root", default=str(Path.home() / "data/overhead/test_jpg/test"))
ap.add_argument("--out", default=str(HERE / "trufor_overhead_metrics.json"))
ap.add_argument("--limit", type=int, default=0, help="debug: every n-th image up to this many; NOT a result")
a = ap.parse_args()
out = Path(a.out)
if out.exists():
    sys.exit(f"{out} exists; not overwriting")
sys.path.insert(0, str(HERE))
from eval_cross_dataset import build_overhead, pix_stats  # noqa: E402
items = build_overhead(Path(a.root))
if a.limit:
    items = items[:: max(1, len(items) // a.limit)][: a.limit]

import cv2, torch  # noqa: E402
from torch.nn import functional as F  # noqa: E402
from PIL import Image  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402
os.chdir(a.src); sys.path.insert(0, a.src)
from lib.config import config, update_config  # noqa: E402
from lib.utils import get_model  # noqa: E402
update_config(config, argparse.Namespace(experiment="trufor_ph3", gpu=None, opts=["TEST.MODEL_FILE", a.weights]))
ck = torch.load(a.weights, map_location="cpu", weights_only=True)
model = get_model(config); model.load_state_dict(ck["state_dict"]); model = model.cuda().eval()

recs = []; t0 = time.perf_counter()
with torch.no_grad():
    for n, it in enumerate(items):
        rgb = np.array(Image.open(it["img"]).convert("RGB"))                      # as dataset_test.TestDataset
        x = (torch.tensor(rgb.transpose(2, 0, 1), dtype=torch.float) / 256.0)[None].cuda()
        pred, conf, det, npp = model(x, save_np=False)
        pm = F.softmax(torch.squeeze(pred, 0), dim=0)[1].cpu().numpy()
        r = {"id": it["id"], "label": it["label"], "cls3": it["cls3"], "score": float(torch.sigmoid(det).item())}
        if it["mask"] is not None:
            m = cv2.imread(str(it["mask"]), cv2.IMREAD_UNCHANGED)
            if m.ndim == 3:
                m = m[:, :, 0]
            gt = (m.astype(np.float32) > (127 if m.max() > 1 else 0.5)).astype(np.uint8)
            assert gt.shape == pm.shape, (gt.shape, pm.shape)
            if gt.sum() > 0:
                r["iou"], r["dice"], r["mcc"] = pix_stats((pm >= 0.5).astype(np.uint8), gt)
        recs.append(r)
        if n % 200 == 0:
            print(n, len(items), flush=True)
el = time.perf_counter() - t0
y = np.array([r["label"] for r in recs]); c = np.array([r["cls3"] for r in recs]); s = np.array([r["score"] for r in recs])
loc = [r for r in recs if "iou" in r]
auc = lambda keep: float(roc_auc_score(y[keep], s[keep])) if len(set(y[keep])) == 2 else None
res = {"protocol": "TRUFOR_PROTOCOL.md", "root": a.root, "weights": a.weights, "n_images": len(recs),
       "n_pristine": int((c == 0).sum()), "n_tampered": int((c == 1).sum()), "n_fake": int((c == 2).sum()),
       "auc_pristine_vs_manipulated": auc(np.ones(len(y), bool)), "auc_pristine_vs_tampered": auc(c != 2), "auc_pristine_vs_fake": auc(c != 1),
       "accuracy_thr0.5": float(((s >= 0.5) == y).mean()), "detection_rate_thr0.5": float((s[y == 1] >= 0.5).mean()) if (y == 1).any() else None,
       "detection_rate_tampered_thr0.5": float((s[c == 1] >= 0.5).mean()) if (c == 1).any() else None,
       "detection_rate_fake_thr0.5": float((s[c == 2] >= 0.5).mean()) if (c == 2).any() else None,
       "false_alarm_rate_thr0.5": float((s[y == 0] >= 0.5).mean()) if (y == 0).any() else None,
       "localization": {"n_images_with_mask": len(loc), "mask_threshold": 0.5, "resolution": "native 512x512",
                        "mean_iou": float(np.mean([r["iou"] for r in loc])) if loc else None,
                        "mean_dice": float(np.mean([r["dice"] for r in loc])) if loc else None,
                        "mean_mcc": float(np.mean([r["mcc"] for r in loc])) if loc else None},
       "wall_sec": el, "torch_version": torch.__version__}
out.write_text(json.dumps(res, indent=1)); Path(str(out).replace(".json", ".per_image.json")).write_text(json.dumps(recs))
print(json.dumps(res, indent=1))
