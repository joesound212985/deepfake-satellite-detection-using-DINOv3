#!/bin/bash
# Pack everything needed to continue on another computer into one folder.
#   ./make_bundle.sh [/path/to/bundle_dir]      (default ~/grsl_bundle)
# Contents:
#   grsl/            this project (git repo, scripts, RUNLOG, results jsons)
#   overhead.tar     main dataset (Google-Form gated, cannot be re-downloaded by script)  ~1.2 GB
#   results_dinov3/  finished runs + the old checkpoint (test_metrics.json, figures, best_checkpoint.pt)
#   rsfake_subset.tar (optional, BUNDLE_RSFAKE=1): the 10k-image RSFAKE subset + csvs, ~1.5 GB,
#                    saves re-downloading 27.6 GB of zips on the new machine
# On the new machine:
#   cp -r <bundle>/grsl ~/grsl; cp -r <bundle>/results_dinov3 ~/results_dinov3
#   cd ~/grsl && ./setup_env.sh && hf auth login
#   OVERHEAD_TAR=<bundle>/overhead.tar ./setup_data.sh      (RSFAKE_TAR=<bundle>/rsfake_subset.tar also accepted)
#   ROWS="full plain frozen resnet web radio" nohup ./run_everything.sh > logs/run_everything.out 2>&1 &
set -e
B=${1:-$HOME/grsl_bundle}
mkdir -p "$B"
echo "[bundle] -> $B"
rsync -a --exclude logs --exclude __pycache__ ~/grsl/ "$B/grsl/"
[ -f "$B/overhead.tar" ] || tar -cf "$B/overhead.tar" -C ~/data/overhead train_jpg test_jpg
rsync -a --include '*/' --include 'test_metrics.json' --include '*.png' --include 'best_checkpoint.pt' --include 'train_logs.json' --exclude '*' \
      --exclude 'last_epoch.pt' ~/results_dinov3/ "$B/results_dinov3/"
if [ "${BUNDLE_RSFAKE:-0}" = "1" ] && [ -f ~/data/rsfake/subset_seed42_n500.csv ]; then
  python - <<'EOF'
import pandas as pd, tarfile, pathlib, os
root = pathlib.Path.home()/"data/rsfake"; df = pd.read_csv(root/"subset_seed42_n500.csv")
out = pathlib.Path(os.environ.get("B", str(pathlib.Path.home()/"grsl_bundle")))/"rsfake_subset.tar"
with tarfile.open(out, "w") as t:
    for p in df.image_path: t.add(root/p, arcname=p)
    for p in ["subset_seed42_n500.csv", "SPLIT/RSFAKE_test_new.csv", "REAL/fmow_process/fmow_test_patches.csv"]: t.add(root/p, arcname=p)
print("[bundle] rsfake_subset.tar written")
EOF
fi
du -sh "$B"/* | sed 's/^/[bundle] /'
echo "[bundle] done"
