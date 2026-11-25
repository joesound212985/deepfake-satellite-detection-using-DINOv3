#!/usr/bin/env python3
"""Live dashboard for the GRSL queue.   python ~/grsl/watch.py      (Ctrl-C to exit; runs keep going)"""
import os, re, json, time, subprocess, sys
from pathlib import Path
from datetime import datetime, timedelta

HOME = Path.home()
GRSL = HOME / "grsl"
RES = HOME / "results_dinov3"
SAT = RES/"vit/dinov3-vitl16-pretrain-sat493m"
WEBD = RES/"vit/dinov3-vitl16-pretrain-lvd1689m"
LAPTOP = {"radio_s42", "plain_s43", "plain_s44", "fullnorm_s42", "webnorm_s42"}  # run on the second machine (run_laptop.sh), copied back
ROWS = [  # tag, results dir, label, est. hours (measured on this machine where a run has finished) -- in queue order
    ("full_s42",   SAT/"full_s42",    "Full model (DINOv3-SAT)",   8.8),
    ("plain_s42",  SAT/"plain_s42",   "Plain decoder",             8.3),
    ("frozen_s42", SAT/"frozen_s42",  "Frozen backbone",           4.8),
    ("web_s42",    WEBD/"web_s42",    "Web-pretrained DINOv3",     8.8),
    ("resnet_s42", RES/"resnet/resnet50/resnet_s42",       "ResNet-50 baseline",  3.0),
    ("radio_s42",  RES/"radio/C-RADIOv4-SO400M/radio_s42", "C-RADIOv4 baseline [laptop]", 13.0),
    ("resnetlr_s42", RES/"resnet/resnet50/resnetlr_s42",   "ResNet-50, LR 1e-4",  3.2),
    ("web_s43",    WEBD/"web_s43",    "Web DINOv3, seed 43",       8.8),
    ("web_s44",    WEBD/"web_s44",    "Web DINOv3, seed 44",       8.8),
    ("full_s43",   SAT/"full_s43",    "Full model, seed 43",       8.8),
    ("full_s44",   SAT/"full_s44",    "Full model, seed 44",       8.8),
    ("plain_s43",  SAT/"plain_s43",   "Plain decoder, seed 43 [laptop]",    8.3),
    ("plain_s44",  SAT/"plain_s44",   "Plain decoder, seed 44 [laptop]",    8.3),
    ("fullnorm_s42", SAT/"fullnorm_s42", "Norm check, SAT (side) [laptop]",  8.8),
    ("webnorm_s42",  WEBD/"webnorm_s42", "Norm check, web (side) [laptop]",  8.8),
]
# (metrics file stem, log stem, label)
CROSS = [("cross_fmow", "fmow-fake-small"), ("cross_rsfake", "RSFAKE-1M subset"), ("cross_uw", "UW fake satellite"),
         ("cross_dmaer", "DM-AER (size-matched)")]
CROSS_WEB = [("cross_web_fmow", "fmow-fake-small"), ("cross_web_rsfake", "RSFAKE-1M subset"), ("cross_web_uw", "UW fake satellite"),
             ("cross_web_dmaer", "DM-AER (size-matched)")]
QUEUE_PAT = "run_everything.sh|run_extra_seeds|run_web_seeds|run_norm_check|run_laptop|run_fewshot|run_frozen_cross"
ROLE = os.environ.get("GRSL_ROLE") or ("laptop" if (GRSL / ".laptop").exists() else "workstation")
if ROLE == "laptop":      # dashboard started on the laptop itself: show only its own rows, read locally
    ROWS = [r for r in ROWS if r[0] in LAPTOP]; LAPTOP = set()
BAR_W = 28


LAPTOP_HOST = "" if ROLE == "laptop" else os.environ.get("GRSL_LAPTOP", "joe@100.72.70.64")   # GRSL_LAPTOP="" disables the laptop block
REMOTE = r'''
import json, glob, os, re, subprocess
H = os.path.expanduser("~")
o = {"done": {}, "tag": None, "log": ""}
def sh(c):
    try: return subprocess.check_output(c, shell=True, text=True, stderr=subprocess.DEVNULL).strip()
    except Exception: return ""
o["gpu"] = sh("nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu,temperature.gpu,power.draw --format=csv,noheader,nounits")
t = [int(open(f).read()) // 1000 for f in glob.glob("/sys/class/hwmon/hwmon*/temp*_input")
     if open(os.path.dirname(f) + "/name").read().strip() == "coretemp"]
o["cpu"] = max(t) if t else None
o["turbo_off"] = sh("cat /sys/devices/system/cpu/intel_pstate/no_turbo")
o["profile"] = sh("powerprofilesctl get")
o["load"] = sh("cut -d' ' -f1 /proc/loadavg")
o["driver"] = bool(sh("pgrep -f 'run_lapto[p]|run_everythin[g]|run_norm_chec[k]'"))
for pid in sh("pgrep -f 'dinov3_forensics_train_v[2].py|train_nor[m].py'").split():
    try:
        for kv in open(f"/proc/{pid}/environ", "rb").read().split(b"\0"):
            if kv.startswith(b"RUN_TAG="): o["tag"] = kv[8:].decode()
    except Exception: pass
if o["tag"]:
    try:
        with open(f"{H}/grsl/logs/{o['tag']}.log", "rb") as f:
            f.seek(0, 2); n = f.tell(); f.seek(max(0, n - 60000)); o["log"] = f.read().decode(errors="ignore")
    except Exception: pass
for f in glob.glob(f"{H}/results_dinov3/*/*/*/test_metrics.json"):
    try:
        m = json.load(open(f)); L = m["localization_tampered"]
        o["done"][os.path.basename(os.path.dirname(f))] = [m["accuracy"], m["macro_f1"], L["mean_iou"], L["mean_dice"], m["train_sec_per_epoch"]]
    except Exception: pass
print(json.dumps(o))
'''
_lap = {"t": 0.0, "data": None}


def laptop(every=15):
    """One ssh call every `every` seconds: sensors, running tag + log tail, finished rows. None if unreachable."""
    if not LAPTOP_HOST: return None
    if time.time() - _lap["t"] < every: return _lap["data"]
    _lap["t"] = time.time()
    try:
        out = subprocess.run(["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=4", LAPTOP_HOST, "python3 -"],
                             input=REMOTE, capture_output=True, text=True, timeout=12).stdout
        _lap["data"] = json.loads(out)
    except Exception:
        _lap["data"] = None
    return _lap["data"]


def cpu_temp():
    try:
        for h in (Path("/sys/class/hwmon")).glob("hwmon*"):
            if (h / "name").read_text().strip() in ("k10temp", "coretemp"):
                return max(int(f.read_text()) // 1000 for f in h.glob("temp*_input"))
    except Exception:
        pass
    return None


def side_jobs():
    """Non-training jobs running here right now (evaluations, SimLBR, few-shot): log name + last progress."""
    out = []
    try:
        for lg in sorted((GRSL / "logs").glob("*.log"), key=lambda f: -f.stat().st_mtime)[:6]:
            if time.time() - lg.stat().st_mtime > 90: continue
            with open(lg, "rb") as f:
                f.seek(0, 2); n = f.tell(); f.seek(max(0, n - 3000)); txt = f.read().decode(errors="ignore").replace("\r", "\n")
            if "[train] epoch" in txt: continue          # a training row, shown in the table
            m = None
            for m in re.finditer(r"(\d+)%\|[^|]*\|\s*(\d+)/(\d+) \[([^\]<]*)<([^,\]]*)", txt): pass
            out.append(f"{lg.stem}" + (f" {m.group(1)}% (left {m.group(5)})" if m else " running"))
    except Exception:
        pass
    return out


def bar(frac, w=BAR_W):
    frac = max(0.0, min(1.0, frac)); n = int(round(frac * w))
    return "█" * n + "░" * (w - n)


def fmt_td(sec):
    if sec is None or sec != sec or sec < 0: return "  --  "
    return str(timedelta(seconds=int(sec)))


def gpu():
    try:
        o = subprocess.check_output(["nvidia-smi", "--query-gpu=memory.used,memory.total,utilization.gpu,temperature.gpu,power.draw",
                                     "--format=csv,noheader,nounits"], text=True).strip().split(", ")
        return f"GPU {int(o[0])/1024:.1f}/{int(o[1])/1024:.0f} GB  util {o[2]}%  {o[3]}°C  {float(o[4]):.0f} W"
    except Exception:
        return "GPU n/a"


def active_rows():
    env = os.environ.get("ROWS")
    if env: return env.split()
    try:  # read ROWS= from the running driver's environment
        for pid in subprocess.check_output(["pgrep", "-f", "run_everything.sh"], text=True).split():
            e = Path(f"/proc/{pid}/environ").read_bytes().split(b"\0")
            for kv in e:
                if kv.startswith(b"ROWS="): return kv[5:].decode().split()
    except Exception:
        pass
    return None


def parse_log(logf: Path):
    """Return dict(epoch, epochs, it, its, it_s, phase, last_val, eval_frac)."""
    d = {"epoch": 0, "epochs": 50, "it": 0, "its": 3000, "it_s": None, "phase": "starting", "val": None, "eval_frac": None}
    if isinstance(logf, str):
        txt = logf.replace("\r", "\n")
    else:
        try:
            txt = logf.read_bytes()[-60000:].decode(errors="ignore").replace("\r", "\n")
        except Exception:
            return d
    m = None
    for m in re.finditer(r"\[train\] epoch (\d+)/(\d+):\s+\d+%\|[^|]*\|\s*(\d+)/(\d+) \[[^\]]*?,\s*([\d.]+)(it/s|s/it)", txt): pass
    if m:
        d.update(epoch=int(m.group(1)), epochs=int(m.group(2)), it=int(m.group(3)), its=int(m.group(4)), phase="train")
        r = float(m.group(5)); d["it_s"] = r if m.group(6) == "it/s" else (1.0 / r if r else None)
    ev = None
    for ev in re.finditer(r"\[eval\]:\s+\d+%\|[^|]*\|\s*(\d+)/(\d+)", txt): pass
    if ev and (not m or ev.start() > m.start()):
        d["phase"] = "eval"; d["eval_frac"] = int(ev.group(1)) / max(int(ev.group(2)), 1)
    vl = None
    for vl in re.finditer(r"\[epoch (\d+)/(\d+)\].*?val_acc=([\d.]+) val_f1=([\d.]+) val_iou=([\d.]+).*?best=([\d.]+) sec=(\d+)", txt): pass
    if vl:
        d["val"] = dict(ep=int(vl.group(1)), acc=float(vl.group(3)), f1=float(vl.group(4)), iou=float(vl.group(5)),
                        best=float(vl.group(6)), sec=int(vl.group(7)))
        if d["epoch"] <= int(vl.group(1)) and d["phase"] != "train":
            d["epoch"] = int(vl.group(1)); d["phase"] = "val" if d["phase"] == "starting" else d["phase"]
    if "[eval] TEST set" in txt or "segmentation confusion" in txt: d["phase"] = "final test eval"
    if "Traceback" in txt[-5000:]: d["phase"] = "ERROR (see log)"
    return d


def running_tag():
    try:
        out = subprocess.check_output(["pgrep", "-af", "dinov3_forensics_train_v2.py|train_norm.py"], text=True)
        for pid in re.findall(r"^(\d+)", out, re.M):
            e = Path(f"/proc/{pid}/environ").read_bytes().split(b"\0")
            for kv in e:
                if kv.startswith(b"RUN_TAG="): return kv[8:].decode()
    except Exception:
        pass
    return None


def progress(d, hrs):
    """(fraction done, seconds remaining, text) for a running row from a parsed log."""
    ep_frac = (d["epoch"] - 1 + d["it"] / max(d["its"], 1)) / max(d["epochs"], 1) if d["phase"] == "train" else d["epoch"] / max(d["epochs"], 1)
    if d["phase"] == "final test eval": ep_frac = 0.98
    ep_frac = max(0.0, min(0.99, ep_frac))
    sec_ep = d["val"]["sec"] + 60 if d["val"] else (d["its"] / d["it_s"] + 60 if d["it_s"] else hrs * 3600 / 50)
    remaining = (1 - ep_frac) * d["epochs"] * sec_ep
    if d["phase"] == "train":
        sub = f"epoch {d['epoch']}/{d['epochs']}  it {d['it']}/{d['its']}  {d['it_s'] or 0:.1f} it/s"
    elif d["phase"] == "eval":
        sub = f"epoch {d['epoch']}/{d['epochs']}  validating {d['eval_frac']*100:.0f}%"
    else:
        sub = f"epoch {d['epoch']}/{d['epochs']}  {d['phase']}"
    return ep_frac, remaining, sub


def val_line(d):
    v = d["val"]
    return f"  {'':<31} last epoch {v['ep']}: val_acc={v['acc']:.4f} val_f1={v['f1']:.4f} val_iou={v['iou']:.4f} best_f1={v['best']:.4f} ({v['sec']}s/epoch)"


def main():
    while True:
        cur = running_tag()
        lap = laptop()
        driver_alive = subprocess.call(["pgrep", "-f", QUEUE_PAT], stdout=subprocess.DEVNULL) == 0
        lines = []
        ct = cpu_temp()
        lines.append(f"GRSL experiment queue   {datetime.now():%a %H:%M:%S}")
        lines.append(f"  {'LAPTOP     ' if ROLE == 'laptop' else 'WORKSTATION'}  {gpu()}   CPU {ct if ct is not None else '--'}°C   queue: {'RUNNING' if driver_alive else 'not running'}")
        if LAPTOP_HOST:
            if lap:
                try:
                    g = lap["gpu"].split(", ")
                    gs = f"GPU {int(g[0])/1024:.1f}/{int(g[1])/1024:.0f} GB  util {g[2]}%  {g[3]}°C  {float(g[4]):.0f} W"
                except Exception:
                    gs = "GPU n/a"
                lines.append(f"  LAPTOP       {gs}   CPU {lap.get('cpu', '--')}°C   queue: {'RUNNING' if lap.get('driver') else 'not running'}"
                             f"   power: {lap.get('profile') or '?'}{', turbo off' if lap.get('turbo_off') == '1' else ''}   load {lap.get('load', '?')}")
            else:
                lines.append("  LAPTOP       not reachable (asleep, off, or Tailscale down)")
        sj = side_jobs()
        me = "LAPTOP" if ROLE == "laptop" else "WORKSTATION"
        if cur: lines.append(f"  {me} now: training {cur} (row marked ▶ below)" + (";  also: " + ",  ".join(sj) if sj else ""))
        elif sj: lines.append(f"  {me} now: no training row at the moment; running evaluation jobs:  " + ",  ".join(sj))
        else: lines.append(f"  {me} now: idle" if not driver_alive else f"  {me} now: between jobs")
        if LAPTOP_HOST and lap:
            lines.append(f"  LAPTOP now: " + (f"training {lap['tag']} (row marked ▶ below)" if lap.get("tag") else "no training process"))
        lines.append("─" * 118)
        total_done = total_left = 0.0; done_runs = 0; n_rows = 0
        lap_left = 0.0; lap_rows = lap_done = 0
        for tag, out, label, hrs in ROWS:
            logf = GRSL / "logs" / f"{tag}.log"
            if tag in LAPTOP:      # runs on the second machine; live state comes over ssh
                lap_rows += 1
                if (out / "test_metrics.json").exists():
                    m = json.load(open(out / "test_metrics.json")); lap_done += 1
                    lines.append(f"✔ {label:<31} {bar(1.0)} 100%  acc={m['accuracy']:.4f} F1={m['macro_f1']:.4f} IoU={m['localization_tampered']['mean_iou']:.4f} "
                                 f"Dice={m['localization_tampered']['mean_dice']:.3f}  copied back")
                elif lap and tag in lap["done"]:
                    a = lap["done"][tag]; lap_done += 1
                    lines.append(f"✔ {label:<31} {bar(1.0)} 100%  acc={a[0]:.4f} F1={a[1]:.4f} IoU={a[2]:.4f} Dice={a[3]:.3f}  done on laptop, not copied back yet")
                elif lap and lap.get("tag") == tag:
                    d = parse_log(lap["log"]); frac, rem, sub = progress(d, hrs); lap_left += rem / 3600
                    lines.append(f"▶ {label:<31} {bar(frac)} {frac*100:3.0f}%  {sub}   ETA {fmt_td(rem)}")
                    if d["val"]: lines.append(val_line(d))
                elif lap:
                    lap_left += hrs
                    lines.append(f"· {label:<31} {bar(0)}   queued on the laptop")
                else:
                    lines.append(f"? {label:<31} {bar(0)}   laptop not reachable")
                continue
            n_rows += 1
            if (out / "test_metrics.json").exists():
                m = json.load(open(out / "test_metrics.json"))
                done_runs += 1; total_done += hrs
                lines.append(f"✔ {label:<31} {bar(1.0)} 100%  acc={m['accuracy']:.4f} F1={m['macro_f1']:.4f} IoU={m['localization_tampered']['mean_iou']:.4f} "
                             f"Dice={m['localization_tampered']['mean_dice']:.3f}  {m['train_sec_per_epoch']:.0f}s/ep")
            elif tag == cur:
                d = parse_log(logf); ep_frac, remaining, sub = progress(d, hrs)
                total_done += hrs * ep_frac; total_left += remaining / 3600
                lines.append(f"▶ {label:<31} {bar(ep_frac)} {ep_frac*100:3.0f}%  {sub}   ETA {fmt_td(remaining)}")
                if d["val"]: lines.append(val_line(d))
            elif logf.exists() and not driver_alive:
                total_left += hrs
                lines.append(f"✖ {label:<31} {bar(0)}   stopped / incomplete (relaunch to resume)")
            else:
                total_left += hrs
                lines.append(f"· {label:<31} {bar(0)}   queued (~{hrs:g} h)")
        lines.append("─" * 118)
        tot = total_done + total_left
        lines.append(f"WORKSTATION  {bar(total_done / tot if tot else 0, 34)} {100*total_done/tot if tot else 0:3.0f}%   {done_runs}/{n_rows} runs done   "
                     f"ETA training ≈ {fmt_td(total_left*3600)}  (finish ≈ {(datetime.now()+timedelta(hours=total_left)):%a %H:%M})")
        if LAPTOP_HOST and lap:
            lines.append(f"LAPTOP       {lap_done}/{lap_rows} runs done   ETA ≈ {fmt_td(lap_left*3600)}  (finish ≈ {(datetime.now()+timedelta(hours=lap_left)):%a %H:%M}; "
                         f"queued rows use workstation hours, a rough guess)")
        lines.append("")
        for title, group in (("Cross-dataset, full_s42 (after radio):", CROSS), ("Cross-dataset, web_s42:", CROSS_WEB)):
            lines.append(title)
            for stem, label in group:
                f = GRSL / f"{stem}_metrics.json"
                if f.exists():
                    o = json.load(open(f))["overall"]
                    auc = o.get("auc_real_vs_manipulated"); acc = o.get("accuracy_thr0.5"); det = o.get("detection_rate_manipulated_thr0.5")
                    lines.append(f"  ✔ {label:<22} n={o['n_images']}  AUC={auc:.4f}  acc={acc:.4f}" if auc is not None else f"  ✔ {label:<22} n={o['n_images']}  detection rate={det:.4f}")
                else:
                    lg = GRSL / "logs" / f"{stem}.log"
                    if lg.exists() and (time.time() - lg.stat().st_mtime) < 120:
                        d = parse_log(lg); lines.append(f"  ▶ {label:<22} evaluating {d['eval_frac']*100 if d['eval_frac'] else 0:.0f}%")
                    else:
                        lines.append(f"  · {label:<22} pending")
        nrob = len(list(GRSL.glob("robust_*_metrics.json"))); nov = len(list(GRSL.glob("overhead_*_metrics.json")))
        lines.append(f"Extras (after cross-dataset): test set two-class {nov}/2   robustness {nrob}/12   "
                     f"figure {'✔' if (GRSL / 'qualitative.png').exists() else '·'}   SimLBR official {len(list(GRSL.glob('simlbr_official_*_metrics.json')))}/2   significance re-scores {len(list(GRSL.glob('overhead_*_s4?_metrics.json')))}/7   analysis_extras.json {'✔' if (GRSL / 'analysis_extras.json').exists() else '·'}")
        fz = len(list(GRSL.glob("cross_frozen_*_metrics.json"))); nf = len(list((HOME / "data/features/fewshot_cls384").glob("*/*.npy")))
        lines.append(f"Added 11-25: frozen_s42 cross-dataset {fz}/4   few-shot features {nf}/16   fewshot_metrics.json {'✔' if (GRSL / 'fewshot_metrics.json').exists() else '·'}")
        tcsv = GRSL / "table.csv"
        lines.append(f"\ntable.csv: {'written ' + datetime.fromtimestamp(tcsv.stat().st_mtime).strftime('%H:%M') if tcsv.exists() else 'pending'}")
        try:
            last = [l for l in (GRSL / "RUNLOG.md").read_text().splitlines() if l.startswith("|")][-2:]
            lines.append("RUNLOG: " + " ‖ ".join(l[2:70] for l in last))
        except Exception:
            pass
        os.system("clear"); print("\n".join(lines)); sys.stdout.flush()
        time.sleep(5)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        pass
