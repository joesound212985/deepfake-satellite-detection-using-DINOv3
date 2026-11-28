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
LAPTOP = {"radio_s42", "plain_s43", "fullnorm_s42", "webnorm_s42"}  # run on the second machine (run_laptop.sh), copied back
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
    ("plain_s44",  SAT/"plain_s44",   "Plain decoder, seed 44",    8.3),   # moved from the laptop 2026-11-26 (run_extra_seeds_v2b.sh)
    ("fullnorm_s42", SAT/"fullnorm_s42", "Norm check, SAT (side) [laptop]",  8.8),
    ("webnorm_s42",  WEBD/"webnorm_s42", "Norm check, web (side) [laptop]",  8.8),
    # stage 5 (run_review_seeds.sh, 2026-11-28): more seeds of the single-run rows, in queue order
    ("frozen_s43", SAT/"frozen_s43",  "Frozen backbone, seed 43",  4.8),
    ("resnet_s43", RES/"resnet/resnet50/resnet_s43",     "ResNet-50, seed 43",          3.0),
    ("frozen_s44", SAT/"frozen_s44",  "Frozen backbone, seed 44",  4.8),
    ("resnet_s44", RES/"resnet/resnet50/resnet_s44",     "ResNet-50, seed 44",          3.0),
    ("resnetlr_s43", RES/"resnet/resnet50/resnetlr_s43", "ResNet-50, LR 1e-4, seed 43", 3.2),
    ("resnetlr_s44", RES/"resnet/resnet50/resnetlr_s44", "ResNet-50, LR 1e-4, seed 44", 3.2),
]
# (metrics file stem, log stem, label)
CROSS = [("cross_fmow", "fmow-fake-small"), ("cross_rsfake", "RSFAKE-1M subset"), ("cross_uw", "UW fake satellite"),
         ("cross_dmaer", "DM-AER (size-matched)")]
CROSS_WEB = [("cross_web_fmow", "fmow-fake-small"), ("cross_web_rsfake", "RSFAKE-1M subset"), ("cross_web_uw", "UW fake satellite"),
             ("cross_web_dmaer", "DM-AER (size-matched)")]
QUEUE_PAT = "run_everything.sh|run_extra_seeds|run_web_seeds|run_norm_check|run_laptop|run_fewshot|run_frozen_cross|run_zeroshot_extras|run_seed_evals|run_trufor|run_review_seeds"
ROLE = os.environ.get("GRSL_ROLE") or ("laptop" if (GRSL / ".laptop").exists() else "workstation")
if ROLE == "laptop":      # dashboard started on the laptop itself: show only its own rows, read locally
    ROWS = [r for r in ROWS if r[0] in LAPTOP]; LAPTOP = set()
BAR_W = 28


def _laptop_default():      # address kept outside the repository
    try:
        return open(os.path.expanduser("~/.config/grsl/laptop")).read().strip()
    except OSError:
        return ""


LAPTOP_HOST = "" if ROLE == "laptop" else os.environ.get("GRSL_LAPTOP", _laptop_default())   # GRSL_LAPTOP="" disables the laptop block
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
o["mem"] = sh("free -g | awk '/Mem:/{print $3\"/\"$2}'")
o["disk"] = sh("df -h --output=avail ~ | tail -1").strip()
o["hist"] = []
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
    for l in sh("grep -ao '\[epoch [0-9]*/[0-9]*\] [^\r]*' " + f"{H}/grsl/logs/{o['tag']}.log").splitlines():
        m = re.search(r"\[epoch (\d+)/.*?train_loss=([\d.]+).*?val_f1=([\d.]+) val_iou=([\d.]+)", l)
        if m: o["hist"].append([int(m.group(1)), float(m.group(3)), float(m.group(4)), float(m.group(2))])
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


_hist = {}


def epoch_hist(logf, every=60):
    """[(epoch, val_f1, val_iou, train_loss)] for every finished epoch of a local log (re-read at most once a minute)."""
    c = _hist.get(str(logf))
    if c and time.time() - c[0] < every: return c[1]
    h = []
    try:
        out = subprocess.run(["grep", "-ao", "\\[epoch [0-9]*/[0-9]*\\] [^\r]*", str(logf)], capture_output=True, text=True, errors="ignore").stdout
        for l in out.splitlines():
            m = re.search(r"\[epoch (\d+)/.*?train_loss=([\d.]+).*?val_f1=([\d.]+) val_iou=([\d.]+)", l)
            if m: h.append([int(m.group(1)), float(m.group(3)), float(m.group(4)), float(m.group(2))])
    except Exception:
        pass
    _hist[str(logf)] = (time.time(), h); return h


def machine_extra():
    """Fan, memory, free disk, load and power profile of this machine, as short text."""
    def sh(c):
        try: return subprocess.check_output(c, shell=True, text=True, stderr=subprocess.DEVNULL).strip()
        except Exception: return ""
    fan = sh("nvidia-smi --query-gpu=fan.speed --format=csv,noheader,nounits")
    parts = [f"fan {fan}%" if fan.isdigit() else "", "RAM " + sh("free -g | awk '/Mem:/{print $3\"/\"$2}'") + " GB",
             "disk free " + sh("df -h --output=avail ~ | tail -1").strip(), "load " + sh("cut -d' ' -f1 /proc/loadavg"),
             "power: " + (sh("powerprofilesctl get") or "?")]
    return "   ".join(p for p in parts if p)


def job_name(stem):
    m = re.match(r"robust_(full|web)_(?:s(\d+)_)?(jpeg|scale)([\d.]+)$", stem)
    if m: return f"Robustness: {m.group(1)} model{' seed ' + m.group(2) if m.group(2) else ''}, {'JPEG quality' if m.group(3) == 'jpeg' else 'down-scale'} {m.group(4)}"
    m = re.match(r"cross_(web_)?s(\d+)_(\w+)$", stem)
    if m: return f"Outside data: {'web' if m.group(1) else 'full'} model seed {m.group(2)} on {m.group(3)}"
    if stem == "trufor_overhead": return "TruFor on the test folder"
    if stem.startswith("zeroshot_extras_seeds"): return "Label-free tests, seeds 43/44"
    m = re.match(r"cross_(web_|frozen_|allreals_|web_allreals_)?(\w+)$", stem)
    if m: return f"Outside data: {(m.group(1) or 'full_').rstrip('_')} model on {m.group(2)}"
    m = re.match(r"overhead_(\w+)$", stem)
    if m: return f"Test-set re-score: {m.group(1)}"
    m = re.match(r"simlbr_official_(\w+)$", stem)
    if m: return f"SimLBR, authors' code ({m.group(1)})"
    return {"fewshot_extract": "Few-shot: feature extraction", "fewshot_analyse": "Few-shot: fitting probes",
            "analyze_extras": "Extra analysis", "make_qualitative": "Qualitative figure",
            "simlbr_dmaer": "SimLBR re-run with DM-AER", "simlbr_nolbr_dmaer": "SimLBR (no blending) with DM-AER"}.get(stem, stem)


def side_jobs():
    """Non-training jobs running here right now (evaluations, SimLBR, few-shot): (name, fraction or None, time left)."""
    out = []
    try:
        for lg in sorted((GRSL / "logs").glob("*.log"), key=lambda f: -f.stat().st_mtime)[:6]:
            if time.time() - lg.stat().st_mtime > 60: continue
            with open(lg, "rb") as f:
                f.seek(0, 2); n = f.tell(); f.seek(max(0, n - 3000)); txt = f.read().decode(errors="ignore").replace("\r", "\n")
            if "[train] epoch" in txt: continue          # a training row, shown in the table
            m = None
            for m in re.finditer(r"(\d+)%\|[^|]*\|\s*(\d+)/(\d+) \[([^\]<]*)<([^,\]]*)", txt): pass
            if m and m.group(1) == "100": continue
            out.append((job_name(lg.stem), int(m.group(1)) / 100 if m else None, m.group(5) if m else ""))
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


# ------------------------------------------------------------------------------------------- rendering
from rich.console import Console, Group
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from rich import box

WS_COL, LAP_COL = "dodger_blue1", "magenta"       # one colour per machine, used for its panel, bars and rows
RES_COL = "gold1"
SPARK = "▁▂▃▄▅▆▇█"
RAMP = ["red1", "dark_orange", "gold1", "yellow3", "chartreuse3", "green3", "spring_green1", "bright_cyan"]


def sc(v):
    """Colour for a 0-1 score (accuracy, F1, AUC): red below 0.55, through orange / yellow, to bright green from 0.9."""
    return "bold bright_green" if v >= 0.9 else "green3" if v >= 0.8 else "yellow3" if v >= 0.65 else "dark_orange" if v >= 0.55 else "red1"


def metric_text(pairs, tail="", tail_style="dim"):
    t = Text()
    for name, v in pairs:
        t.append(name + " ", style="grey70"); t.append(f"{v:.4f}", style=sc(v)); t.append("  ")
    if tail: t.append(tail, style=tail_style)
    return t


def spark(h, col):
    """Coloured sparkline of validation F1 over the finished epochs, with best / latest numbers."""
    t = Text()
    if not h: return t
    f = [x[1] for x in h]; lo, hi = min(f), max(f); bi = f.index(hi)
    t.append("F1 ", style="grey70")
    for v in f[-24:]:
        k = int((v - lo) / (hi - lo) * 7) if hi > lo else 7
        t.append(SPARK[k], style=RAMP[k])
    t.append(f"  best {hi:.4f}", style="bold bright_green"); t.append(f" @ep {h[bi][0]}", style="grey70")
    t.append("   IoU ", style="grey70"); t.append(f"{h[-1][2]:.4f}", style=sc(h[-1][2]))
    t.append("   train loss ", style="grey70")
    t.append(f"{h[-1][3]:.4f}", style="bright_cyan")
    if len(h) > 1:
        d = h[-1][3] - h[-2][3]; t.append(f" {'▼' if d < 0 else '▲'}{abs(d):.4f}", style="green3" if d < 0 else "dark_orange")
    return t


def gpu_raw():
    try:
        o = subprocess.check_output(["nvidia-smi", "--query-gpu=memory.used,memory.total,utilization.gpu,temperature.gpu,power.draw",
                                     "--format=csv,noheader,nounits"], text=True).strip().split(", ")
        return [float(x) for x in o]
    except Exception:
        return None


def tcol(t, warn, hot):
    return "green" if t < warn else ("yellow" if t < hot else "bold red")


def gauge(frac, width, col):
    frac = max(0.0, min(1.0, frac)); n = int(round(frac * width))
    t = Text(); t.append("█" * n, style=col); t.append("░" * (width - n), style="grey35"); return t


def sensors(g, cpu, extra=""):
    t = Text()
    if g:
        t.append("GPU ", style="bold bright_green"); t.append_text(gauge(g[2] / 100, 12, "bright_green" if g[2] > 50 else "yellow")); t.append(f" {g[2]:3.0f}%  ")
        t.append(f"{g[3]:.0f}°C", style=tcol(g[3], 80, 87)); t.append("  ⚡", style="gold1"); t.append(f"{g[4]:.0f} W    ", style="gold1")
        t.append("VRAM ", style="bold bright_cyan"); t.append_text(gauge(g[0] / g[1], 10, "bright_cyan")); t.append(f" {g[0]/1024:.1f}/{g[1]/1024:.0f} GB    ")
    else:
        t.append("GPU n/a    ", style="red")
    t.append("CPU ", style="bold orchid1"); t.append(f"{cpu}°C" if cpu is not None else "--", style=tcol(cpu, 85, 97) if cpu is not None else "dim")
    if extra: t.append("    " + extra, style="grey62")
    return t


def row_table(rows, col):
    tb = Table(box=None, show_header=False, padding=(0, 1), expand=True)
    tb.add_column(width=1); tb.add_column(width=38, no_wrap=True, overflow="ellipsis"); tb.add_column(width=BAR_W, no_wrap=True)
    tb.add_column(width=4, justify="right"); tb.add_column(ratio=1, no_wrap=True, overflow="ellipsis")
    for st, label, frac, detail, sub in rows:
        if st == "done":
            tb.add_row(Text("✔", style="bold bright_green"), Text(label, style="green3"), gauge(1, BAR_W, "green3"), Text("100%", style="green3"),
                       detail if isinstance(detail, Text) else Text(detail, style="green"))
        elif st == "run":
            tb.add_row(Text("▶", style=f"bold {col}"), Text(label, style=f"bold {col}"), gauge(frac, BAR_W, col), f"{frac*100:.0f}%", Text(detail, style="bold white"))
            if isinstance(sub, Text) and sub.plain: tb.add_row("", Text("  └ validation history", style="grey50"), Text(""), "", sub)
            elif sub and not isinstance(sub, Text): tb.add_row("", "", Text(""), "", Text(sub, style="white"))
        elif st == "eval":
            tb.add_row(Text("◆", style="bold yellow"), Text(label, style="bold yellow"), gauge(frac, BAR_W, "yellow"), f"{frac*100:.0f}%", Text(detail, style="yellow"))
        elif st == "stop":
            tb.add_row(Text("✖", style="bold red"), Text(label, style="red"), gauge(0, BAR_W, col), "", Text(detail, style="red"))
        elif st == "unk":
            tb.add_row(Text("?", style="yellow"), Text(label, style="yellow"), gauge(0, BAR_W, col), "", Text(detail, style="yellow"))
        else:
            tb.add_row(Text("◌", style="steel_blue1"), Text(label, style="steel_blue1"), gauge(0, BAR_W, col), "", Text(detail, style="steel_blue"))
    return tb


def kv_line(title, pairs):
    t = Text(); t.append(title + ":  ", style="bold orchid1")
    for n, v in pairs:
        t.append(n + " ", style="grey70"); t.append(f"{v:.3f}", style=sc(v)); t.append("   ")
    return t


def done_detail(m, tail):
    L = m["localization_tampered"]
    return metric_text([("acc", m["accuracy"]), ("F1", m["macro_f1"]), ("IoU", L["mean_iou"]), ("Dice", L["mean_dice"])], tail)


TAB = 0


def render(tab=None):
    """tab=None: every view one after the other (--once / piped output); otherwise only that tab, under a tab bar."""
    space = None; res = []
    cur = running_tag(); lap = laptop()
    driver_alive = subprocess.call(["pgrep", "-f", QUEUE_PAT], stdout=subprocess.DEVNULL) == 0
    me, mycol = ("LAPTOP", LAP_COL) if ROLE == "laptop" else ("WORKSTATION", WS_COL)
    loc, rem = [], []
    done_h = left_h = lap_left = 0.0; n_loc = n_done = lap_done = 0; nxt = None
    for tag, out, label, hrs in ROWS:
        label = label.replace(" [laptop]", ""); logf = GRSL / "logs" / f"{tag}.log"; tm = out / "test_metrics.json"
        if tag in LAPTOP:
            if tm.exists():
                lap_done += 1; rem.append(("done", label, 1, done_detail(json.load(open(tm)), "copied back"), ""))
            elif lap and tag in lap["done"]:
                a = lap["done"][tag]; lap_done += 1
                rem.append(("done", label, 1, metric_text([("acc", a[0]), ("F1", a[1]), ("IoU", a[2]), ("Dice", a[3])], f"{a[4]:.0f} s/ep  on laptop, not copied back"), ""))
            elif lap and lap.get("tag") == tag:
                d = parse_log(lap["log"]); frac, r, sub = progress(d, hrs); lap_left += r / 3600
                rem.append(("run", label, frac, f"{sub}   ETA {fmt_td(r)}  → {(datetime.now()+timedelta(seconds=r)):%a %H:%M}", spark(lap.get("hist") or [], LAP_COL)))
            elif lap:
                lap_left += hrs; rem.append(("wait", label, 0, "queued", ""))
            else:
                rem.append(("unk", label, 0, "laptop not reachable", ""))
            continue
        n_loc += 1
        if tm.exists():
            m = json.load(open(tm)); n_done += 1; done_h += hrs
            loc.append(("done", label, 1, done_detail(m, f"{m['train_sec_per_epoch']:.0f} s/ep"), ""))
        elif tag == cur:
            d = parse_log(logf); frac, r, sub = progress(d, hrs); done_h += hrs * frac; left_h += r / 3600
            loc.append(("run", label, frac, f"{sub}   ETA {fmt_td(r)}  → {(datetime.now()+timedelta(seconds=r)):%a %H:%M}", spark(epoch_hist(logf), mycol)))
        elif logf.exists() and not driver_alive:
            left_h += hrs; loc.append(("stop", label, 0, "stopped / incomplete (run go.sh to resume)", ""))
        else:
            left_h += hrs; nxt = nxt or label; loc.append(("wait", label, 0, f"queued (~{hrs:g} h)", ""))

    # ---- this machine
    sj = side_jobs(); st = Text()
    for name, frac, left in sj:      # evaluation jobs get their own live rows, right under the finished ones
        k = max([i for i, r in enumerate(loc) if r[0] in ("done", "run")] + [-1]) + 1
        loc.insert(k, ("eval", name, frac or 0, (f"evaluation job, {left} left" if left else "evaluation job, running"), ""))
    if cur:
        st.append(" ● TRAINING ", style="bold black on green"); st.append(f"  {cur}", style="bold green")
        if sj: st.append(f"    + {len(sj)} evaluation job(s), rows marked ◆", style="yellow")
    elif sj:
        st.append(" ● EVALUATING ", style="bold black on yellow")
        st.append(f"  {sj[0][0]}", style="bold yellow"); st.append("   (no training row right now; rows marked ◆ are live)", style="yellow")
        if nxt: st.append(f"   next training: {nxt}", style="dim")
    elif driver_alive:
        st.append(" ● BETWEEN JOBS ", style="bold black on yellow")
    else:
        st.append(" ○ IDLE ", style="bold white on red"); st.append("  queue not running (./go.sh starts it)", style="red")
    tot = done_h + left_h; foot = Text()
    foot.append_text(gauge(done_h / tot if tot else 0, 40, mycol))
    foot.append(f"  {n_done}/{n_loc} runs done    training left ≈ {fmt_td(left_h*3600)}    finish ≈ {(datetime.now()+timedelta(hours=left_h)):%a %H:%M}")
    panels = [Panel(Group(sensors(gpu_raw(), cpu_temp(), machine_extra()), st, Text(""), row_table(loc, mycol), Text(""), foot),
                    title=f"[bold {mycol}] ◆ {me} ◆ [/]", border_style=mycol, box=box.HEAVY, padding=(0, 1))]

    # ---- the laptop, seen from the workstation
    if LAPTOP_HOST:
        if lap:
            try: g = [float(x) for x in lap["gpu"].split(", ")]
            except Exception: g = None
            st2 = Text()
            if lap.get("tag"): st2.append(" ● TRAINING ", style="bold black on green"); st2.append(f"  {lap['tag']}", style="bold green")
            elif lap.get("driver"): st2.append(" ● BETWEEN JOBS ", style="bold black on yellow")
            else: st2.append(" ○ IDLE ", style="bold white on red"); st2.append("  no job running", style="red")
            foot2 = Text(); nl = len(rem)
            foot2.append_text(gauge(lap_done / nl if nl else 0, 40, LAP_COL))
            foot2.append(f"  {lap_done}/{nl} runs done    left ≈ {fmt_td(lap_left*3600)}    finish ≈ {(datetime.now()+timedelta(hours=lap_left)):%a %H:%M} (rough)")
            ex = (f"RAM {lap.get('mem', '?')} GB   disk free {lap.get('disk', '?')}   load {lap.get('load', '?')}   "
                  f"power: {lap.get('profile') or '?'}{', turbo off' if lap.get('turbo_off') == '1' else ''}")
            body = Group(sensors(g, lap.get("cpu"), ex), st2, Text(""), row_table(rem, LAP_COL), Text(""), foot2)
        else:
            body = Group(Text(" ○ NOT REACHABLE ", style="bold white on red"), Text("asleep, switched off, or Tailscale is down", style="red"),
                         Text(""), row_table(rem, LAP_COL))
        panels.append(Panel(body, title=f"[bold {LAP_COL}] ◆ LAPTOP (RTX 5090) ◆ [/]", border_style=LAP_COL, box=box.HEAVY, padding=(0, 1)))

    # ---- results (workstation only)
    if (GRSL / "cross_fmow_metrics.json").exists():
        summ = None
        try:      # own test set, every finished run grouped by variant: mean over seeds and the range
            grp = {}
            for tag, out, label, hrs in ROWS:
                tm = out / "test_metrics.json"; v = None
                if tm.exists():
                    m = json.load(open(tm)); L = m["localization_tampered"]; v = [m["accuracy"], m["macro_f1"], L["mean_iou"]]
                elif lap and tag in lap.get("done", {}): v = lap["done"][tag][:3]
                if v: grp.setdefault(tag.rsplit("_s", 1)[0], []).append((tag.rsplit("_s", 1)[1], v))
            if grp:
                summ = Table(box=box.SIMPLE_HEAD, padding=(0, 1), expand=False, header_style=f"bold {RES_COL}")
                summ.add_column("own test set (3,150 images), mean over finished seeds"); summ.add_column("seeds", style="grey70")
                for h in ("accuracy", "macro-F1", "tamper IoU"): summ.add_column(h, justify="center")
                summ.add_column("accuracy", no_wrap=True)
                for name, runs in sorted(grp.items(), key=lambda kv: -sum(r[1][0] for r in kv[1]) / len(kv[1])):
                    cells = []
                    for i in range(3):
                        x = [r[1][i] for r in runs]; mu = sum(x) / len(x)
                        c = Text(f"{mu:.4f}", style=sc(mu))
                        if len(x) > 1: c.append(f" ({min(x):.4f}–{max(x):.4f})", style="grey62")
                        cells.append(c)
                    acc = sum(r[1][0] for r in runs) / len(runs)
                    summ.add_row(Text(name, style="bold"), ",".join(r[0] for r in runs), *cells, gauge((acc - 0.5) / 0.5, 20, sc(acc).replace("bold ", "")))
        except Exception:
            summ = None
        tb = Table(box=box.SIMPLE_HEAD, padding=(0, 1), expand=False, header_style=f"bold {RES_COL}")
        tb.add_column("zero-shot on outside data (AUC / accuracy)")
        for h in ("full_s42", "web_s42", "frozen_s42"): tb.add_column(h, justify="center")
        for key, label in (("fmow", "fmow-fake-small"), ("rsfake", "RSFAKE-1M subset"), ("uw", "UW fake satellite"), ("dmaer", "DM-AER (size-matched)"),
                           ("gdd_indomain", "GDD in-domain"), ("gdd_crossgenerator", "GDD cross-generator"),
                           ("gdd_crossdomain", "GDD cross-domain"), ("gdd_localization", "GDD tamper set")):
            cells = []
            for pre in ("cross_", "cross_web_", "cross_frozen_"):
                f = GRSL / f"{pre}{key}_metrics.json"
                if f.exists():
                    o = json.load(open(f))["overall"]; auc = o.get("auc_real_vs_manipulated"); acc = o.get("accuracy_thr0.5")
                    c = Text(f"{auc:.3f}", style=sc(auc)); c.append(" / ", style="grey50"); c.append(f"{acc:.3f}", style=sc(acc)); cells.append(c)
                else:
                    cells.append(Text("pending", style="dim"))
            tb.add_row(label, *cells)
        extra = []
        try:      # few-shot: AUC on the held-out half, linear probe on frozen CLS features
            fs = json.load(open(GRSL / "fewshot_metrics.json"))["datasets"]
            if (GRSL / "fewshot_gdd_metrics.json").exists():
                fs.update(json.load(open(GRSL / "fewshot_gdd_metrics.json"))["datasets"])
            ft = Table(box=box.SIMPLE_HEAD, padding=(0, 1), expand=False, header_style=f"bold {RES_COL}")
            ft.add_column("few-shot AUC, full_s42 features (labelled images per class)")
            for h in ("zero-shot", "4", "16", "64", "256", "whole pool"): ft.add_column(h, justify="center")
            col = sc
            for key, label in (("fmow", "fmow-fake-small"), ("rsfake", "RSFAKE-1M subset"), ("uw", "UW fake satellite"), ("dmaer", "DM-AER (size-matched)"),
                               ("gdd_indomain", "GDD in-domain"), ("gdd_crossgenerator", "GDD cross-generator"),
                               ("gdd_crossdomain", "GDD cross-domain"), ("gdd_localization", "GDD tamper set")):
                if key not in fs: continue
                d = fs[key]; e = d["extractors"]["full_s42"]; z = (d.get("zero_shot") or {}).get("full_s42", {}).get("auc")
                cells = [Text(f"{z:.3f}", style=col(z)) if isinstance(z, float) else Text("-", style="dim")]
                for k in ("k4", "k16", "k64", "k256"):
                    v = e.get(k); a = v["auc"]["mean"] if isinstance(v, dict) else None
                    cells.append(Text(f"{a:.3f}", style=col(a)) if a is not None else Text("-", style="dim"))
                a = e["whole_pool"]["auc"]; cells.append(Text(f"{a:.3f}", style=col(a)))
                ft.add_row(label, *cells)
            extra.append(ft)
        except Exception:
            pass
        try:      # authors' SimLBR code trained on our data, mean of 3 seeds
            for bb in ("sat", "web"):
                f = GRSL / f"simlbr_official_{bb}_metrics.json"
                if not f.exists(): continue
                m = json.load(open(f))["variants"]["lbr"]["mean_std_over_seeds"]
                extra.append(kv_line("SimLBR official (" + bb + ") AUC", [(n, m[k]['auc_real_vs_manipulated']['mean']) for k, n in (("overhead_test", "own test"), ("cross_fmow", "fmow"),
                    ("cross_rsfake", "rsfake"), ("cross_uw", "uw"), ("cross_dmaer", "dmaer")) if k in m]))
        except Exception:
            pass
        try:      # label-free transfer experiments (ZEROSHOT_EXTRAS_PROTOCOL.md): AUC of the frozen-backbone probe
            zx = json.load(open(GRSL / "zeroshot_extras_metrics.json"))["B_frozen_probe"]
            for ex in ("sat_orig", "full_s42"):
                extra.append(kv_line(f"frozen probe trained on own data ({ex}) AUC", [(n, zx[ex][k]['auc']) for k, n in (("own_test", "own test"), ("rsfake", "rsfake"), ("uw", "uw"), ("dmaer", "dmaer"),
                    ("gdd_crossgenerator", "gdd-cg"), ("gdd_crossdomain", "gdd-cd"))]))
        except Exception:
            pass
        nrob = len(list(GRSL.glob("robust_*_metrics.json"))); nov = len(list(GRSL.glob("overhead_*_metrics.json")))
        nf = len(list((HOME / "data/features/fewshot_cls384").glob("*/*.npy")))
        yes = lambda b: "[bold bright_green]✔[/]" if b else "[dark_orange]◌[/]"
        ex = Text.from_markup(
            f"robustness {nrob}/12   test two-class {nov}/2   figure {yes((GRSL / 'qualitative.png').exists())}   "
            f"SimLBR official {len(list(GRSL.glob('simlbr_official_*_metrics.json')))}/2   "
            f"significance re-scores {len(list(GRSL.glob('overhead_*_s4?_metrics.json')))}/7   "
            f"few-shot feature files {nf}   few-shot results {yes((GRSL / 'fewshot_metrics.json').exists())}   "
            f"few-shot GDD {yes((GRSL / 'fewshot_gdd_metrics.json').exists())}   few-shot seeds 43/44 {yes((GRSL / 'fewshot_seeds_metrics.json').exists())}   "
            f"intervals {yes((GRSL / 'cross_detail.json').exists())}   label-free extras {yes((GRSL / 'zeroshot_extras_metrics.json').exists())}   "
            f"supplement {yes((GRSL / 'supplement/SUPPLEMENT.pdf').exists())}   "
            f"analysis {yes((GRSL / 'analysis_extras.json').exists())}   table.csv {yes((GRSL / 'table.csv').exists())}")
        key = Text.assemble(("score colours:  ", "grey62"), ("≥0.90 ", sc(0.95)), ("≥0.80 ", sc(0.85)), ("≥0.65 ", sc(0.7)), ("≥0.55 ", sc(0.6)), ("<0.55", sc(0.5)))
        P = lambda title, *body: Panel(Group(*body), title=f"[bold {RES_COL}] ★ {title} ★ [/]", border_style=RES_COL, box=box.HEAVY, padding=(0, 1))
        res = [("Own test", [P("OWN TEST SET: finished runs by variant", *([summ] if summ else [Text("no finished run yet", style="dim")]), Text(""), ex, key)]),
               ("Zero-shot", [P("ZERO-SHOT ON OUTSIDE DATA", tb, *[e for e in extra if isinstance(e, Text)], Text(""), key)]),
               ("Few-shot", [P("FEW-SHOT ADAPTATION", *([e for e in extra if isinstance(e, Table)] or [Text("pending", style="dim")]), key)])]
    try:
        last = [l for l in (GRSL / "RUNLOG.md").read_text().splitlines() if l.startswith("|")][-1]
        log = Text("last log entry: " + last[2:150], style="dim", no_wrap=True, overflow="ellipsis")
    except Exception:
        log = Text("")
    allh = max(left_h, lap_left)
    head = Text.assemble((" GRSL ", "bold black on gold1"), (" experiment queue ", "bold white on dodger_blue3"), (f"  {datetime.now():%a %d %b %H:%M:%S}  ", "bright_cyan"),
                         ("■ workstation", WS_COL), "  ", ("■ laptop", LAP_COL),
                         (f"   all training done ≈ {(datetime.now()+timedelta(hours=allh)):%a %H:%M}" if ROLE != "laptop" else "", "bold bright_green"),
                         (f"  (in {int(allh)} h {int(allh % 1 * 60):02d} m)" if ROLE != "laptop" else "", "green3"),
                         )
    if ROLE != "laptop" and (HOME / "spacecraft/status.py").exists():      # second project: dataset downloads (its own repo)
        try:
            if str(HOME / "spacecraft") not in sys.path: sys.path.insert(0, str(HOME / "spacecraft"))
            import status as _space; space = _space.panel()
        except Exception:
            pass
    if ROLE == "laptop" and (HOME / ".spacecraft_status.json").exists():   # sent by the workstation (~/spacecraft/push_status.sh)
        try:
            sp = json.load(open(HOME / ".spacecraft_status.json")); t = Text(); age = time.time() - sp["t"]
            stl = {"done": ("✔", "bold bright_green"), "downloading": ("▶", "bold bright_cyan"), "failed": ("✖", "bold red1"),
                   "stopped": ("■", "dark_orange"), "queued": ("◌", "steel_blue1")}
            for name, have, gb, state, lic, what, rate in sp["rows"]:
                mark, sty = stl.get(state, ("?", "yellow")); t.append(f"{mark} {name}", style=sty)
                if state == "downloading": t.append(f" {have:.1f}/{gb:g} GB" + (f" {rate:.0f} MB/s" if rate else ""), style="bright_cyan")
                elif state == "done": t.append(f" {have:.0f} GB", style="green3")
                t.append("   ")
            nd = sum(r[3] == "done" for r in sp["rows"])
            space = (Panel(t, title=f"[bold spring_green1] ✦ SPACECRAFT PROJECT (on the workstation): dataset downloads  {nd}/{len(sp['rows'])} done  ·  "
                                            f"{sum(r[1] for r in sp['rows']):.0f} GB on disk  ·  {sp['free']} free ✦ [/]",
                                   subtitle=f"[grey62]updated {int(age // 60)} min ago[/]", border_style="spring_green1", box=box.HEAVY, padding=(0, 1)))
        except Exception:
            pass
    tabs = [("Runs", panels)] + res + ([("Spacecraft", [space])] if space is not None else [])
    if tab is None: return Group(head, *[x for _, v in tabs for x in v], log)
    global TAB
    TAB = tab % len(tabs); bar = Text(" ")
    for i, (name, _) in enumerate(tabs):
        bar.append(f" {i+1} {name} ", style="bold black on gold1" if i == TAB else "bold white on grey23"); bar.append(" ")
    bar.append("   keys 1-%d or ←/→ switch tabs   q quits (runs keep going)" % len(tabs), style="grey62")
    return Group(head, bar, *tabs[TAB][1], log)


def main():
    if "--once" in sys.argv or not sys.stdout.isatty():
        Console(force_terminal=True, width=int(os.environ.get("COLUMNS", 130))).print(render()); return
    born = os.path.getmtime(__file__); synced = 0
    import select, termios, tty
    global TAB
    fd = sys.stdin.fileno(); old = termios.tcgetattr(fd); tty.setcbreak(fd)      # single key presses, no Enter needed
    import atexit; atexit.register(lambda: termios.tcsetattr(fd, termios.TCSADRAIN, old))
    with Live(render(TAB), screen=True, auto_refresh=False) as live:
        while True:
            end = time.time() + 5
            while time.time() < end:                         # wait up to 5 s, react to a key at once
                if select.select([fd], [], [], max(0.0, end - time.time()))[0]:
                    k = os.read(fd, 8)
                    if k in (b"q", b"Q"): return
                    if k.isdigit() and k != b"0": TAB = int(k) - 1
                    elif k in (b"\x1b[C", b"\t", b"l", b" "): TAB += 1
                    elif k in (b"\x1b[D", b"h"): TAB -= 1
                    else: continue
                    live.update(render(TAB), refresh=True)
            if os.path.getmtime(__file__) != born:      # watch.py was updated: restart this view with the new code
                live.stop(); termios.tcsetattr(fd, termios.TCSADRAIN, old); os.execv(sys.executable, [sys.executable] + sys.argv)
            if ROLE == "laptop" and time.time() - synced > 600:   # laptop: pull the dashboard code and result files
                synced = time.time()                               # (never logs or RUNLOG.md, which a live run has open)
                subprocess.Popen("git fetch -q origin ojcs-experiments && git checkout -q origin/ojcs-experiments -- "
                                 "watch.py '*_metrics.json' table.csv", shell=True, cwd=GRSL,
                                 stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            live.update(render(TAB), refresh=True)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        pass
