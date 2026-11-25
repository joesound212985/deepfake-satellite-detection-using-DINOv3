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


def job_name(stem):
    m = re.match(r"robust_(full|web)_(jpeg|scale)([\d.]+)$", stem)
    if m: return f"Robustness: {m.group(1)} model, {'JPEG quality' if m.group(2) == 'jpeg' else 'down-scale'} {m.group(3)}"
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
        t.append("GPU "); t.append_text(gauge(g[2] / 100, 12, "green" if g[2] > 50 else "yellow")); t.append(f" {g[2]:3.0f}%  ")
        t.append(f"{g[3]:.0f}°C", style=tcol(g[3], 80, 87)); t.append(f"  {g[4]:.0f} W    ")
        t.append("VRAM "); t.append_text(gauge(g[0] / g[1], 10, "cyan")); t.append(f" {g[0]/1024:.1f}/{g[1]/1024:.0f} GB    ")
    else:
        t.append("GPU n/a    ", style="red")
    t.append("CPU "); t.append(f"{cpu}°C" if cpu is not None else "--", style=tcol(cpu, 85, 97) if cpu is not None else "dim")
    if extra: t.append("    " + extra, style="dim")
    return t


def row_table(rows, col):
    tb = Table(box=None, show_header=False, padding=(0, 1), expand=True)
    tb.add_column(width=1); tb.add_column(width=38, no_wrap=True, overflow="ellipsis"); tb.add_column(width=BAR_W, no_wrap=True)
    tb.add_column(width=4, justify="right"); tb.add_column(ratio=1, no_wrap=True, overflow="ellipsis")
    for st, label, frac, detail, sub in rows:
        if st == "done":
            tb.add_row(Text("✔", style="bold green"), Text(label, style="green"), gauge(1, BAR_W, "green"), "100%", Text(detail, style="green"))
        elif st == "run":
            tb.add_row(Text("▶", style=f"bold {col}"), Text(label, style=f"bold {col}"), gauge(frac, BAR_W, col), f"{frac*100:.0f}%", Text(detail, style="bold white"))
            if sub: tb.add_row("", "", Text(""), "", Text(sub, style="white"))
        elif st == "eval":
            tb.add_row(Text("◆", style="bold yellow"), Text(label, style="bold yellow"), gauge(frac, BAR_W, "yellow"), f"{frac*100:.0f}%", Text(detail, style="yellow"))
        elif st == "stop":
            tb.add_row(Text("✖", style="bold red"), Text(label, style="red"), gauge(0, BAR_W, col), "", Text(detail, style="red"))
        elif st == "unk":
            tb.add_row(Text("?", style="yellow"), Text(label, style="yellow"), gauge(0, BAR_W, col), "", Text(detail, style="yellow"))
        else:
            tb.add_row(Text("·", style="dim"), Text(label, style="dim"), gauge(0, BAR_W, col), "", Text(detail, style="dim"))
    return tb


def done_detail(m, tail):
    L = m["localization_tampered"]
    return f"acc {m['accuracy']:.4f}  F1 {m['macro_f1']:.4f}  IoU {L['mean_iou']:.4f}  {tail}"


def render():
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
                rem.append(("done", label, 1, f"acc {a[0]:.4f}  F1 {a[1]:.4f}  IoU {a[2]:.4f}  on laptop, not copied back", ""))
            elif lap and lap.get("tag") == tag:
                d = parse_log(lap["log"]); frac, r, sub = progress(d, hrs); lap_left += r / 3600
                rem.append(("run", label, frac, f"{sub}   ETA {fmt_td(r)}", val_line(d).strip() if d["val"] else ""))
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
            loc.append(("run", label, frac, f"{sub}   ETA {fmt_td(r)}", val_line(d).strip() if d["val"] else ""))
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
    panels = [Panel(Group(sensors(gpu_raw(), cpu_temp()), st, Text(""), row_table(loc, mycol), Text(""), foot),
                    title=f"[bold {mycol}] {me} [/]", border_style=mycol, box=box.HEAVY, padding=(0, 1))]

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
            ex = f"power: {lap.get('profile') or '?'}{', turbo off' if lap.get('turbo_off') == '1' else ''}   load {lap.get('load', '?')}"
            body = Group(sensors(g, lap.get("cpu"), ex), st2, Text(""), row_table(rem, LAP_COL), Text(""), foot2)
        else:
            body = Group(Text(" ○ NOT REACHABLE ", style="bold white on red"), Text("asleep, switched off, or Tailscale is down", style="red"),
                         Text(""), row_table(rem, LAP_COL))
        panels.append(Panel(body, title=f"[bold {LAP_COL}] LAPTOP (RTX 5090) [/]", border_style=LAP_COL, box=box.HEAVY, padding=(0, 1)))

    # ---- results (workstation only)
    if ROLE != "laptop":
        tb = Table(box=box.SIMPLE_HEAD, padding=(0, 1), expand=False, header_style="bold")
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
                    cells.append(Text(f"{auc:.3f} / {acc:.3f}", style="green" if auc >= 0.8 else ("yellow" if auc >= 0.65 else "red")))
                else:
                    cells.append(Text("pending", style="dim"))
            tb.add_row(label, *cells)
        extra = []
        try:      # few-shot: AUC on the held-out half, linear probe on frozen CLS features
            fs = json.load(open(GRSL / "fewshot_metrics.json"))["datasets"]
            ft = Table(box=box.SIMPLE_HEAD, padding=(0, 1), expand=False, header_style="bold")
            ft.add_column("few-shot AUC, full_s42 features (labelled images per class)")
            for h in ("zero-shot", "4", "16", "64", "256", "whole pool"): ft.add_column(h, justify="center")
            col = lambda a: "green" if a >= 0.8 else ("yellow" if a >= 0.65 else "red")
            for key, label in (("fmow", "fmow-fake-small"), ("rsfake", "RSFAKE-1M subset"), ("uw", "UW fake satellite"), ("dmaer", "DM-AER (size-matched)")):
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
                extra.append(Text("SimLBR official (" + bb + ") AUC:  " + "   ".join(
                    f"{n} {m[k]['auc_real_vs_manipulated']['mean']:.3f}" for k, n in (("overhead_test", "own test"), ("cross_fmow", "fmow"),
                    ("cross_rsfake", "rsfake"), ("cross_uw", "uw"), ("cross_dmaer", "dmaer")) if k in m)))
        except Exception:
            pass
        nrob = len(list(GRSL.glob("robust_*_metrics.json"))); nov = len(list(GRSL.glob("overhead_*_metrics.json")))
        nf = len(list((HOME / "data/features/fewshot_cls384").glob("*/*.npy")))
        yes = lambda b: "[green]✔[/]" if b else "[dim]·[/]"
        ex = Text.from_markup(
            f"robustness {nrob}/12   test two-class {nov}/2   figure {yes((GRSL / 'qualitative.png').exists())}   "
            f"SimLBR official {len(list(GRSL.glob('simlbr_official_*_metrics.json')))}/2   "
            f"significance re-scores {len(list(GRSL.glob('overhead_*_s4?_metrics.json')))}/7   "
            f"few-shot features {nf}/16   few-shot results {yes((GRSL / 'fewshot_metrics.json').exists())}   "
            f"analysis {yes((GRSL / 'analysis_extras.json').exists())}   table.csv {yes((GRSL / 'table.csv').exists())}")
        panels.append(Panel(Group(tb, *extra, ex), title="[bold] RESULTS SO FAR [/]", border_style="grey50", padding=(0, 1)))
    try:
        last = [l for l in (GRSL / "RUNLOG.md").read_text().splitlines() if l.startswith("|")][-1]
        log = Text("last log entry: " + last[2:150], style="dim", no_wrap=True, overflow="ellipsis")
    except Exception:
        log = Text("")
    head = Text.assemble(("GRSL experiment queue", "bold"), f"   {datetime.now():%a %d %b %H:%M:%S}   ",
                         ("■ workstation", WS_COL), "  ", ("■ laptop", LAP_COL), ("   Ctrl-C closes this view; the runs keep going", "dim"))
    return Group(head, *panels, log)


def main():
    if "--once" in sys.argv or not sys.stdout.isatty():
        Console(force_terminal=True, width=int(os.environ.get("COLUMNS", 130))).print(render()); return
    with Live(render(), screen=True, auto_refresh=False) as live:
        while True:
            time.sleep(5)
            live.update(render(), refresh=True)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        pass
