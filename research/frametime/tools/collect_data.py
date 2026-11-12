#!/usr/bin/env python3
"""Collect frame-time data with headless bot sessions (usage: collect_data.sh --help).

Plans (--plan, comma-separated, run in order within the --max-minutes / --max-mb caps):
  bulk     one pinned worker per CPU in --cpus, random sessions back to back (seed, stress level,
           loop mode, quality policy, length), as in phase 1
  solo     the same random sessions, but one game at a time on the first CPU of --cpus
  matched  the same fixed session specs run twice: one at a time (solo), then all at once on
           --cpus (parallel); measures whether running several games at once distorts timings
  levers   paired single-lever sweeps: per stress level, one fixed game at L0 and with each lever
           lowered on its own (emulate, solo); gives each lever's marginal cost
  vhw      "virtual hardware" (LIT_REVIEW_HW 3.4): the same games under co-runners on other cores
           (bandwidth hog, cache pirate) and on a P-core vs an E-core, one at a time

Before sessions run, a hardware fingerprint (spark-hw-v1, ~3 s) is measured once per
(CPU, co-runner) and cached in <out>/fingerprints/; every session header embeds its fingerprint.
Whole sessions are assigned to train/val/test; matched/lever/vhw groups share one split.
MANIFEST.json is rewritten after every session. Refuses to run on battery power.
While running, research/frametime/data/.collecting holds this PID (other agents avoid heavy work).
"""
import argparse
import datetime as dt
import hashlib
import json
import math
import os
import random
import shutil
import subprocess
import sys
import threading
import time

HERE = os.path.dirname(os.path.abspath(__file__))
FT = os.path.dirname(HERE)
ROOT = os.path.dirname(os.path.dirname(FT))
sys.path.insert(0, os.path.join(FT, "train"))
from framelog import read_log  # noqa: E402

RECORD = 232  # bytes per frame (engine/framelog.hpp)
BUDGET_MS = 1000 / 60
MARKER = os.path.join(FT, "data", ".collecting")

# Stress presets: ranges each session samples from. S0 is the unmodified game.
STRESS = {
    "S0": dict(weight=0.14, out=(1.0, 1.0), spawn=(1.0, 1.0), burst=(1.0, 1.0), contend_p=0.0),
    "S1": dict(weight=0.20, out=(1.25, 1.6), spawn=(1.0, 1.5), burst=(1.0, 2.0), contend_p=0.15),
    "S2": dict(weight=0.24, out=(1.6, 2.1), spawn=(1.5, 2.5), burst=(2.0, 3.5), contend_p=0.2),
    "S3": dict(weight=0.24, out=(2.1, 2.6), spawn=(2.0, 3.5), burst=(3.0, 5.0), contend_p=0.5),
    "S4": dict(weight=0.18, out=(2.6, 3.2), spawn=(3.0, 5.0), burst=(4.0, 7.0), contend_p=0.7),
}
MID = {k: dict(out=sum(v["out"]) / 2, spawn=sum(v["spawn"]) / 2, burst=sum(v["burst"]) / 2) for k, v in STRESS.items()}
POLICIES = [("random", 0.50), ("fixed", 0.22), ("reactive", 0.18), ("paired", 0.10)]
SPLIT_CYCLE = ["train"] * 14 + ["val"] * 3 + ["test"] * 3  # 70/15/15 by session/group, per stress level
LEVELS = (4, 4, 4, 2)  # quality levels per lever: scale, particles, glow, aa


# ---------------------------------------------------------------- machine state

def power_source():
    """'AC', 'battery' or 'unknown' (servers; same rule as the engine's powerSource())."""
    for name in ("ADP1", "ADP0", "AC", "ACAD", "AC0"):
        base = f"/sys/class/power_supply/{name}"
        try:
            if open(base + "/type").read().strip() != "Mains":
                continue
            return {"1": "AC", "0": "battery"}.get(open(base + "/online").read().strip(), "unknown")
        except OSError:
            continue
    return "unknown"


def core_type(cpu):
    """'P'/'E' from the hybrid-CPU sysfs lists (i5-13500H: P = 0-7, E = 8-15), 'uniform' otherwise."""
    def parse(path):
        try:
            out = set()
            for part in open(path).read().strip().split(","):
                a, _, b = part.partition("-")
                out.update(range(int(a), int(b or a) + 1))
            return out
        except (OSError, ValueError):
            return None
    p, e = parse("/sys/devices/cpu_core/cpus"), parse("/sys/devices/cpu_atom/cpus")
    if p is None and e is None:
        return "uniform"
    return "P" if cpu in (p or ()) else "E" if cpu in (e or ()) else "unknown"


def busy_others(sample_s=2.0, ignore=("code", "chrome", "Xorg", "cinnamon", "Discord", "WeChatAppEx", "pipewire",
                                      "pulseaudio", "gnome-shell", "firefox")):
    """Processes (not ours, not ignored desktop apps) using > 0.5 core over the sample window."""
    def snap():
        out = {}
        for pid in os.listdir("/proc"):
            if not pid.isdigit():
                continue
            try:
                f = open(f"/proc/{pid}/stat").read()
                name = f[f.index("(") + 1:f.rindex(")")]
                parts = f[f.rindex(")") + 2:].split()
                out[int(pid)] = (name, int(parts[11]) + int(parts[12]))
            except (OSError, ValueError, IndexError):
                pass
        return out
    a = snap()
    time.sleep(sample_s)
    b = snap()
    hz = os.sysconf("SC_CLK_TCK")
    me = os.getpid()
    busy = []
    for pid, (name, t) in b.items():
        if pid == me or pid not in a or name in ignore or name.startswith("spark_warden"):
            continue
        cores = (t - a[pid][1]) / hz / sample_s
        if cores > 0.5:
            busy.append((name, pid, round(cores, 2)))
    return busy


def wait_quiet(max_wait_s):
    """Wait (up to max_wait_s) until no other heavy process runs. Returns what was still busy."""
    t0 = time.time()
    while True:
        busy = busy_others()
        if not busy or time.time() - t0 > max_wait_s:
            if busy:
                print(f"warning: still busy after {max_wait_s:.0f}s: {busy}; collecting anyway (tagged)", flush=True)
            return busy
        print(f"waiting for other heavy processes to finish: {busy}", flush=True)
        time.sleep(15)


# ---------------------------------------------------------------- session specs

def pick(rng, items):
    r, acc = rng.random(), 0.0
    for name, w in items:
        acc += w
        if r < acc:
            return name
    return items[-1][0]


def random_quality(rng):
    """Levels (scale, particles, glow, aa): full quality 45% of the time per lever, else uniform."""
    return [0 if rng.random() < 0.45 else rng.randrange(n) for n in LEVELS]


def random_spec(rng, loop_weights):
    stress = pick(rng, [(k, v["weight"]) for k, v in STRESS.items()])
    s = STRESS[stress]
    u = lambda lo_hi: round(rng.uniform(*lo_hi), 3)
    contend = None
    if rng.random() < s["contend_p"]:
        contend = f"1:{round(rng.uniform(1.5, 8), 2)}:{round(rng.uniform(5, 40), 2)}"
    return dict(stress=stress, out_scale=u(s["out"]), spawn_mult=u(s["spawn"]), burst_mult=u(s["burst"]),
                contend=contend, loop=pick(rng, loop_weights), policy=pick(rng, POLICIES),
                steps=int(math.exp(rng.uniform(math.log(1800), math.log(18000)))), seed=rng.randrange(1, 2**31),
                quality=random_quality(rng), corunner="")


def fixed_spec(stress, seed, loop, steps, quality, policy="fixed", corunner=""):
    m = MID[stress]
    return dict(stress=stress, out_scale=round(m["out"], 3), spawn_mult=round(m["spawn"], 3),
                burst_mult=round(m["burst"], 3), contend=None, loop=loop, policy=policy, steps=steps, seed=seed,
                quality=list(quality), corunner=corunner)


# ---------------------------------------------------------------- collector

class Collector:
    def __init__(self, args):
        self.args = args
        self.out = os.path.abspath(args.out)
        self.sess_dir = os.path.join(self.out, "sessions")
        self.fp_dir = os.path.join(self.out, "fingerprints")
        os.makedirs(self.sess_dir, exist_ok=True)
        os.makedirs(self.fp_dir, exist_ok=True)
        self.manifest_path = os.path.join(self.out, "MANIFEST.json")
        self.lock = threading.Lock()
        self.manifest = self.load_manifest()
        self.audit()
        ids = [int(s["id"][1:]) for s in self.manifest["sessions"]]
        ids += [int(f[1:6]) for f in os.listdir(self.sess_dir) if f.startswith("s") and f[1:6].isdigit()]
        # Also skip ids used by moved-aside files, so a new session never shares an id with an orphan.
        orphan_dir = os.path.join(self.out, "orphans")
        if os.path.isdir(orphan_dir):
            ids += [int(f[1:6]) for f in os.listdir(orphan_dir) if f.startswith("s") and f[1:6].isdigit()]
        self.id_counter = max(ids, default=0)
        self.split_counter = {}
        for s in self.manifest["sessions"]:
            self.split_counter[s["stress"]] = self.split_counter.get(s["stress"], 0) + 1
        self.group_split = {s["group"]: s["split"] for s in self.manifest["sessions"] if s.get("group")}
        self.deadline = time.time() + args.max_minutes * 60
        self.bytes_cap = int(args.max_mb * 1024 * 1024)
        self.rng = random.Random(args.plan_seed)
        self.stop = False
        self.reserved = 0
        self.run_id = f"{args.phase}-{dt.datetime.now().strftime('%H%M%S')}"
        self.run_start_bytes = self.total_bytes()
        self.fp_files = {}
        self.busy_at_start = []

    # -- manifest
    def load_manifest(self):
        if os.path.exists(self.manifest_path):
            with open(self.manifest_path) as f:
                return json.load(f)
        return {"dataset": self.args.dataset, "format": "spark-framelog", "format_version": 2, "record_size": RECORD,
                "created_utc": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
                "budget_ms": BUDGET_MS, "split_by": "session (matched/lever/vhw groups share a split)",
                "stress_levels": STRESS, "machine": None, "runs": [], "sessions": [], "totals": {},
                "fingerprints": {}}

    def audit(self):
        """Keep only manifest sessions whose file matches the entry; move unreferenced session files to orphans/"""
        keep, dropped = [], []
        for e in self.manifest["sessions"]:
            try:
                h, rec = read_log(os.path.join(self.out, e["file"]))
                ok = h["session"]["tags"].get("session") == e["id"] and len(rec) == e["frames"]
            except Exception:  # noqa: BLE001
                ok = False
            (keep if ok else dropped).append(e)
        referenced = {os.path.basename(e["file"]) for e in keep}
        orphans = [f for f in os.listdir(self.sess_dir) if f not in referenced]
        orphan_dir = os.path.join(self.out, "orphans")  # unreferenced files are moved here, never deleted
        os.makedirs(orphan_dir, exist_ok=True)
        for f in orphans:
            dst = os.path.join(orphan_dir, f)
            n = 1
            while os.path.exists(dst):  # never overwrite an earlier orphan
                dst = os.path.join(orphan_dir, f"{f}.{n}")
                n += 1
            shutil.move(os.path.join(self.sess_dir, f), dst)
        if dropped or orphans:
            print(f"audit: dropped {[e['id'] for e in dropped]}, moved to orphans/ {orphans}", flush=True)
            self.manifest["sessions"] = keep
            self.manifest.setdefault("audit_log", []).append(
                dict(utc=dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
                     dropped=[e["id"] for e in dropped], moved_to_orphans=sorted(orphans)))
            self.write_manifest()

    def total_bytes(self):
        return sum(s["bytes"] for s in self.manifest["sessions"])

    def write_manifest(self):
        m = self.manifest
        sessions = m["sessions"]
        tot = {"sessions": len(sessions), "frames": sum(s["frames"] for s in sessions),
               "bytes": sum(s["bytes"] for s in sessions)}
        for key in ("split", "stress", "loop", "plan", "core_type", "condition"):
            d = tot.setdefault(f"by_{key}", {})
            for s in sessions:
                g = d.setdefault(str(s.get(key)), {"sessions": 0, "frames": 0, "bytes": 0})
                g["sessions"] += 1
                g["frames"] += s["frames"]
                g["bytes"] += s["bytes"]
        m["totals"] = tot
        tmp = self.manifest_path + ".tmp"
        with open(tmp, "w") as f:
            json.dump(m, f, indent=1)
        os.replace(tmp, self.manifest_path)

    def next_id(self):
        while True:
            self.id_counter += 1
            sid = f"s{self.id_counter:05d}"
            if not os.path.exists(os.path.join(self.sess_dir, sid + ".sfl")):
                return sid

    def split_for(self, spec):
        g = spec.get("group")
        if g and g in self.group_split:
            return self.group_split[g]
        i = self.split_counter.get(spec["stress"], 0)
        self.split_counter[spec["stress"]] = i + 1
        split = SPLIT_CYCLE[(i * 7 + 3) % len(SPLIT_CYCLE)]
        if g:
            self.group_split[g] = split
        return split

    # -- fingerprints
    def fingerprint(self, cpu, corunner=""):
        """Measure (once per run) the fingerprint for a pinned CPU + co-runner, cached in <out>/fingerprints."""
        key = (cpu, corunner)
        if key in self.fp_files:
            return self.fp_files[key]
        name = f"fp-cpu{cpu}-{(corunner or 'none').replace(':', '_').replace('@', '_').replace(',', '_')}.json"
        path = os.path.join(self.fp_dir, name)
        cmd = (["taskset", "-c", str(cpu)] if cpu is not None else []) + \
              [self.args.game, "--fingerprint", path, "--fp-cache", os.path.join(self.fp_dir, "cache")]
        if corunner:
            cmd += ["--corunner", corunner]
        t0 = time.time()
        p = subprocess.run(cmd, capture_output=True, text=True)
        if p.returncode != 0:
            print(f"fingerprint cpu{cpu} {corunner or ''} failed: {p.stderr.strip()[:200]}", flush=True)
            self.fp_files[key] = None
            return None
        fpj = json.load(open(path))
        self.fp_files[key] = path
        with self.lock:
            self.manifest.setdefault("fingerprints", {})[name] = dict(
                cpu=cpu, corunner=corunner or None, core_type=fpj["state"]["core_type"], power=fpj["state"]["power"],
                key=fpj["key"], measured_utc=dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
                seconds=round(time.time() - t0, 2),
                summary={k: round(fpj["fp"][k], 5) for k in (
                    "ghz_start", "frame_ms_s1", "frame_ms_s32", "step_ms_normal", "lat_l3_ns", "lat_dram_ns",
                    "bw_dram_gbs", "noise_excess_p99_us", "sleep_over_p99_us", "raster_calib", "cliff_ratio")})
            self.write_manifest()
        print(f"fingerprint cpu{cpu} ({fpj['state']['core_type']}) {corunner or 'no co-runner'}: "
              f"{fpj['fp']['ghz_start']:.2f} GHz, frame {fpj['fp']['frame_ms_s1']:.3f} ms, "
              f"DRAM {fpj['fp']['lat_dram_ns']:.0f} ns {fpj['fp']['bw_dram_gbs']:.1f} GB/s "
              f"({time.time() - t0:.1f}s)", flush=True)
        return path

    # -- one session
    def admit(self, spec):
        """Reserve disk budget for a session; False when caps are reached."""
        with self.lock:
            if self.stop or time.time() > self.deadline - 15:
                return False
            if power_source() == "battery" and not self.args.allow_battery:
                print("on battery power: stopping collection", flush=True)
                self.stop = True
                return False
            headroom = self.bytes_cap - (self.total_bytes() - self.run_start_bytes) - self.reserved
            if headroom // RECORD < 1500:
                self.stop = True
                return False
            spec["steps"] = int(min(spec["steps"], headroom // RECORD * 0.98))
            self.reserved += spec["steps"] * RECORD
            return True

    def run_session(self, spec, cpu, condition, concurrent):
        if not self.admit(spec):
            return False
        with self.lock:
            sid = self.next_id()
            split = self.split_for(spec)
        try:
            path = os.path.join(self.sess_dir, sid + ".sfl")
            fpf = self.fingerprint(cpu, spec.get("corunner", "")) if not self.args.no_fingerprint else None
            remaining = self.deadline - time.time()
            cmd = (["taskset", "-c", str(cpu)] if cpu is not None else []) + [
                self.args.game, "--headless", "--bot", "--mute", "--loop", spec["loop"], "--seed", str(spec["seed"]),
                "--frames", str(spec["steps"]), "--max-seconds", f"{max(5.0, remaining - 5):.1f}", "--log", path,
                "--out-scale", str(spec["out_scale"]), "--burst-mult", str(spec["burst_mult"]),
                "--spawn-mult", str(spec["spawn_mult"]), "--quality", ",".join(map(str, spec["quality"])),
                "--ctl", spec["policy"], "--research-seed", str(spec["seed"] ^ 0x5A5A),
                "--fp", fpf or "none",
                "--tag", f"session={sid}", "--tag", f"stress={spec['stress']}", "--tag", f"phase={self.args.phase}",
                "--tag", f"plan={spec['plan']}", "--tag", f"group={spec.get('group') or ''}",
                "--tag", f"condition={condition}", "--tag", f"cpu={cpu}", "--tag", f"workers={concurrent}",
                "--tag", f"core_type={core_type(cpu) if cpu is not None else 'unpinned'}",
                "--tag", f"power={power_source()}"]
            if spec.get("contend"):
                cmd += ["--contend", spec["contend"]]
            if spec.get("corunner"):
                cmd += ["--corunner", spec["corunner"]]
            t0 = time.time()
            p = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True)
            if p.returncode not in (0, 1):  # 1 = bot never reached wave 2 (short sessions); still valid data
                print(f"[{sid}] exit {p.returncode}: {p.stderr.strip()[:200]}", flush=True)
            if os.path.exists(path):
                self.record(sid, path, spec, cpu, split, condition, concurrent, time.time() - t0, p.returncode)
        finally:
            with self.lock:
                self.reserved -= spec["steps"] * RECORD
        return True

    def record(self, sid, path, spec, cpu, split, condition, concurrent, wall, rc):
        import numpy as np
        try:
            header, rec = read_log(path)
        except Exception as e:  # noqa: BLE001
            print(f"[{sid}] unreadable log ({e}); dropped", flush=True)
            os.remove(path)
            return
        power_start, power_now = header["machine"].get("power", "unknown"), power_source()
        if len(rec) < 60 or (not self.args.allow_battery and "battery" in (power_start, power_now)):
            os.remove(path)
            print(f"[{sid}] dropped ({len(rec)} frames, power {power_start}->{power_now})", flush=True)
            return
        t = rec["t_total_us"] / 1000.0
        fpj = header["machine"].get("fingerprint")
        entry = dict(id=sid, file=os.path.relpath(path, self.out), split=split, run=self.run_id,
                     phase=self.args.phase, plan=spec["plan"], group=spec.get("group"), condition=condition,
                     frames=int(len(rec)), steps=int(rec["n_steps"].sum()), bytes=os.path.getsize(path),
                     stress=spec["stress"], loop=spec["loop"], policy=spec["policy"], seed=spec["seed"],
                     out_scale=spec["out_scale"], spawn_mult=spec["spawn_mult"], burst_mult=spec["burst_mult"],
                     contend=spec.get("contend"), corunner=spec.get("corunner") or None,
                     quality_start=spec["quality"], cpu=cpu,
                     core_type=core_type(cpu) if cpu is not None else "unpinned", concurrent_workers=concurrent,
                     pair=spec.get("pair"), wall_s=round(wall, 1), exit_code=rc,
                     ended_early=bool(int(rec["n_steps"].sum()) < spec["steps"]),
                     quality_changes=int((rec["flags"] & 2 != 0).sum()),
                     t_total_ms={"p50": round(float(np.percentile(t, 50)), 3),
                                 "p95": round(float(np.percentile(t, 95)), 3),
                                 "p99": round(float(np.percentile(t, 99)), 3), "max": round(float(t.max()), 3),
                                 "frac_over_budget": round(float((t > BUDGET_MS).mean()), 4)},
                     cpu_mhz_p50=int(np.percentile(rec["cpu_mhz"], 50)), power=power_start,
                     platform_profile=header["machine"].get("platform_profile", "unknown"),
                     hostname=header["machine"].get("hostname"),
                     fingerprint_key=fpj["key"] if fpj else None,
                     busy_others_at_start=[b[0] for b in self.busy_at_start] or None)
        with self.lock:
            if self.manifest.get("machine") is None:
                self.manifest["machine"] = {k: v for k, v in header["machine"].items() if k != "fingerprint"}
            self.manifest["sessions"].append(entry)
            self.write_manifest()
        print(f"[{sid}] {spec['plan']:7s} {condition:8s} cpu{cpu} {spec['stress']} {spec['loop']:7s} "
              f"{spec['policy']:8s} q={''.join(map(str, spec['quality']))} {spec.get('corunner') or '':12s} {split:5s} "
              f"{len(rec):6d} fr p50 {entry['t_total_ms']['p50']:6.2f} p99 {entry['t_total_ms']['p99']:6.2f} "
              f"over {entry['t_total_ms']['frac_over_budget']:.3f} ({wall:.0f}s)", flush=True)

    # -- runners
    def run_pool(self, cpus, make_specs, condition="parallel"):
        """One pinned worker per CPU, pulling specs from make_specs(cpu) until the caps."""
        def worker(cpu):
            for spec in make_specs(cpu):
                if self.stop or not self.run_session(spec, cpu, condition, len(cpus)):
                    return
        threads = [threading.Thread(target=worker, args=(c,)) for c in cpus]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

    def run_sequential(self, jobs, condition="solo"):
        for spec, cpu in jobs:
            if self.stop or not self.run_session(spec, cpu, condition, 1):
                return

    def run_simultaneous(self, jobs, condition="parallel"):
        threads = [threading.Thread(target=self.run_session, args=(spec, cpu, condition, len(jobs)))
                   for spec, cpu in jobs]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

    # -- plans
    def plan_random(self, plan, cpus, minutes):
        end = min(self.deadline, time.time() + minutes * 60)
        loop_w = [("emulate", 1 - self.args.paced_frac), ("paced", self.args.paced_frac)]
        rng_lock = threading.Lock()

        def specs(cpu):
            while time.time() < end - 20:
                with rng_lock:
                    s = random_spec(self.rng, loop_w)
                    s["plan"] = plan
                    if s["policy"] == "paired":  # same game twice (emulate) at two fixed qualities
                        q2 = random_quality(self.rng)
                        while q2 == s["quality"]:
                            q2 = random_quality(self.rng)
                        g = f"pair-{s['seed']}"
                        s.update(loop="emulate", steps=min(s["steps"], 6000), policy="fixed", group=g, pair=g)
                        s2 = dict(s, quality=q2)
                yield s
                if s.get("pair"):
                    yield s2
        saved = self.deadline
        self.deadline = end
        if plan == "solo":
            self.run_pool(cpus[:1], specs, "solo")
        else:
            self.run_pool(cpus, specs, "parallel")
        self.deadline = saved

    def plan_matched(self, cpus):
        """K fixed specs (one per CPU): each run alone, then all at the same time."""
        rng = random.Random(self.args.plan_seed + 1)
        jobs = []
        for i, cpu in enumerate(cpus):
            stress = ["S1", "S2", "S3", "S4", "S2", "S3"][i % 6]
            loop = "paced" if i % 2 == 0 else "emulate"
            g = f"matched-{self.run_id}-{i}"
            s = fixed_spec(stress, rng.randrange(1, 2**31), loop, self.args.matched_steps, (0, 0, 0, 0))
            s.update(plan="matched", group=g)
            jobs.append((s, cpu))
        self.run_sequential([(dict(s), c) for s, c in jobs], "solo")
        self.run_simultaneous([(dict(s), c) for s, c in jobs], "parallel")

    def plan_levers(self, cpu):
        """Per stress level: one fixed game at L0 and with each lever lowered on its own (paired)."""
        rng = random.Random(self.args.plan_seed + 2)
        jobs = []
        for stress in ("S1", "S2", "S3", "S4"):
            seed = rng.randrange(1, 2**31)
            g = f"levers-{self.run_id}-{stress}"
            qs = [(0, 0, 0, 0)] + [(l, 0, 0, 0) for l in (1, 2, 3)] + [(0, l, 0, 0) for l in (1, 2, 3)] + \
                 [(0, 0, l, 0) for l in (1, 2, 3)] + [(0, 0, 0, 1)]
            for q in qs:
                s = fixed_spec(stress, seed, "emulate", self.args.lever_steps, q)
                s.update(plan="levers", group=g)
                jobs.append((s, cpu))
        self.run_sequential(jobs, "solo")

    def plan_vhw(self, cpu_p, cpu_e, corunners):
        """The same fixed games under each co-runner on a P-core, and once on an E-core, one at a time."""
        rng = random.Random(self.args.plan_seed + 3)
        jobs = []
        for stress, loop in (("S2", "emulate"), ("S3", "paced"), ("S4", "emulate")):
            seed = rng.randrange(1, 2**31)
            g = f"vhw-{self.run_id}-{stress}"
            for co in [""] + corunners:
                s = fixed_spec(stress, seed, loop, self.args.vhw_steps, (0, 0, 0, 0), corunner=co)
                s.update(plan="vhw", group=g)
                jobs.append((s, cpu_p))
            if cpu_e is not None:
                s = fixed_spec(stress, seed, loop, self.args.vhw_steps, (0, 0, 0, 0))
                s.update(plan="vhw", group=g)
                jobs.append((s, cpu_e))
        self.run_sequential(jobs, "solo")


def resolve_cpus(spec):
    """'0,2,4' or 'auto:N' (N CPUs spread evenly over this process's affinity, e.g. one per CCX on Delta)."""
    if spec.startswith("auto:"):
        n = int(spec[5:])
        allowed = sorted(os.sched_getaffinity(0))
        step = max(1, len(allowed) // n)
        return allowed[::step][:n]
    return [int(c) for c in spec.split(",") if c != ""]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--game", default=os.path.join(ROOT, "build", "spark_warden"))
    ap.add_argument("--out", default=os.path.join(FT, "data-v2"))
    ap.add_argument("--dataset", default="laptop-v2")
    ap.add_argument("--plan", default="bulk", help="comma list of: bulk, solo, matched, levers, vhw")
    ap.add_argument("--max-minutes", type=float, default=30, help="wall-time cap for this invocation")
    ap.add_argument("--max-mb", type=float, default=300, help="disk cap for the sessions written by this invocation")
    ap.add_argument("--bulk-minutes", type=float, default=1e9, help="time for the bulk plan (rest of the cap)")
    ap.add_argument("--solo-minutes", type=float, default=10)
    ap.add_argument("--cpus", default="0,2,4,6,8,12", help="worker CPUs: list or auto:N")
    ap.add_argument("--solo-cpu", type=int, default=None, help="CPU for solo/levers/vhw (default first of --cpus)")
    ap.add_argument("--ecore-cpu", type=int, default=None, help="E-core for the vhw P-vs-E comparison")
    ap.add_argument("--corunners", default="bw:1@4,bw:2@4,6,cache:8@4,cache:12@4",
                    help="vhw co-runner specs, separated by '+' or ',' between specs (see --corunner in the game)")
    ap.add_argument("--paced-frac", type=float, default=0.6, help="fraction of random sessions with real sleeping")
    ap.add_argument("--matched-steps", type=int, default=3000)
    ap.add_argument("--lever-steps", type=int, default=1500)
    ap.add_argument("--vhw-steps", type=int, default=2400)
    ap.add_argument("--plan-seed", type=int, default=int(time.time()) & 0xFFFFFF)
    ap.add_argument("--phase", default="A")
    ap.add_argument("--allow-battery", action="store_true")
    ap.add_argument("--no-fingerprint", action="store_true")
    ap.add_argument("--quiet-wait", type=float, default=600, help="max seconds to wait for other heavy processes")
    args = ap.parse_args()
    if not os.path.exists(args.game):
        sys.exit(f"build the game first: {args.game} not found (make)")
    if power_source() == "battery" and not args.allow_battery:
        sys.exit("refusing to collect: on battery power (battery changes CPU power limits; --allow-battery overrides)")
    cpus = resolve_cpus(args.cpus)
    solo_cpu = args.solo_cpu if args.solo_cpu is not None else cpus[0]
    c = Collector(args)
    os.makedirs(os.path.dirname(MARKER), exist_ok=True)
    with open(MARKER, "w") as f:
        f.write(f"pid={os.getpid()} start={dt.datetime.now().isoformat(timespec='seconds')} out={c.out}\n")
    try:
        c.busy_at_start = wait_quiet(args.quiet_wait)
        t0 = time.time()
        # Corunner specs are "kind:arg@cpus"; split on '+' if given, else on the boundary before a kind.
        raw = args.corunners.replace("+", "|")
        for kind in ("bw:", "cache:"):
            raw = raw.replace("," + kind, "|" + kind)
        corunners = [x for x in raw.split("|") if x]
        for plan in args.plan.split(","):
            if c.stop or time.time() > c.deadline - 30:
                break
            print(f"== plan {plan} ({(c.deadline - time.time()) / 60:.1f} min left)", flush=True)
            if plan == "bulk":
                c.plan_random("bulk", cpus, args.bulk_minutes)
            elif plan == "solo":
                c.plan_random("solo", [solo_cpu], args.solo_minutes)
            elif plan == "matched":
                c.plan_matched(cpus)
            elif plan == "levers":
                c.plan_levers(solo_cpu)
            elif plan == "vhw":
                c.plan_vhw(solo_cpu, args.ecore_cpu, corunners)
            else:
                sys.exit(f"unknown plan {plan}")
        c.manifest["runs"].append(dict(id=c.run_id, phase=args.phase, plans=args.plan, cpus=cpus,
                                       max_minutes=args.max_minutes, max_mb=args.max_mb, plan_seed=args.plan_seed,
                                       wall_s=round(time.time() - t0, 1),
                                       bytes=c.total_bytes() - c.run_start_bytes,
                                       busy_others_at_start=c.busy_at_start))
        c.write_manifest()
    finally:
        if os.path.exists(MARKER):
            os.remove(MARKER)
    tot = c.manifest["totals"]
    print(f"\ndataset {c.manifest['dataset']}: {tot['sessions']} sessions, {tot['frames']:,} frames, "
          f"{tot['bytes'] / 2**20:.1f} MB -> {c.manifest_path}")


if __name__ == "__main__":
    main()
