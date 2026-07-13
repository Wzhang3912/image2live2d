#!/usr/bin/env python3
"""P0 — cross-backend physics **feel parity** oracle (headless).

The same IRR ``PhysicsRig`` (a driver->output pendulum with mass/drag/length) is integrated by two very
different runtimes: Cubism (``physics3.json`` — Mobility/Delay/Acceleration vertices) and nijilive
(``SimplePhysics`` — gravity/length/angle-damping). "Feel parity" means the *same* rig swings the same
in both: same natural frequency, damping and amplitude. There is no shared runtime to A/B headlessly,
so this tool gives the agent a numeric + visual pre-check before a human eyeballs the two apps:

1. **Cubism side, calibrated to real pro rigs** — it reads a real ``.physics3.json`` (Hiyori/Akari, if
   present locally) and reports the artist's Mobility/Delay/Acceleration/weight ranges, then flags any of
   our emitted vertices that fall OUTSIDE those ranges (the reliable Cubism check — we can't run Cubism's
   integrator headlessly, but we can prove our emission sits in the regime real artists use).
2. **nijilive side, simulated** — nijilive's SimplePhysics is a damped driven pendulum, which we
   integrate directly: for each output param we report natural frequency, damping ratio, peak overshoot
   and settle time, and plot the swing response to a head-turn step (``--plot feel.png``).
3. A **parity table** putting both backends' per-param constants side by side so a divergence (e.g. one
   backend swinging 3x more, or a much faster settle) is obvious.

    python tools/feel_parity.py [--plot out.png] [--physics3 path/to/real.physics3.json]

Real models are read locally, never committed. This is an oracle for tuning physics3.py + puppet.py so
their FEEL matches; the final subjective match is still confirmed by a human in nijigenerate + Cubism.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from image2live2d.core import decompose  # noqa: E402
from image2live2d.pipeline import rig_from_stack  # noqa: E402
from image2live2d.samples import make_sample_fullbody  # noqa: E402

# Keep these mirrored with the emitters (physics3.py / puppet.py); this tool reads them to compare.
_NIJI_GRAVITY, _NIJI_PPM, _NIJI_LEN_PX = 9.8, 1000.0, 120.0
_NIJI_OUT_SCALE, _NIJI_ANCHOR_PX = 3.0, 60.0
_CUB_LEN_UNITS = 12.0

_VTS_MODELS = ("~/Library/Application Support/Steam/steamapps/common/VTube Studio/VTubeStudio.app/"
               "Contents/Resources/Data/StreamingAssets/Live2DModels/{0}_vts/{0}.physics3.json")
_DEFAULT_REAL = [_VTS_MODELS.format(m) for m in ("hiyori", "akari")]   # union both -> the real regime


# --------------------------------------------------------------------------------------------------
# nijilive SimplePhysics — a damped pendulum with a moving pivot, integrated directly.
# --------------------------------------------------------------------------------------------------
def _niji_pendulum(length: float, drag: float, *, driver, dt=1 / 240, secs=6.0):
    """Integrate the small-angle driven damped pendulum nijilive's SimplePhysics runs.

    Pivot moves horizontally by ``driver(t)`` (normalized -1..1) * anchor_px / length_px; the bob lags.
    Returns (t[], theta[], omega_n, zeta, output[]). ``output`` = theta * output_scale (the param sway).
    """
    L_m = max(0.1, length) * _NIJI_LEN_PX / _NIJI_PPM        # pendulum length in metres
    omega_n = math.sqrt(_NIJI_GRAVITY / L_m)                 # natural angular frequency
    ang_damp = min(1.5, max(0.1, 0.4 + drag))                # puppet.py angle_damping
    c = ang_damp * omega_n                                   # velocity damping ~ zeta*2*omega
    zeta = c / (2.0 * omega_n)
    n = int(secs / dt)
    th = thv = 0.0
    a_prev = a_prev2 = 0.0
    ts, ths, outs = [], [], []
    anchor_gain = _NIJI_ANCHOR_PX / (_NIJI_LEN_PX * max(0.1, length))   # pivot travel / length
    for i in range(n):
        t = i * dt
        a = driver(t) * anchor_gain                          # pivot x (in length units)
        a_acc = (a - 2 * a_prev + a_prev2) / (dt * dt)       # pivot acceleration excites the pendulum
        a_prev2, a_prev = a_prev, a
        # small-angle driven damped pendulum: th'' = -w^2 th - c th' - a''/L(=1 in length units)
        thacc = -(omega_n ** 2) * th - c * thv - a_acc
        thv += thacc * dt
        th += thv * dt
        ts.append(t); ths.append(th); outs.append(th * _NIJI_OUT_SCALE)
    return ts, ths, omega_n, zeta, outs


def _response_metrics(ts, ys):
    """Peak |value|, settle time (last time |y| leaves +-5% of the peak), and dominant frequency."""
    peak = max((abs(y) for y in ys), default=0.0)
    if peak < 1e-9:
        return 0.0, 0.0, 0.0
    thresh = 0.05 * peak
    settle = ts[-1]
    for t, y in zip(reversed(ts), reversed(ys)):
        if abs(y) > thresh:
            settle = t
            break
    # dominant frequency from zero-crossings of the (mean-removed) tail
    m = sum(ys) / len(ys)
    crossings = sum(1 for a, b in zip(ys, ys[1:]) if (a - m) * (b - m) < 0)
    freq = crossings / (2.0 * (ts[-1] - ts[0])) if ts[-1] > ts[0] else 0.0
    return peak, settle, freq


# --------------------------------------------------------------------------------------------------
# Cubism side: read a real physics3.json and check our emission sits in the artist regime.
# --------------------------------------------------------------------------------------------------
def _real_ranges(path: Path):
    doc = json.loads(path.read_text())
    mob, delay, accel, inw, outw, outsc = [], [], [], [], [], []
    for s in doc.get("PhysicsSettings", []):
        for v in s.get("Vertices", []):
            mob.append(v["Mobility"]); delay.append(v["Delay"]); accel.append(v["Acceleration"])
        for i in s.get("Input", []):
            inw.append(i["Weight"])
        for o in s.get("Output", []):
            outw.append(o["Weight"]); outsc.append(o.get("Scale", 1))
    rng = lambda x: (min(x), max(x)) if x else (0.0, 0.0)  # noqa: E731
    return {"Mobility": rng(mob), "Delay": rng(delay), "Acceleration": rng(accel),
            "InputWeight": rng(inw), "OutputWeight": rng(outw), "OutputScale": rng(outsc)}


def _cub_vertex(mass, drag, length):
    """The emitted Cubism tip vertex — read straight from physics3.py so this can't drift from it."""
    from image2live2d.backends.live2d.physics3 import _vertices
    tip = _vertices(mass, drag, length)[1]                # [0]=root, [1]=swinging tip
    return {"Mobility": tip["Mobility"], "Delay": tip["Delay"],
            "Acceleration": tip["Acceleration"], "tipY": tip["Position"]["Y"]}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Cross-backend physics feel-parity oracle.")
    ap.add_argument("--plot", type=Path, default=None, help="write the nijilive swing plot here (PNG)")
    ap.add_argument("--physics3", type=Path, default=None, help="a real .physics3.json to calibrate vs")
    args = ap.parse_args(argv)

    d = Path(tempfile.mkdtemp())
    rig = rig_from_stack(decompose.from_layer_dir(make_sample_fullbody(d)), name="ref")

    # Cubism regime check: union the ranges across every real model found (the true artist regime).
    paths = [args.physics3] if args.physics3 else [Path(c).expanduser() for c in _DEFAULT_REAL]
    found = [p for p in paths if p and p.exists()]
    real = None
    if found:
        per = [_real_ranges(p) for p in found]
        real = {k: (min(r[k][0] for r in per), max(r[k][1] for r in per)) for k in per[0]}
        print(f"Cubism regime (union of {', '.join(p.stem for p in found)}): "
              + "  ".join(f"{k} {lo:.2f}-{hi:.2f}" for k, (lo, hi) in real.items()))
    else:
        print("Cubism regime: no real physics3.json found (pass --physics3) — skipping range check")

    print(f"\n{'output param':16} | IRR L/m/drag | Cubism mob/delay/accel/tipY | "
          f"niji f(Hz) zeta peak settle(s)")
    step = lambda t: 0.0 if t < 0.2 else 1.0  # a head-turn step at t=0.2s  # noqa: E731
    curves = []
    for ph in rig.physics:
        cv = _cub_vertex(ph.mass, ph.drag, ph.length)
        ts, ths, wn, zeta, outs = _niji_pendulum(ph.length, ph.drag, driver=step)
        peak, settle, freq = _response_metrics(ts, outs)
        flags = ""
        if real:
            for key, val in (("Mobility", cv["Mobility"]), ("Delay", cv["Delay"]),
                             ("Acceleration", cv["Acceleration"])):
                lo, hi = real[key]
                if not (lo - 1e-6 <= val <= hi + 1e-6):
                    flags += f" !{key}={val:.2f}∉[{lo:.2f},{hi:.2f}]"
        print(f"{ph.output_param:16} | {ph.length:4.2f} {ph.mass:4.2f} {ph.drag:4.2f} | "
              f"{cv['Mobility']:4.2f} {cv['Delay']:4.2f} {cv['Acceleration']:4.2f} {cv['tipY']:5.1f} | "
              f"{freq:5.2f} {zeta:4.2f} {peak:5.2f} {settle:5.2f}{flags}")
        curves.append((ph.output_param, ts, outs))

    if args.plot:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(9, 5))
        for name, ts, outs in curves:
            ax.plot(ts, outs, label=name, lw=1.4)
        ax.axvline(0.2, color="k", ls=":", lw=0.8, label="head-turn step")
        ax.set_xlabel("seconds"); ax.set_ylabel("output param sway (nijilive sim)")
        ax.set_title("nijilive SimplePhysics swing response to a head-turn step (feel-parity oracle)")
        ax.legend(fontsize=7, ncol=2); ax.grid(alpha=0.3)
        fig.tight_layout(); fig.savefig(args.plot, dpi=110)
        print(f"\nwrote {args.plot}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
