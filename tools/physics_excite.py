"""Does the motion we ship actually MOVE the physics we wired? Answered headlessly.

Wiring a pendulum to a driver is not the same as the driver moving it. We spent four PRs tuning hair,
accessory and cloth pendulums against a real pro rig, emitted the model, and watched it in Cubism
Viewer — and the idle never touched ``ParamAngleX``, so five of eight chains had a constant input and a
provably frozen output. The physics was correct and connected to nothing. Nothing caught it, because
the only thing that *could* catch it was a human squinting at a render.

This is that check, as a number. It reimplements Cubism's own physics update — ``CubismPhysics::
Evaluate`` / ``UpdateParticles``: normalize the input parameter, translate the root particle, integrate
the particle chain under gravity, read the tip's angle back out through ``Scale`` — and runs it against
the **emitted** ``.physics3.json`` and the **emitted** ``.motion3.json``. For every clip it reports how
far each hair/cloth output parameter actually travels, as a fraction of its range.

Two things it is worth knowing before trusting the output:

* **A pendulum responds to acceleration, not position.** Move a head slowly and the hair simply hangs
  straight down from wherever the head now is: the relative angle the output measures stays near zero.
  A 12-degree *slow* drift moves ``ParamHairFront`` by ~1% of range; the same head with one quick
  glance in it reaches ~10%. If a clip reads ``weak``, the fix is usually a faster move, not a bigger
  one.
* **+Y is DOWN in Cubism physics space.** The SDK's gravity is ``RadianToDirection(angle) = (sin, cos)``
  — ``(0, +1)`` at rest — and particles hang at *positive* Y (both real models on hand do: Hiyori's
  11-particle chain reaches Y=150). Point gravity the other way and every pendulum becomes an inverted
  one, slams into its stops, and reports a saturated 100%-of-range swing that looks like a pass.

Self-check: ``--flat`` pins every driver at its default, which must report exactly ``0.000`` / FROZEN.
If it doesn't, the simulator is lying and nothing else it says can be believed. Run it first.

Usage:
    PYTHONPATH=src python tools/physics_excite.py <bundle_dir> [--clip NAME] [--flat]
    PYTHONPATH=src python tools/physics_excite.py <bundle_dir>            # every clip, summarised
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

# See the module docstring: +Y is down, and gravity at rest points along it.
GRAVITY_AT_REST = (0.0, 1.0)
AIR_RESISTANCE = 5.0          # CubismPhysics::AirResistance
DT = 1.0 / 60.0

# Below this share of its own range, an output parameter's swing will not read on screen.
WEAK_FRAC = 0.05


# --------------------------------------------------------------------------------------------------
# motion3 curve sampling
# --------------------------------------------------------------------------------------------------
def _curve(motion: dict, pid: str) -> list[tuple[float, float]] | None:
    """(time, value) control points of a motion3 curve. Bezier handles are dropped — we sample at 60fps
    and these curves are slow, so the straight-line reading is within a fraction of a degree."""
    for c in motion["Curves"]:
        if c["Id"] != pid:
            continue
        seg = c["Segments"]
        pts, i = [(seg[0], seg[1])], 2
        while i < len(seg):
            step = 7 if int(seg[i]) == 1 else 3      # bezier carries 3 control points; the rest carry 1
            pts.append((seg[i + step - 2], seg[i + step - 1]))
            i += step
        return pts
    return None


def _sample(pts: list[tuple[float, float]], t: float) -> float:
    if t <= pts[0][0]:
        return pts[0][1]
    for (t0, v0), (t1, v1) in zip(pts, pts[1:]):
        if t <= t1:
            return v0 + (t - t0) / max(t1 - t0, 1e-9) * (v1 - v0)
    return pts[-1][1]


# --------------------------------------------------------------------------------------------------
# the Cubism physics model
# --------------------------------------------------------------------------------------------------
def _normalize(value, pmin, pmax, pdef, nmin, nmax, ndef) -> float:
    """Cubism's ``NormalizeParameterValue``: piecewise-linear about the default, into the physics
    setting's own normalization window."""
    value = min(max(value, min(pmin, pmax)), max(pmin, pmax))
    if value < pdef:
        span = pdef - min(pmin, pmax)
        return ndef + (value - pdef) / span * (ndef - nmin) if span else ndef
    span = max(pmin, pmax) - pdef
    return ndef + (value - pdef) / span * (nmax - ndef) if span else ndef


def _dir_to_radian(frm: tuple[float, float], to: tuple[float, float]) -> float:
    ret = math.atan2(to[1], to[0]) - math.atan2(frm[1], frm[0])
    while ret < -math.pi:
        ret += math.pi * 2
    while ret > math.pi:
        ret -= math.pi * 2
    return ret


class Chain:
    """One ``PhysicsSetting``: its particle chain, and its input/output parameter wiring."""

    def __init__(self, setting: dict, ranges: dict[str, tuple[float, float, float]]):
        self.inputs = setting["Input"]
        self.out = setting["Output"][0]
        self.norm = setting["Normalization"]
        self.verts = setting["Vertices"]
        self.ranges = ranges
        self.pos = [(v["Position"]["X"], v["Position"]["Y"]) for v in self.verts]
        self.vel = [(0.0, 0.0)] * len(self.verts)
        self.last_grav = [GRAVITY_AT_REST] * len(self.verts)

    @property
    def driver(self) -> str:
        return self.inputs[0]["Source"]["Id"]

    @property
    def output(self) -> str:
        return self.out["Destination"]["Id"]

    def step(self, params: dict[str, float]) -> float:
        """Advance one frame; return the output parameter's value."""
        # --- input: parameters -> root-particle translation + gravity angle ------------------------
        tx = ty = angle = 0.0
        for inp in self.inputs:
            pid = inp["Source"]["Id"]
            pmin, pmax, pdef = self.ranges[pid]
            kind = inp["Type"]
            band = self.norm["Angle"] if kind == "Angle" else self.norm["Position"]
            n = _normalize(params.get(pid, pdef), pmin, pmax, pdef,
                           band["Minimum"], band["Maximum"], band["Default"]) * inp["Weight"] / 100.0
            if kind == "X":
                tx += n
            elif kind == "Y":
                ty += n
            else:                       # "Angle" — a roll tips gravity rather than moving the anchor
                angle += n

        # --- particles: gravity + delay/mobility, each re-constrained to its own radius -------------
        self.pos[0] = (tx, ty)
        rad = math.radians(angle)
        grav = (math.sin(rad), math.cos(rad))
        for i in range(1, len(self.verts)):
            v = self.verts[i]
            last = self.pos[i]
            delay = v["Delay"] * DT * 30.0
            fx, fy = grav[0] * v["Acceleration"], grav[1] * v["Acceleration"]
            dx = self.pos[i][0] - self.pos[i - 1][0]
            dy = self.pos[i][1] - self.pos[i - 1][1]
            r = _dir_to_radian(self.last_grav[i], grav) / AIR_RESISTANCE
            rx = math.cos(r) * dx - dy * math.sin(r)
            ry = math.sin(r) * dx + dy * math.cos(r)
            px = self.pos[i - 1][0] + rx + self.vel[i][0] * delay + fx * delay * delay
            py = self.pos[i - 1][1] + ry + self.vel[i][1] * delay + fy * delay * delay
            ux, uy = px - self.pos[i - 1][0], py - self.pos[i - 1][1]
            n = math.hypot(ux, uy) or 1e-9
            self.pos[i] = (self.pos[i - 1][0] + ux / n * v["Radius"],
                           self.pos[i - 1][1] + uy / n * v["Radius"])
            if delay:
                self.vel[i] = ((self.pos[i][0] - last[0]) / delay * v["Mobility"],
                               (self.pos[i][1] - last[1]) / delay * v["Mobility"])
            self.last_grav[i] = grav

        # --- output: the tip's angle off the previous segment, through Scale, clamped to range -------
        idx = self.out["VertexIndex"]
        trans = (self.pos[idx][0] - self.pos[idx - 1][0], self.pos[idx][1] - self.pos[idx - 1][1])
        # measured against the previous segment, or — for the first particle, which has none — against
        # "down". (The SDK gets there by negating Options.Gravity (0,-1), landing on (0,+1).)
        parent = ((self.pos[idx - 1][0] - self.pos[idx - 2][0],
                   self.pos[idx - 1][1] - self.pos[idx - 2][1]) if idx >= 2 else GRAVITY_AT_REST)
        value = _dir_to_radian(parent, trans) * self.out["Scale"]
        pmin, pmax, _ = self.ranges[self.output]
        return min(max(value, pmin), pmax)


# --------------------------------------------------------------------------------------------------
# the report
# --------------------------------------------------------------------------------------------------
def param_ranges(moc3: Path) -> dict[str, tuple[float, float, float]]:
    """(min, max, default) per parameter, read from the ``.moc3`` through the native Cubism core — the
    same numbers the real runtime clamps against."""
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from cubism_core import Model

    m = Model(str(moc3))
    return {pid: (float(m._mins[i]), float(m._maxes[i]), float(m._defaults[i]))
            for i, pid in enumerate(m.param_ids)}


def excite(bundle: Path, clip: str, ranges, *, flat: bool = False) -> list[dict]:
    """Run every physics chain against one clip. Returns a row per chain."""
    phys = json.loads((bundle / "model.physics3.json").read_text())
    motion = json.loads((bundle / f"model.{clip}.motion3.json").read_text())
    dur = motion["Meta"]["Duration"]

    rows = []
    for setting in phys["PhysicsSettings"]:
        chain = Chain(setting, ranges)
        curves = {i["Source"]["Id"]: _curve(motion, i["Source"]["Id"]) for i in chain.inputs}

        lo = hi = dlo = dhi = None
        t = 0.0
        while t < dur * 3:                       # 3 loops: let the transient die, measure the last one
            params = {pid: (ranges[pid][2] if (flat or pts is None) else _sample(pts, t % dur))
                      for pid, pts in curves.items()}
            v = chain.step(params)
            if t >= dur * 2:                     # steady state only
                lo, hi = (v, v) if lo is None else (min(lo, v), max(hi, v))
                d = params[chain.driver]
                dlo, dhi = (d, d) if dlo is None else (min(dlo, d), max(dhi, d))
            t += DT

        pmin, pmax, _ = ranges[chain.output]
        swing = hi - lo
        rows.append({
            "output": chain.output, "driver": chain.driver,
            "driver_swing": dhi - dlo, "swing": swing, "frac": swing / (pmax - pmin),
        })
    return rows


def _verdict(row: dict) -> str:
    if row["swing"] < 1e-6:
        return "FROZEN"
    return "weak" if row["frac"] < WEAK_FRAC else "swings"


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    flat = "--flat" in argv
    clip = None
    if "--clip" in argv:
        i = argv.index("--clip")
        clip, argv = argv[i + 1], argv[:i] + argv[i + 2:]
    args = [a for a in argv if not a.startswith("--")]
    if not args:
        print(__doc__)
        return 2
    bundle = Path(args[0])

    ranges = param_ranges(bundle / "model.moc3")
    model3 = json.loads((bundle / "model.model3.json").read_text())
    clips = [clip] if clip else sorted(
        f["File"].split(".")[-3] for g in model3["FileReferences"]["Motions"].values() for f in g
    )

    detail = clip is not None
    if flat:
        print("SELF-CHECK: every driver pinned at its default. Everything below MUST read FROZEN —\n"
              "if it does not, the simulator is broken and none of its other numbers mean anything.\n")

    if detail:
        for name in clips:
            rows = excite(bundle, name, ranges, flat=flat)
            print(f"{name}")
            print(f"  {'output':<16} {'driver':<16} {'driver':>9} {'output swing':>19}   verdict")
            for r in rows:
                print(f"  {r['output']:<16} {r['driver']:<16} {r['driver_swing']:>6.1f}deg "
                      f"{r['swing']:>8.3f} ({r['frac']:>5.1%})   {_verdict(r)}")
        return 0

    # Summary: one line per clip. A chain frozen in a clip that does not drive it is correct and
    # expected — head clips must NOT move the skirt, and that isolation is the point of the drive sheet.
    # The clip that has to move everything is the idle, because it is the one that runs by default.
    print(f"{'clip':<14} {'chains moving':>13}  {'biggest swing':>13}   notes")
    bad = 0
    for name in clips:
        rows = excite(bundle, name, ranges, flat=flat)
        live = [r for r in rows if _verdict(r) == "swings"]
        top = max((r["frac"] for r in rows), default=0.0)
        note = ""
        if name == "idle" and not flat:
            dead = [r["output"] for r in rows if _verdict(r) != "swings"]
            if dead:
                note = f"!! idle leaves {len(dead)} chain(s) unexcited: {', '.join(dead)}"
                bad += len(dead)
            else:
                note = "idle excites every chain"
        print(f"{name:<14} {len(live):>6}/{len(rows):<6} {top:>12.1%}   {note}")
    print(f"\n({WEAK_FRAC:.0%} of range is the floor for 'moving'. A pendulum answers to acceleration, "
          f"not position:\n a slow drift barely swings it — see the module docstring.)")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
