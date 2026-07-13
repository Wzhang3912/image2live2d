#!/usr/bin/env python3
"""Calibrate the dynamics score against a REAL Live2D model (``.moc3`` + ``.physics3.json``).

This is the bridge from a professional rig to our calibrator. It uses the Cubism core
(``cubism_core.py``) to pull every drawable's rest-pose mesh out of a real ``.moc3``, then compares our
geometric "needs physics?" verdict for each drawable against the artist's ground truth.

Ground truth — *which drawables the rigger gave physics* — is read from the **names** of the physics
output parameters, not by perturbing them: an auto-physics rig names each output after the part it
drives (``Param_Angle_Rotation_3_hair_left`` → the ``hair_left*`` drawables; ``ParamSkirtPhysicsA`` →
``skirt*``). Perturbing instead labels ~everything, because a deformer rotation cascades to its
children. Roles (for the sway-eligibility gate) are likewise inferred from drawable names. Both
heuristics are model-naming-dependent and best-effort — this measures a real rig, but treat the exact
number as indicative, not gospel.

FINDING (Akari, VTS): the score under-detects badly on a production rig. Even after merging segments
into parts, the physics parts (hair strands, tie, skirt) read ``free_edge≈0`` because the rig LAYERS
dozens of shade/overlay/back meshes behind every part, so the free-edge "opens into void" cue (our
decisive signal, weight 0.45) never fires — something always fills the gap. ``--align`` collapses the
coincident depth layers (align_pro_model, IoU merge) and lifts recall a little (0.08 → 0.17), but it
PLATEAUS: the dominant backing is spatially-*offset* layers (back-hair behind front-hair) that are
legitimately different regions and can't be safely merged by geometry alone. Conclusion: the free-edge
cue is *representation-specific* — it works on the SPARSE, non-overlapping parts our decomposer emits
from a flat image, not on a hand-rig's dense layering, and geometry-only alignment can't fully recover
it. A robust cross-representation score would down-weight free-edge when scene density is high and lean
on cantilever/slenderness. Do NOT transfer thresholds from a real drawable dump; this ships the harness,
the alignment transform, and the evidence for that limit.

    python tools/calibrate_moc3.py path/to/model.moc3 path/to/model.physics3.json [--verbose]

The model is read locally and never committed (it's proprietary). Point --core at your Cubism core if
it isn't auto-found (see cubism_core.py).
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from align_pro_model import merge_overlapping, rasterize_cells  # noqa: E402
from cubism_core import Model  # noqa: E402
from image2live2d.core.structure.calibrate import best_thresholds, evaluate  # noqa: E402
from image2live2d.core.structure.dynamics import analyze_meshes  # noqa: E402
from image2live2d.core.types import Layer, LayerStack  # noqa: E402
from image2live2d.irr.schema import Mesh, SemanticRole  # noqa: E402


def _camel_to_snake(s: str) -> str:
    return re.sub(r"(?<!^)(?=[A-Z])", "_", s).lower()


def physics_target_stems(param_ids) -> set[str]:
    """The part-name stems a set of physics *output* params target, decoded from their names."""
    stems: set[str] = set()
    for p in param_ids:
        m = re.match(r"Param_Angle_Rotation_\d+_(.+)$", p)
        if m:
            stems.add(m.group(1).lower())
            continue
        m = re.match(r"Param(.+?)Physics", p) or re.match(r"(.+?)Physics", p)
        if m:
            stems.add(_camel_to_snake(m.group(1)))
    return stems


# breast physics params drive the drawables the artist named oppai / under_boob.
_STEM_ALIASES = {"breast": ("oppai", "under_boob")}

# Drawable-name -> role, most specific first. Facial/skin/limb -> a non-sway-eligible role (the dynamics
# gate makes those rigid regardless of shape); hair/cloth -> eligible so geometry decides.
_ROLE_HINTS: tuple[tuple[str, SemanticRole], ...] = (
    ("eye", SemanticRole.eye_l), ("brow", SemanticRole.eyebrow_l), ("blush", SemanticRole.blush),
    ("mouth", SemanticRole.mouth), ("lip", SemanticRole.mouth), ("fang", SemanticRole.mouth),
    ("tongue", SemanticRole.mouth), ("tear", SemanticRole.face_base), ("sign", SemanticRole.face_base),
    ("nose", SemanticRole.nose), ("ear", SemanticRole.ear_l), ("face", SemanticRole.face_base),
    ("head", SemanticRole.face_base), ("neck", SemanticRole.neck),
    ("hair", SemanticRole.hair_side), ("uh", SemanticRole.hair_side),   # 'uh' = under-hair strands
    ("tie", SemanticRole.clothing), ("skirt", SemanticRole.clothing), ("shirt", SemanticRole.clothing),
    ("collar", SemanticRole.clothing), ("sleeve", SemanticRole.clothing),
    ("oppai", SemanticRole.clothing), ("boob", SemanticRole.clothing), ("belly", SemanticRole.clothing),
    ("under", SemanticRole.clothing), ("clip", SemanticRole.accessory),
    ("arm", SemanticRole.arm_l), ("finger", SemanticRole.hand_l), ("thumb", SemanticRole.hand_l),
    ("leg", SemanticRole.leg_l), ("knee", SemanticRole.leg_l), ("shoe", SemanticRole.leg_l),
    ("f1", SemanticRole.hand_l), ("f2", SemanticRole.hand_l), ("f3", SemanticRole.hand_l),
    ("shadow", SemanticRole.torso), ("belt", SemanticRole.clothing),
)


def group_key(did: str) -> str:
    """Collapse a pro model's fine drawables into one *part* the way our pipeline decomposes — a hair
    strand split into ``hair_left2``…``hair_left8`` (+ ``_shade``) is one ``hair_left`` part. Our
    dynamics score reads a whole-part silhouette (free edge, cantilever, slenderness); scoring a single
    mid-strand segment (glued to its neighbours) is meaningless, so we merge before scoring."""
    k = re.sub(r"\d+$", "", did)
    k = re.sub(r"_(shade|light|main|overlay|inner|ex)$", "", k)
    k = re.sub(r"\d+$", "", k)
    return k or did


def _role_of(did: str):
    low = did.lower()
    for key, role in _ROLE_HINTS:
        if key in low:
            return role
    return None


def _has_physics(did: str, stems: set[str]) -> bool:
    low = did.lower()
    for stem in stems:
        targets = (stem, *_STEM_ALIASES.get(stem, ()))
        if any(low.startswith(t) or t in low for t in targets):
            return True
    return False


def _dominant_role(members):
    """The role of the first member (largest first) that has an inferable role, else None."""
    for m in members:
        r = _role_of(m)
        if r is not None:
            return r
    return None


def build_corpus(moc3: Path, physics3: Path, core: str | None, *, align: bool = False, res: int = 64):
    """(labeled dynamics, skipped-count). Each label is (PartDynamics, pro_has_physics). With
    ``align``, depth-layered parts that heavily overlap are merged into one spatial region first (see
    align_pro_model) so the free-edge cue isn't defeated by a rig's stacked shade/back layers."""
    model = Model(str(moc3), core)
    draws = model.drawables()
    out_params = {o["Destination"]["Id"]
                  for s in json.loads(physics3.read_text()).get("PhysicsSettings", []) or []
                  for o in s.get("Output", []) or []}
    stems = physics_target_stems(out_params)

    # Merge fine drawables into parts by name (our pipeline's granularity), concatenating vertices and
    # offsetting each drawable's triangle indices into the merged vertex list.
    parts: dict[str, dict] = {}
    skipped = 0
    for i, d in enumerate(draws):
        if len(d.vertices) < 3 or not d.triangles:
            skipped += 1
            continue
        key = group_key(d.id)
        part = parts.setdefault(key, {"verts": [], "tris": [], "order": i})
        base = len(part["verts"])
        part["verts"].extend(d.vertices)
        part["tris"].extend((a + base, b + base, c + base) for a, b, c in d.triangles)

    # Normalise all part vertices into a shared [0,1] frame (y already up in Cubism space) so the free-
    # edge detector — and the overlap rasteriser — see one canvas.
    allv = [v for p in parts.values() for v in p["verts"]]
    xs = [x for x, _ in allv]
    ys = [y for _, y in allv]
    minx, maxx, miny, maxy = min(xs), max(xs), min(ys), max(ys)
    sx, sy = max(maxx - minx, 1e-6), max(maxy - miny, 1e-6)
    for p in parts.values():
        p["nverts"] = [((x - minx) / sx, (y - miny) / sy) for x, y in p["verts"]]

    # Optionally collapse depth layers: parts whose footprints heavily overlap become one region.
    if align:
        cells = {k: rasterize_cells(p["nverts"], p["tris"], res) for k, p in parts.items()}
        groups = merge_overlapping(cells, thresh=0.5)
    else:
        groups = [[k] for k in parts]

    layers, meshes, want = [], [], {}
    for grp in groups:
        members = sorted(grp, key=lambda k: -len(parts[k]["nverts"]))
        role = _dominant_role(members)
        if role is None:
            skipped += 1
            continue
        rep = members[0]
        nverts, tris = [], []
        for k in grp:
            base = len(nverts)
            nverts.extend(parts[k]["nverts"])
            tris.extend((a + base, b + base, c + base) for a, b, c in parts[k]["tris"])
        meshes.append(Mesh(part_id=rep, vertices=nverts, uvs=[(0.0, 0.0)] * len(nverts), triangles=tris))
        layers.append(Layer(id=rep, semantic_role=role, texture_path=Path(f"{rep}.png"),
                           draw_order=min(parts[k]["order"] for k in grp), width=0, height=0))
        want[rep] = any(_has_physics(k, stems) for k in grp)   # physics if ANY layer was rigged
    stack = LayerStack(layers=layers, canvas_width=1, canvas_height=1)
    labeled = [(dyn, want[dyn.part_id]) for dyn in analyze_meshes(stack, meshes)]
    return labeled, skipped


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Calibrate the dynamics score against a real .moc3.")
    ap.add_argument("moc3", type=Path)
    ap.add_argument("physics3", type=Path)
    ap.add_argument("--core", default=None)
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument("--align", action="store_true",
                    help="collapse depth-layered parts into flat spatial regions before scoring")
    ap.add_argument("--res", type=int, default=64, help="overlap-rasteriser resolution (with --align)")
    args = ap.parse_args(argv)

    labeled, skipped = build_corpus(args.moc3, args.physics3, args.core,
                                    align=args.align, res=args.res)
    if not labeled:
        print("no roleable drawables found", file=sys.stderr)
        return 1

    if args.verbose:
        from image2live2d.core.structure.calibrate import predicted_physics
        for d, truth in sorted(labeled, key=lambda x: (x[1], -x[0].score)):
            pred = predicted_physics(d)
            tag = "ok" if pred == truth else ("MISS(fn)" if truth else "OVER(fp)")
            if tag != "ok" or truth:
                print(f"  {d.part_id:26} score={d.score:.2f} free={d.free_edge_ratio:.2f} "
                      f"{d.verdict.value:8} pro={'phys' if truth else 'rigid':5} {tag}")
        print()

    n_phys = sum(1 for _, t in labeled if t)
    m = evaluate(labeled)
    best = best_thresholds(labeled)
    b = best.metrics
    print(f"{args.moc3.name}: {len(labeled)} roleable drawables ({skipped} skipped), "
          f"{n_phys} pro-physics")
    print(f"current defaults:  precision={m.precision:.3f} recall={m.recall:.3f} f1={m.f1:.3f} "
          f"acc={m.accuracy:.3f} (tp={m.tp} fp={m.fp} fn={m.fn} tn={m.tn})")
    print(f"best thresholds:   gentle_t={best.gentle_t:.2f} free_edge_floor={best.free_edge_floor:.2f} "
          f"-> f1={b.f1:.3f} (tp={b.tp} fp={b.fp} fn={b.fn} tn={b.tn})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
