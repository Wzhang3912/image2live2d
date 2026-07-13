"""Stage 4b — Structure. The **dynamics-score detector**: decide *which parts need physics* and how.

This is the root-cause fix for universal auto-rigging (see docs/AUTORIG_PHYSICS_UNIVERSAL_PLAN.md).
Today a part's ``SemanticRole`` (its *identity*, from the decomposer) doubles as its *motion recipe*,
so only the handful of enumerated roles get bespoke motion and two parts sharing a role (twin-tails)
can't move independently. Here we sever identity from motion: a part's dynamics are decided from
**measurable physical properties of its geometry**, exactly the cues a human Live2D rigger reads by
eye when deciding "does this element deserve physics?" Role survives only as a weak *prior*.

The judgment is reduced to detectable signals, combined into a continuous **dynamics score** per part:

  * **free-edge ratio** (the decisive cue) — a part swings only if it has a boundary that opens into
    *empty space* (a free hem/tip), not one glued to another part. A cheek/eye/collar has no free
    edge; a bang, ribbon end or skirt hem does. Detected by walking the part's alpha boundary and
    asking, at each exposed edge, whether *another part* fills the gap (attached) or it's void (free).
  * **cantilever / overhang** — does mass hang *past its attachment* under gravity? (bangs below the
    hairline, a ponytail past the crown, a skirt below the waist). Fully-supported parts don't swing.
  * **slenderness** — long thin things swing (strands); compact things don't (mesh/occupancy PCA).
  * **material prior** — a weak nudge from role + name hints (``bow``/``ribbon``/``tail``/``skirt``…).

Two further signals from the plan — **attachment fraction** (how pinned the boundary is) and
**depth/layer isolation** (how much of the footprint is the part's own vs backed by others) — are also
computed per part (a complete feature vector for calibration / a future learned classifier) but are
*not* combined into the score: measured against a real pro rig they carry no discriminative power, and
depth-isolation is in fact inverted on hand rigs (rigged parts are the more layered ones). See the note
on ``_score_one``.

A high score → gets physics; a middle band → gentle motion only; low → rigid. The threshold is biased
toward **restraint** (over-rigging reads as cheap jitter), with the free-edge detector as a safety net
so an obviously-hanging free edge is never left dead. Dynamic parts are also given a **physical class**
(``strand``/``sheet``/``jiggle``) that later picks the generic motion synthesizer.

Design mirrors ``core.landmark``: a **pure core** (``score_dynamics``) takes per-part alpha *samplers*
and is fully testable without Pillow or any ML extras; a thin **Pillow wrapper** (``analyze_stack``)
reads each layer's PNG. Coordinate space matches the rest of the pipeline (model space, **y up**,
canvas ``[0, 1]``) so anchors compose directly with mesh vertices and landmarks.

Scope note: this module produces the per-part *dynamics verdict + class + anchor*. Assembling these
into the full ``RigGraph`` (parenting, material constants) and rewiring ``author_rig`` to consume it
is the next P1 increment — kept separate so this lands with **zero change to current output**. The
score weights/thresholds below are the knobs P1b calibrates against real pro models.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum
from typing import Callable

from ..types import LayerStack
from ...irr.schema import SemanticRole, Vec2

# (u, v) in [0, 1], u left->right, v top->bottom (v down) over the FULL canvas -> alpha 0..255. Same
# convention as core.landmark / core.mesh. Full-canvas (not per-bbox) so parts share one coordinate
# frame and the free-edge detector can ask whether *another* part fills a given gap.
AlphaSampler = Callable[[float, float], int]

DEFAULT_SAMPLES = 96          # NxN probe grid over the whole canvas per part
DEFAULT_ALPHA_THRESHOLD = 8   # below this a texel counts as transparent

# --- Dynamics score weights (sum to 1) --------------------------------------------------------------
# free-edge and cantilever dominate (they are what a human actually reads); slenderness refines;
# material is only a nudge. These four numbers are the primary target of P1b corpus calibration.
_W_FREE_EDGE = 0.45
_W_CANTILEVER = 0.30
_W_SLENDER = 0.15
_W_MATERIAL = 0.10

# --- Density-aware free-edge trust ------------------------------------------------------------------
# The free-edge cue assumes a SPARSE layout (a boundary that opens into empty space reads as free — the
# case for our decomposer's non-overlapping parts). On a densely LAYERED representation (a hand-built
# Live2D rig stacks shade / highlight / back meshes behind every part) that assumption breaks: almost
# nothing opens into void, so free-edge collapses toward 0 for *every* part and stops discriminating
# (measured on real pro models — see tools/calibrate_moc3.py). We therefore trust free-edge less as the
# scene's overlap density rises, shifting that weight onto cantilever + slenderness, which don't depend
# on cross-part void. Below _DENSITY_LO the weights are the base values exactly, so a sparse scene (our
# whole current pipeline + tests) is byte-identical; the reweight only engages on dense inputs.
_DENSITY_LO = 0.35     # overlap density at/below which free-edge is fully trusted (base weights)
_DENSITY_HI = 0.70     # overlap density at/above which free-edge is maximally down-weighted
_FREE_EDGE_SHIFT = 0.30  # weight moved off free-edge (onto cantilever+slenderness) at full density

_SLENDER_REF = 4.0     # slenderness (aspect) at which the slender signal saturates to 1
_STRAND_ASPECT = 2.5   # slenderness at/above which a dynamic part is a strand (else sheet/jiggle)
_SHEET_CANT = 0.4      # cantilever at/above which a non-slender dynamic part is a hanging sheet

# --- Verdict thresholds (biased toward restraint) ---------------------------------------------------
_DYNAMIC_T = 0.55      # score >= this -> full physics
_GENTLE_T = 0.30       # score in [_GENTLE_T, _DYNAMIC_T) -> gentle motion only. Lowered 0.33 -> 0.30
#                        per P1b calibration on the (our-domain) taste corpus: at 0.33 a hair strand
#                        tucked against the head (low free edge, score ~0.30) was left dead; 0.30
#                        recovers it with no new false positives on that corpus.
_FREE_EDGE_FLOOR = 0.6  # safety net: an eligible part with this much free edge gets >= gentle motion
#                         even if its score fell just short — never leave an obvious hanging edge dead.

# Sway physics applies to soft appendages only. Skin and facial features move by deformation; limbs
# ARTICULATE about a joint (handled in author_rig), they don't pendulum-sway. This is a coarse,
# categorical gate (soft-appendage vs skin/limb/structure) — NOT a per-style rule. WITHIN the eligible
# set the geometry decides, so a novel accessory/garment is judged by shape, not by an enumerated role.
_SWAY_ELIGIBLE_ROLES = {
    SemanticRole.hair_front, SemanticRole.hair_side, SemanticRole.hair_back,
    SemanticRole.clothing, SemanticRole.accessory,
}

# Material prior by role (a soft material is likelier to swing). A weak nudge, never the decision.
_MATERIAL_PRIOR: dict[SemanticRole, float] = {
    SemanticRole.hair_front: 0.8, SemanticRole.hair_side: 0.8, SemanticRole.hair_back: 0.8,
    SemanticRole.clothing: 0.6, SemanticRole.accessory: 0.5,
}
# Name hints (substring of the part id) that raise the prior for soft/danging elements.
_SOFT_NAME_HINTS = (
    "bow", "ribbon", "tail", "braid", "ahoge", "skirt", "cape", "cloak", "scarf",
    "tie", "sleeve", "frill", "hem", "fringe", "bang", "twin", "pony",
)


class DynamicsVerdict(str, Enum):
    """How much secondary physics a part warrants."""

    rigid = "rigid"      # no own dynamics — rides its parent
    gentle = "gentle"    # subtle motion only
    dynamic = "dynamic"  # full pendulum/cloth physics


class PhysicalClass(str, Enum):
    """The motion archetype of a (gentle/dynamic) part — picks the generic synthesizer later."""

    rigid = "rigid"
    strand = "strand"    # slender, one end anchored -> 1-D pendulum (hair strand, ponytail, ribbon)
    sheet = "sheet"      # wide, hangs from a top edge -> multi-zone hem (skirt, cape, sleeve)
    jiggle = "jiggle"    # blobby soft -> small isotropic spring


# --------------------------------------------------------------------------------------------------
# Pure-core input / output
# --------------------------------------------------------------------------------------------------
@dataclass
class PartProbe:
    """One part's identity + an alpha sampler over the full canvas (the pure-core input)."""

    part_id: str
    role: SemanticRole
    alpha_at: AlphaSampler
    draw_order: int = 0


@dataclass
class PartDynamics:
    """The dynamics verdict for one part, with the signals that produced it (for QA / calibration)."""

    part_id: str
    role: SemanticRole
    free_edge_ratio: float   # 0..1 — fraction of the part's exposed boundary that opens into void
    cantilever: float        # 0..1 — how far mass hangs past its attachment (gravity direction)
    slenderness: float       # >=1 — principal-axis elongation (aspect)
    principal_angle: float   # radians, 0 = horizontal (long-axis orientation, model space y up)
    material_prior: float    # 0..1 — soft-material nudge from role + name hints
    anchor: Vec2             # model space (y up) — attachment centroid (where it hangs from)
    coverage: float          # occupied fraction of the probe grid (sanity: ~0 = empty layer)
    sway_eligible: bool      # False for skin/limbs/facial parts (they deform/articulate, not sway)
    score: float             # 0..1 — combined dynamics score
    verdict: DynamicsVerdict
    physical_class: PhysicalClass
    # The plan's remaining two signals (attachment fraction + depth/layer isolation), computed for a
    # complete feature vector but NOT folded into `score` — measured on a real pro rig they add no
    # discriminative power (see the note on `_score_one`). Exposed for QA / calibration / future ML.
    attachment_fraction: float = 0.0   # attached / exposed boundary (0..1). How pinned it is — the
    #                                    complement of free_edge_ratio; the swing-relevant *position*
    #                                    of the attachment is carried by `anchor`.
    depth_isolation: float = 0.0       # 1 - overlap (0..1). 1 = the footprint is the part's own (floats
    #                                    free); 0 = fully backed by other parts.


# --------------------------------------------------------------------------------------------------
# Detectors (pure core)
# --------------------------------------------------------------------------------------------------
def _occupancy(probe: PartProbe, samples: int, threshold: int) -> list[list[bool]]:
    """Sample a part onto an ``samples`` x ``samples`` boolean occupancy grid (row-major, v down)."""
    grid = [[False] * samples for _ in range(samples)]
    row = grid  # local alias
    for j in range(samples):
        v = (j + 0.5) / samples
        gj = row[j]
        for i in range(samples):
            u = (i + 0.5) / samples
            if probe.alpha_at(u, v) >= threshold:
                gj[i] = True
    return grid


def _material_prior(role: SemanticRole, part_id: str) -> float:
    base = _MATERIAL_PRIOR.get(role, 0.0)
    pid = part_id.lower()
    if any(h in pid for h in _SOFT_NAME_HINTS):
        base = min(1.0, base + 0.2)
    return base


_NEIGHBORS = ((1, 0), (-1, 0), (0, 1), (0, -1))


def score_dynamics(
    probes: list[PartProbe],
    *,
    samples: int = DEFAULT_SAMPLES,
    alpha_threshold: int = DEFAULT_ALPHA_THRESHOLD,
) -> list[PartDynamics]:
    """Score every part's dynamics from the *whole set* (free-edge needs cross-part coverage).

    Returns one ``PartDynamics`` per non-empty probe (empty layers are skipped, like ``core.landmark``
    skips a ``None`` silhouette). Pure: no IO, no Pillow, no ML — fully testable with synthetic
    samplers.
    """
    if samples < 2:
        raise ValueError(f"samples must be >= 2, got {samples}")

    grids = {p.part_id: _occupancy(p, samples, alpha_threshold) for p in probes}
    # Per-cell coverage count across ALL parts, so a part can ask "does anything else fill this gap?"
    total = [[0] * samples for _ in range(samples)]
    for grid in grids.values():
        for j in range(samples):
            tj, gj = total[j], grid[j]
            for i in range(samples):
                if gj[i]:
                    tj[i] += 1

    # Overlap density = fraction of occupied cells covered by more than one part. ~0 for our sparse,
    # non-overlapping decomposition; high for a densely layered pro rig. Reweights free-edge (below).
    occupied = multi = 0
    for tj in total:
        for t in tj:
            if t:
                occupied += 1
                if t >= 2:
                    multi += 1
    density = multi / occupied if occupied else 0.0

    out: list[PartDynamics] = []
    for p in probes:
        d = _score_one(p, grids[p.part_id], total, samples, density)
        if d is not None:
            out.append(d)
    return out


def _score_weights(density: float) -> tuple[float, float, float, float]:
    """(free-edge, cantilever, slenderness, material) weights for a scene of the given overlap
    ``density``. At/below ``_DENSITY_LO`` these are the base weights exactly (so a sparse scene is
    byte-identical); toward ``_DENSITY_HI`` weight moves off the now-unreliable free-edge cue onto
    cantilever/slenderness, split in proportion to their base weights. Weights always sum to 1."""
    span = _DENSITY_HI - _DENSITY_LO
    s = _FREE_EDGE_SHIFT * _clamp((density - _DENSITY_LO) / span, 0.0, 1.0) if span > 0 else 0.0
    cs = _W_CANTILEVER + _W_SLENDER
    return (
        _W_FREE_EDGE - s,
        _W_CANTILEVER + s * (_W_CANTILEVER / cs),
        _W_SLENDER + s * (_W_SLENDER / cs),
        _W_MATERIAL,
    )


def _score_one(
    probe: PartProbe, grid: list[list[bool]], total: list[list[int]], n: int, density: float = 0.0,
) -> PartDynamics | None:
    free_edges = attached_edges = 0
    au = av = 0.0            # attachment centroid accumulators (of the part's own attached cells)
    an = 0
    ncov = 0                 # occupied cells
    shared = 0               # occupied cells this part shares with >=1 other part (for depth isolation)
    su = sv = suu = svv = suv = 0.0
    vmin, vmax = math.inf, -math.inf

    for j in range(n):
        gj = grid[j]
        v = (j + 0.5) / n
        for i in range(n):
            if not gj[i]:
                continue
            u = (i + 0.5) / n
            ncov += 1
            if total[j][i] >= 2:
                shared += 1
            su += u
            sv += v
            suu += u * u
            svv += v * v
            suv += u * v
            if v < vmin:
                vmin = v
            if v > vmax:
                vmax = v
            # Classify each exposed edge of this occupied cell.
            for di, dj in _NEIGHBORS:
                ni, nj = i + di, j + dj
                if ni < 0 or ni >= n or nj < 0 or nj >= n:
                    free_edges += 1                 # canvas edge -> opens into the void
                    continue
                if grid[nj][ni]:
                    continue                        # interior direction (part covers the neighbor)
                if total[nj][ni] > 0:               # another part fills the gap -> attached
                    attached_edges += 1
                    au += u
                    av += v
                    an += 1
                else:                               # nothing there -> a free edge
                    free_edges += 1

    if ncov == 0:
        return None

    exposed = free_edges + attached_edges
    free_edge_ratio = (free_edges / exposed) if exposed else 0.0
    # The plan's remaining two signals — diagnostics only (see the return + module note on why they are
    # NOT in `score`). attachment_fraction is the pinned complement of free-edge; depth_isolation is how
    # much of the part's footprint is its own vs backed by other parts.
    attachment_fraction = (attached_edges / exposed) if exposed else 0.0
    depth_isolation = 1.0 - (shared / ncov)

    cu, cv = su / ncov, sv / ncov
    # Slenderness + orientation from the occupied-cell covariance (in model space: y up -> flip v).
    cov_uu = suu / ncov - cu * cu
    cov_vv = svv / ncov - cv * cv
    cov_uv = suv / ncov - cu * cv
    tr = cov_uu + cov_vv
    det = cov_uu * cov_vv - cov_uv * cov_uv
    disc = max(tr * tr / 4.0 - det, 0.0)
    lam1 = tr / 2.0 + math.sqrt(disc)
    lam2 = tr / 2.0 - math.sqrt(disc)
    slenderness = math.sqrt(lam1 / lam2) if lam2 > 1e-12 else _SLENDER_REF
    # y up: covariance in (u, y=1-v) has the same structure; the long-axis angle flips sign in v.
    principal_angle = 0.5 * math.atan2(-2.0 * cov_uv, cov_uu - cov_vv)

    # Anchor = attachment centroid (model space, y up); fall back to the top-centre if unattached.
    # In a DENSE scene the "attachment" is spurious — a part is backed on all sides by other layers, so
    # its attachment centroid drifts to its own centre and cantilever collapses to ~0 (same failure as
    # free-edge). So as density rises we slide the anchor toward the part's TOP edge (vmin), where a
    # hanging appendage really attaches, restoring cantilever. At/below _DENSITY_LO this is a no-op.
    trust = _clamp((density - _DENSITY_LO) / (_DENSITY_HI - _DENSITY_LO), 0.0, 1.0) \
        if _DENSITY_HI > _DENSITY_LO else 0.0
    if an:
        backed_v = av / an
        anchor_v = backed_v * (1.0 - trust) + vmin * trust
        anchor = (au / an, 1.0 - anchor_v)
    else:
        anchor = (cu, 1.0 - vmin)
        anchor_v = vmin

    # Cantilever: how far the mass centroid sits *below* the attachment (v down -> larger v = lower),
    # normalized by the half-height. 1 = fully hanging off the top; 0 = supported / attached at mass.
    half_h = 0.5 * max(vmax - vmin, 1e-6)
    cantilever = _clamp((cv - anchor_v) / half_h, 0.0, 1.0)

    material = _material_prior(probe.role, probe.part_id)
    slender_norm = _clamp((slenderness - 1.0) / (_SLENDER_REF - 1.0), 0.0, 1.0)
    w_free, w_cant, w_slender, w_material = _score_weights(density)
    score = (w_free * free_edge_ratio + w_cant * cantilever
             + w_slender * slender_norm + w_material * material)

    eligible = probe.role in _SWAY_ELIGIBLE_ROLES
    verdict = _verdict(score, free_edge_ratio, eligible)
    phys = _physical_class(verdict, slenderness, cantilever)

    # NOTE — attachment_fraction / depth_isolation complete the plan's six-signal list but are
    # deliberately absent from `score`. Measured on a real pro rig (Akari, via tools/calibrate_moc3):
    # attachment_fraction ranks physics vs non-physics parts at AUC≈0.50 (chance — it is 1-free_edge,
    # and free-edge itself collapses on a dense rig), and depth_isolation at AUC≈0.43 — *inverted*: on a
    # hand rig the parts the artist gave physics are the MORE layered ones (a strand is backed by its own
    # shade/highlight/back meshes), the opposite of the plan's "isolated part floats free" intuition.
    # Blending either into free-edge moved AUC by <0.03. So they ship as diagnostics for QA /
    # calibration / a future learned classifier, not as score terms — keeping the score byte-identical.
    return PartDynamics(
        part_id=probe.part_id, role=probe.role,
        free_edge_ratio=free_edge_ratio, cantilever=cantilever, slenderness=slenderness,
        principal_angle=principal_angle, material_prior=material, anchor=anchor,
        coverage=ncov / (n * n), sway_eligible=eligible, score=score,
        verdict=verdict, physical_class=phys,
        attachment_fraction=attachment_fraction, depth_isolation=depth_isolation,
    )


def _verdict(score: float, free_edge_ratio: float, eligible: bool) -> DynamicsVerdict:
    if not eligible:
        return DynamicsVerdict.rigid           # skin/limbs/facial: deform or articulate, never sway
    if score >= _DYNAMIC_T:
        return DynamicsVerdict.dynamic
    if score >= _GENTLE_T or free_edge_ratio >= _FREE_EDGE_FLOOR:
        return DynamicsVerdict.gentle          # safety net: an obvious free edge is never left dead
    return DynamicsVerdict.rigid


def _physical_class(verdict: DynamicsVerdict, slenderness: float, cantilever: float) -> PhysicalClass:
    if verdict is DynamicsVerdict.rigid:
        return PhysicalClass.rigid
    if slenderness >= _STRAND_ASPECT:
        return PhysicalClass.strand
    if cantilever >= _SHEET_CANT:
        return PhysicalClass.sheet
    return PhysicalClass.jiggle


def _clamp(v: float, lo: float, hi: float) -> float:
    return lo if v < lo else hi if v > hi else v


# --------------------------------------------------------------------------------------------------
# Mesh adapter: LayerStack + meshes -> per-part dynamics (no Pillow, deterministic)
# --------------------------------------------------------------------------------------------------
# The pipeline authors from meshes, not textures. A grid mesh already drops transparent cells, so the
# mesh silhouette IS the alpha silhouette: sampling point-in-mesh gives the same free-edge/cantilever
# cues the alpha path does, but purely from geometry. Fewer probes than the alpha path (a mesh boundary
# is cleaner than antialiased alpha), keeping it cheap enough to run inside author_rig / generate_physics.
_MESH_SAMPLES = 64


def _in_triangle(px: float, py: float, tri: tuple[Vec2, Vec2, Vec2]) -> bool:
    (ax, ay), (bx, by), (cx, cy) = tri
    d1 = (px - bx) * (ay - by) - (ax - bx) * (py - by)
    d2 = (px - cx) * (by - cy) - (bx - cx) * (py - cy)
    d3 = (px - ax) * (cy - ay) - (cx - ax) * (py - ay)
    has_neg = d1 < 0 or d2 < 0 or d3 < 0
    has_pos = d1 > 0 or d2 > 0 or d3 > 0
    return not (has_neg and has_pos)          # same sign on all three edges -> inside (or on) the tri


def _mesh_sampler(mesh) -> AlphaSampler:
    """A binary alpha sampler (255 inside, else 0) for a mesh. The scorer's ``v`` runs top->bottom
    (down) while mesh space is y up, so a probe ``(u, v)`` tests the model point ``(u, 1 - v)``."""
    tris = [(mesh.vertices[a], mesh.vertices[b], mesh.vertices[c]) for a, b, c in mesh.triangles]

    def alpha_at(u: float, v: float) -> int:
        px, py = u, 1.0 - v
        for tri in tris:
            if _in_triangle(px, py, tri):
                return 255
        return 0

    return alpha_at


def mesh_probes(stack: LayerStack, meshes: list) -> list[PartProbe]:
    """Per-part probes backed by point-in-mesh sampling (skips background/other and meshless layers)."""
    mesh_by_part = {m.part_id: m for m in meshes}
    probes: list[PartProbe] = []
    for layer in stack.layers:
        if layer.semantic_role in (SemanticRole.background, SemanticRole.other):
            continue
        m = mesh_by_part.get(layer.id)
        if m is not None:
            probes.append(PartProbe(layer.id, layer.semantic_role, _mesh_sampler(m), layer.draw_order))
    return probes


def analyze_meshes(
    stack: LayerStack,
    meshes: list,
    *,
    samples: int = _MESH_SAMPLES,
    alpha_threshold: int = DEFAULT_ALPHA_THRESHOLD,
) -> list[PartDynamics]:
    """Score every part's dynamics from its **mesh** silhouette — the deterministic, Pillow-free twin of
    ``analyze_stack`` used inside the mesh pipeline. Same scorer, geometry-derived occupancy."""
    return score_dynamics(mesh_probes(stack, meshes), samples=samples, alpha_threshold=alpha_threshold)


# --------------------------------------------------------------------------------------------------
# Pillow wrapper: LayerStack -> per-part dynamics
# --------------------------------------------------------------------------------------------------
def analyze_stack(
    stack: LayerStack,
    *,
    samples: int = DEFAULT_SAMPLES,
    alpha_threshold: int = DEFAULT_ALPHA_THRESHOLD,
) -> list[PartDynamics]:
    """Score the dynamics of every layer in ``stack`` from its PNG alpha (needs Pillow).

    Each layer PNG spans the whole canvas (same assumption as ``core.landmark``), so its sampler maps
    model ``(u, v)`` directly to a pixel. Background/other layers are skipped up front; the rest are
    scored together so the free-edge detector sees the full occupancy.
    """
    from PIL import Image  # local import: keep the module importable without Pillow

    probes: list[PartProbe] = []
    for layer in stack.layers:
        if layer.semantic_role in (SemanticRole.background, SemanticRole.other):
            continue
        with Image.open(layer.texture_path) as img:
            rgba = img.convert("RGBA")
            w, h = rgba.size
            alpha_px = rgba.getchannel("A").load()

        def alpha_at(u: float, v: float, _ap=alpha_px, _w=w, _h=h) -> int:
            px = min(_w - 1, max(0, int(u * _w)))
            py = min(_h - 1, max(0, int(v * _h)))
            return _ap[px, py]

        probes.append(PartProbe(layer.id, layer.semantic_role, alpha_at, layer.draw_order))

    return score_dynamics(probes, samples=samples, alpha_threshold=alpha_threshold)
