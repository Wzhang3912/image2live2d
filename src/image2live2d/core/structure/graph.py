"""RigGraph — the character's movable skeleton: each part's kinematic parent + anchor + dynamics.

This is the P1 backbone that ``author_rig`` / ``generate_physics`` consume instead of ad-hoc role
scans (see docs/AUTORIG_PHYSICS_UNIVERSAL_PLAN.md). A ``RigGraph`` is built from the parts + meshes
(+ optional per-part dynamics from :mod:`.dynamics`); each ``RigNode`` records which structural group
a part rides (``"head"`` / ``"body"``) and where it attaches, so downstream stages parent motion
generically rather than enumerating roles.

Kinematic parenting is intentionally the *same rule* the rig used before (head-role parts ride the
head, body-role parts ride the body, an accessory binds to whichever group is nearest its attachment
point) so this can be dropped in with **no change to current output** — the first consumer just reads
the graph where it used to compute the split inline. Parenting here is mesh-based (no alpha), so it
stays usable in the pure pipeline; the alpha-based dynamics score is layered on by ``analyze_structure``
when textures are available.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from ..types import LayerStack
from ...irr.schema import Mesh, SemanticRole, Vec2
from .dynamics import DynamicsVerdict, PhysicalClass

# Structural groups a part can ride. Kept here (not in rig.author) so the graph owns the taxonomy and
# rig.author imports it — matching the pre-refactor _HEAD_ROLES / _BODY_ROLES exactly.
HEAD_ROLES: frozenset[SemanticRole] = frozenset({
    SemanticRole.face_base, SemanticRole.hair_front, SemanticRole.hair_side, SemanticRole.hair_back,
    SemanticRole.eyebrow_l, SemanticRole.eyebrow_r, SemanticRole.eye_l, SemanticRole.eye_r,
    SemanticRole.eye_white_l, SemanticRole.eye_white_r, SemanticRole.pupil_l, SemanticRole.pupil_r,
    SemanticRole.nose, SemanticRole.mouth, SemanticRole.mouth_cavity,
    SemanticRole.ear_l, SemanticRole.ear_r, SemanticRole.blush,
})

BODY_ROLES: frozenset[SemanticRole] = frozenset({
    SemanticRole.neck, SemanticRole.torso, SemanticRole.arm_l, SemanticRole.arm_r,
    SemanticRole.hand_l, SemanticRole.hand_r, SemanticRole.leg_l, SemanticRole.leg_r,
    SemanticRole.clothing,
})

HEAD = "head"
BODY = "body"
# Limb structural groups. A garment appendage that sits *over* an arm (a sleeve/cuff) rides that arm's
# articulation rather than the body sway — the P4 "sleeve→arm" parenting. Kept distinct from BODY so the
# garment planner can pick the arm's motion param (ParamArm*) as the pendulum driver.
ARM_L = "arm_l"
ARM_R = "arm_r"

# A clothing part is bound to an arm only when its footprint sits *predominantly* over that arm — a real
# sleeve/cuff is mostly on the arm, whereas a bodice/top is mostly over the torso. The overlap must both
# clear this floor AND exceed the part's overlap with the torso, so an ordinary torso garment (the whole
# existing wardrobe) is never captured and stays on the body — keeping current output byte-identical.
_SLEEVE_ARM_OVERLAP_MIN = 0.5


@dataclass
class RigNode:
    """One movable part in the graph. ``parent`` is the structural group it rides (``HEAD``/``BODY``/
    ``ARM_L``/``ARM_R``, or ``None`` for background/other). Dynamics fields are ``None`` until alpha
    analysis fills them."""

    part_id: str
    role: SemanticRole
    parent: str | None
    anchor: Vec2                                   # attachment point (model space, y up): top-centre
    bbox: tuple[float, float, float, float]        # (x0, y0, x1, y1), y up
    verdict: DynamicsVerdict | None = None
    physical_class: PhysicalClass | None = None
    dynamics_score: float | None = None


@dataclass
class RigGraph:
    """The parts' movable skeleton. ``nodes`` are in stack order (deterministic downstream iteration)."""

    nodes: list[RigNode] = field(default_factory=list)

    def __post_init__(self) -> None:
        self._by_id = {n.part_id: n for n in self.nodes}

    def node(self, part_id: str) -> RigNode | None:
        return self._by_id.get(part_id)

    def parent_of(self, part_id: str) -> str | None:
        n = self._by_id.get(part_id)
        return n.parent if n else None

    def children(self, parent: str) -> list[RigNode]:
        return [n for n in self.nodes if n.parent == parent]


# --------------------------------------------------------------------------------------------------
# Assembly (pure core — mesh-based, no alpha/Pillow)
# --------------------------------------------------------------------------------------------------
def _bbox(verts: list[Vec2]) -> tuple[float, float, float, float]:
    xs = [x for x, _ in verts]
    ys = [y for _, y in verts]
    return min(xs), min(ys), max(xs), max(ys)


def _point_bbox_dist(p: Vec2, box: tuple[float, float, float, float]) -> float:
    """Euclidean distance from point ``p`` to an axis-aligned bbox (0 if inside)."""
    px, py = p
    x0, y0, x1, y1 = box
    dx = max(x0 - px, 0.0, px - x1)
    dy = max(y0 - py, 0.0, py - y1)
    return math.hypot(dx, dy)


def build_rig_graph(
    stack: LayerStack,
    meshes: list[Mesh],
    landmarks: object | None = None,
) -> RigGraph:
    """Assemble the ``RigGraph`` from parts + meshes (pure; no alpha).

    Parenting reproduces the rig's prior behaviour exactly: a head-role part rides ``HEAD``, a
    body-role part rides ``BODY``, and an accessory binds to whichever group is nearest its attachment
    point (top-centre) — falling to the only group present when the other is absent. ``landmarks`` is
    accepted for parity with ``analyze_structure`` and future anchor refinement; parenting is
    mesh-based so the graph is usable without textures.
    """
    mesh_by_part = {m.part_id: m for m in meshes}
    # Reference bboxes for the accessory split (only parts that actually have a mesh, in stack order).
    head_ref = [mesh_by_part[ly.id] for ly in stack.layers
                if ly.id in mesh_by_part and ly.semantic_role in HEAD_ROLES]
    body_ref = [mesh_by_part[ly.id] for ly in stack.layers
                if ly.id in mesh_by_part and ly.semantic_role in BODY_ROLES]
    # References for the sleeve→arm split: each arm's footprint, and the torso core to compare against.
    arm_ref = {ARM_L: [mesh_by_part[ly.id].vertices for ly in stack.layers
                       if ly.id in mesh_by_part and ly.semantic_role is SemanticRole.arm_l],
               ARM_R: [mesh_by_part[ly.id].vertices for ly in stack.layers
                       if ly.id in mesh_by_part and ly.semantic_role is SemanticRole.arm_r]}
    torso_ref = [mesh_by_part[ly.id].vertices for ly in stack.layers
                 if ly.id in mesh_by_part and ly.semantic_role in (SemanticRole.torso, SemanticRole.neck)]

    nodes: list[RigNode] = []
    for layer in stack.layers:
        m = mesh_by_part.get(layer.id)
        if m is None:
            continue
        box = _bbox(m.vertices)
        x0, _, x1, y1 = box
        anchor = ((x0 + x1) / 2.0, y1)              # top-centre (model y up)
        role = layer.semantic_role
        if role in HEAD_ROLES:
            parent: str | None = HEAD
        elif role is SemanticRole.clothing:
            parent = _sleeve_arm(box, arm_ref, torso_ref) or BODY   # a sleeve rides its arm, else body
        elif role in BODY_ROLES:
            parent = BODY
        elif role is SemanticRole.accessory:
            parent = _accessory_parent(anchor, head_ref, body_ref)
        else:
            parent = None                            # background / other
        nodes.append(RigNode(part_id=layer.id, role=role, parent=parent, anchor=anchor, bbox=box))
    return RigGraph(nodes)


def _bbox_overlap_frac(box: tuple[float, float, float, float], verts: list[Vec2]) -> float:
    """Fraction of ``box``'s area that lies inside the bounding box of ``verts`` (0 if no overlap)."""
    if not verts:
        return 0.0
    ox0, oy0, ox1, oy1 = _bbox(verts)
    bx0, by0, bx1, by1 = box
    iw = max(0.0, min(bx1, ox1) - max(bx0, ox0))
    ih = max(0.0, min(by1, oy1) - max(by0, oy0))
    area = max((bx1 - bx0) * (by1 - by0), 1e-12)
    return (iw * ih) / area


def _sleeve_arm(
    box: tuple[float, float, float, float],
    arm_ref: dict[str, list[list[Vec2]]],
    torso_ref: list[list[Vec2]],
) -> str | None:
    """Bind a clothing part to an arm when it sits *predominantly* over that arm — the sleeve/cuff case.

    Returns ``ARM_L``/``ARM_R`` only if the part's footprint overlaps an arm by at least
    ``_SLEEVE_ARM_OVERLAP_MIN`` of its own area *and* more than it overlaps the torso; otherwise ``None``
    (the caller falls back to the body). This is deliberately strict so an ordinary torso garment — which
    sits over the torso, not an arm — is never captured, leaving the existing wardrobe on the body."""
    torso_ov = max((_bbox_overlap_frac(box, v) for v in torso_ref), default=0.0)
    best_side, best_ov = None, _SLEEVE_ARM_OVERLAP_MIN
    for side, groups in arm_ref.items():
        ov = max((_bbox_overlap_frac(box, v) for v in groups), default=0.0)
        if ov >= best_ov and ov > torso_ov:
            best_side, best_ov = side, ov
    return best_side


def _accessory_parent(anchor: Vec2, head_ref: list[Mesh], body_ref: list[Mesh]) -> str:
    """Bind an accessory to the nearest of the head/body groups by its attachment point (top-centre),
    falling to whichever group exists when the other is absent. Matches the prior _classify_accessories."""
    if not head_ref:
        return BODY
    if not body_ref:
        return HEAD
    dh = min(_point_bbox_dist(anchor, _bbox(h.vertices)) for h in head_ref)
    db = min(_point_bbox_dist(anchor, _bbox(b.vertices)) for b in body_ref)
    return HEAD if dh <= db else BODY


# --------------------------------------------------------------------------------------------------
# Pillow wrapper: attach the alpha-based dynamics score to the graph
# --------------------------------------------------------------------------------------------------
def analyze_structure(
    stack: LayerStack,
    meshes: list[Mesh],
    landmarks: object | None = None,
    *,
    samples: int | None = None,
) -> RigGraph:
    """Full assembly: build the mesh-based graph, then layer on each part's dynamics score from its
    texture alpha (needs Pillow). Nodes gain ``verdict`` / ``physical_class`` / ``dynamics_score``.
    """
    from .dynamics import DEFAULT_SAMPLES, analyze_stack

    graph = build_rig_graph(stack, meshes, landmarks)
    dyn = {d.part_id: d for d in analyze_stack(stack, samples=samples or DEFAULT_SAMPLES)}
    for node in graph.nodes:
        d = dyn.get(node.part_id)
        if d is not None:
            node.verdict = d.verdict
            node.physical_class = d.physical_class
            node.dynamics_score = d.score
    return graph
