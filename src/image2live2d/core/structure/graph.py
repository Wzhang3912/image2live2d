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
    SemanticRole.nose, SemanticRole.mouth, SemanticRole.ear_l, SemanticRole.ear_r,
    SemanticRole.blush,
})

BODY_ROLES: frozenset[SemanticRole] = frozenset({
    SemanticRole.neck, SemanticRole.torso, SemanticRole.arm_l, SemanticRole.arm_r,
    SemanticRole.hand_l, SemanticRole.hand_r, SemanticRole.leg_l, SemanticRole.leg_r,
    SemanticRole.clothing,
})

HEAD = "head"
BODY = "body"


@dataclass
class RigNode:
    """One movable part in the graph. ``parent`` is the structural group it rides (``HEAD``/``BODY``,
    or ``None`` for background/other). Dynamics fields are ``None`` until alpha analysis fills them."""

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
        elif role in BODY_ROLES:
            parent = BODY
        elif role is SemanticRole.accessory:
            parent = _accessory_parent(anchor, head_ref, body_ref)
        else:
            parent = None                            # background / other
        nodes.append(RigNode(part_id=layer.id, role=role, parent=parent, anchor=anchor, bbox=box))
    return RigGraph(nodes)


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
