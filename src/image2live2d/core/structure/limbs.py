"""Split parts that are really a mirrored *pair* into a left and a right.

A decomposer hands back what is visually contiguous, not what is anatomically separate. On a real
character it returned **both arms as one layer**, labelled ``accessory``; both eyebrows as one layer
labelled ``eyebrow_l``; both ears as ``ear_l``. Every one of those is a pair masquerading as a part,
and each one breaks the rig in its own way:

* **Both arms in one mesh** cannot articulate. A left and a right arm swing *oppositely* about
  *different* shoulders, so one mesh spanning both can only ever move as a rigid sheet — the arms read
  as cardboard however well they are parented.
* **Both brows labelled ``eyebrow_l``** means ``eyebrow_r`` does not exist, so ``ParamBrowRY`` drives
  nothing at all and half the expression rig is dead.
* **Both earrings in one part** share one pendulum, so they swing in lockstep instead of independently.

The lobes are already there in the geometry: ``grid_mesh`` drops empty cells, so two separated alpha
blobs become two disconnected triangle clusters that :func:`mesh_components` recovers with no alpha
access. This module finds the parts whose mesh is two mirror lobes, decides what the pair actually
*is*, and rewrites it as two parts.

Three rules, in confidence order:

1. **A one-sided role that contains both sides is the pair.** ``eyebrow_l`` with two mirror lobes and
   no ``eyebrow_r`` anywhere is simply mislabelled: it *is* the two brows. Unambiguous.
2. **A wide, low, lateral pair the decomposer dumped in the junk drawer is the arms.** Deliberately
   strict (see :func:`_looks_like_arms`) — the cost of a false positive is a garment articulating like
   a limb.
3. **Anything else keeps its role and just stops being one rigid sheet.** Two earrings become two
   parts, each with its own mesh and its own pendulum.

The halves *share the source texture* — nothing is written to disk. Each half's mesh carries only its
own lobe's triangles, and UVs are full-canvas, so each samples its own side of the shared image.
"""

from __future__ import annotations

from ..types import Layer, LayerStack
from ...irr.schema import Mesh, SemanticRole, Vec2
from .strands import mesh_components

# A part is one of a mirrored pair only if the two lobes really do mirror: comparable size, sitting at
# the same height, on opposite sides of the body. A speckle beside a blob is not a pair.
_PAIR_SIZE_RATIO = 0.45      # the smaller lobe must be at least this fraction of the larger
_PAIR_LEVEL_TOL = 0.10       # their centroids must sit within this much of the same height (model units)

# Roles the decomposer uses as a junk drawer — the ones worth re-reading from geometry.
_UNSORTED: frozenset[SemanticRole] = frozenset({
    SemanticRole.accessory, SemanticRole.clothing, SemanticRole.other,
})

# One-sided roles and their twins. A part carrying the left role but holding *both* lobes is the pair.
_TWINS: dict[SemanticRole, tuple[SemanticRole, SemanticRole]] = {
    SemanticRole.eyebrow_l: (SemanticRole.eyebrow_l, SemanticRole.eyebrow_r),
    SemanticRole.eyebrow_r: (SemanticRole.eyebrow_l, SemanticRole.eyebrow_r),
    SemanticRole.ear_l: (SemanticRole.ear_l, SemanticRole.ear_r),
    SemanticRole.ear_r: (SemanticRole.ear_l, SemanticRole.ear_r),
    SemanticRole.eye_l: (SemanticRole.eye_l, SemanticRole.eye_r),
    SemanticRole.eye_r: (SemanticRole.eye_l, SemanticRole.eye_r),
    SemanticRole.eye_white_l: (SemanticRole.eye_white_l, SemanticRole.eye_white_r),
    SemanticRole.eye_white_r: (SemanticRole.eye_white_l, SemanticRole.eye_white_r),
    SemanticRole.pupil_l: (SemanticRole.pupil_l, SemanticRole.pupil_r),
    SemanticRole.pupil_r: (SemanticRole.pupil_l, SemanticRole.pupil_r),
    SemanticRole.arm_l: (SemanticRole.arm_l, SemanticRole.arm_r),
    SemanticRole.leg_l: (SemanticRole.leg_l, SemanticRole.leg_r),
}

# Arms reach *outside* the head's column and hang down the upper body. Both bounds are deliberately
# strict: mistaking a garment for a limb would give it shoulder and elbow articulation.
_ARM_MIN_HEIGHT_FRAC = 0.12  # each arm spans at least this fraction of the character's height
_ARM_MIN_LEVEL_FRAC = 0.35   # ...and hangs no lower than this fraction of the way down from the head


def _bbox(verts: list[Vec2]) -> tuple[float, float, float, float]:
    xs = [x for x, _ in verts]
    ys = [y for _, y in verts]
    return min(xs), min(ys), max(xs), max(ys)


def _centroid(verts: list[Vec2]) -> Vec2:
    return (sum(x for x, _ in verts) / len(verts), sum(y for _, y in verts) / len(verts))


def _sub_mesh(mesh: Mesh, part_id: str, keep: list[int]) -> Mesh | None:
    """The lobe ``keep`` as a standalone mesh, with vertices re-indexed and only its own triangles."""
    remap = {vi: i for i, vi in enumerate(keep)}
    tris = [(remap[a], remap[b], remap[c]) for a, b, c in mesh.triangles
            if a in remap and b in remap and c in remap]
    if len(keep) < 3 or not tris:
        return None
    return Mesh(
        part_id=part_id,
        vertices=[mesh.vertices[i] for i in keep],
        uvs=[mesh.uvs[i] for i in keep],
        triangles=tris,
    )


def _mirror_lobes(mesh: Mesh, midline_x: float) -> tuple[list[int], list[int]] | None:
    """The mesh's two lobes as ``(left, right)`` — or ``None`` if it isn't a mirrored pair."""
    comps = mesh_components(mesh)
    if len(comps) != 2:
        return None
    a, b = comps
    if min(len(a), len(b)) < _PAIR_SIZE_RATIO * max(len(a), len(b)):
        return None                                   # a speckle beside a blob is not a pair
    ca = _centroid([mesh.vertices[i] for i in a])
    cb = _centroid([mesh.vertices[i] for i in b])
    if abs(ca[1] - cb[1]) > _PAIR_LEVEL_TOL:
        return None                                   # a pair sits at the same height
    if (ca[0] < midline_x) == (cb[0] < midline_x):
        return None                                   # both on one side: not a left and a right
    return (a, b) if ca[0] < cb[0] else (b, a)


def _looks_like_arms(
    mesh: Mesh, lobes: tuple[list[int], list[int]], *, head_box, body_box,
) -> bool:
    """Is this junk-drawer pair the character's arms?

    Arms are the pair that hangs *outside the head's column* and *down the upper body*. Earrings also
    come in mirrored pairs but sit inside the head's width; shoes sit at the feet. Both tests must hold
    for both lobes, so a garment is never mistaken for a limb.
    """
    hx0, _, hx1, hy0 = head_box[0], head_box[1], head_box[2], head_box[1]
    _, by0, _, _ = body_box
    height = max(body_box[3] - by0, 1e-6)
    # the waistline we require the arms to stay above: part-way down from the head to the feet
    floor_y = hy0 - _ARM_MIN_LEVEL_FRAC * (hy0 - by0)
    for lobe in lobes:
        verts = [mesh.vertices[i] for i in lobe]
        cx, _ = _centroid(verts)
        x0, y0, x1, y1 = _bbox(verts)
        if hx0 <= cx <= hx1:
            return False                              # inside the head's column — an earring, not an arm
        if (y1 - y0) < _ARM_MIN_HEIGHT_FRAC * height:
            return False                              # too small to be a limb
        if y1 < floor_y:
            return False                              # hangs too low — that's a leg or a shoe
    return True


def split_bundled_pairs(stack: LayerStack, meshes: list[Mesh]) -> list[str]:
    """Rewrite every part that is really a mirrored pair as two parts. Mutates ``stack`` and ``meshes``;
    returns the ids of the parts created."""
    mesh_by_part = {m.part_id: m for m in meshes}
    present = {ly.semantic_role for ly in stack.layers}
    all_verts = [v for m in meshes for v in m.vertices]
    if not all_verts:
        return []
    body_box = _bbox(all_verts)
    midline_x = (body_box[0] + body_box[2]) / 2.0

    head = [m for ly in stack.layers if (m := mesh_by_part.get(ly.id))
            and ly.semantic_role in (SemanticRole.face_base, SemanticRole.neck)]
    head_box = _bbox([v for m in head for v in m.vertices]) if head else None

    created: list[str] = []
    for layer in list(stack.layers):
        mesh = mesh_by_part.get(layer.id)
        if mesh is None:
            continue
        lobes = _mirror_lobes(mesh, midline_x)
        if lobes is None:
            continue

        role = layer.semantic_role
        twin = _TWINS.get(role)
        if twin and twin[1] not in present and twin[0] not in (present - {role}):
            roles = twin                              # 1. a one-sided role holding both sides
        elif (role in _UNSORTED and head_box
                and _looks_like_arms(mesh, lobes, head_box=head_box, body_box=body_box)):
            roles = (SemanticRole.arm_l, SemanticRole.arm_r)      # 2. the junk-drawer arms
        else:
            roles = (role, role)                      # 3. keep the role; just stop being one sheet

        halves = []
        for lobe, new_role, side in zip(lobes, roles, ("l", "r")):
            base = f"{layer.draw_order:02d}_{new_role.value}"
            pid = base if roles[0] is not roles[1] else f"{base}_{side}"
            sub = _sub_mesh(mesh, pid, lobe)
            if sub is None:
                halves = []
                break
            halves.append((Layer(id=pid, semantic_role=new_role, texture_path=layer.texture_path,
                                 draw_order=layer.draw_order, width=layer.width, height=layer.height,
                                 bbox=layer.bbox), sub))
        if not halves:
            continue

        i = stack.layers.index(layer)
        stack.layers[i:i + 1] = [ly for ly, _ in halves]
        j = meshes.index(mesh)
        meshes[j:j + 1] = [m for _, m in halves]
        for ly, m in halves:
            mesh_by_part[ly.id] = m
            present.add(ly.semantic_role)
            created.append(ly.id)

    return created
