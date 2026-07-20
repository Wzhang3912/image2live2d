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

# --- fused legs ------------------------------------------------------------------------------------
# Legs cannot be recovered as connected components: the thighs meet, so both legs are one blob joined at
# the hips. But they are only fused at the *top* — below the crotch a real gap opens between them, and
# grid_mesh drops those empty cells, so the gap is already a hole in the lattice. Find the hole, and the
# seam to cut along is the line it traces.
_SEAM_GAP_MIN = 1.6          # a row has a hole when its widest interior gap exceeds this many grid steps
_SEAM_MIN_ROWS_FRAC = 0.30   # the hole must run up at least this fraction of the part's rows from the hem
_LEG_MIN_HEIGHT_FRAC = 0.25  # legs are a big part of a body; a trim or a slit in a skirt is not
_LEG_SEAM_CENTRED = 0.10     # the seam must sit within this fraction of the body's width of its midline


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


# --- arms the decomposer mislabelled as legs -------------------------------------------------------
# See-through sometimes labels a character's ARMS as ``leg_l``/``leg_r`` — a slim figure with her arms at
# her sides, a chibi with stubby arms — and the pipeline trusts the filename role, so the rig builds LEG
# articulation on the arms and the character gets no arm motion at all (2 of 8 test characters). Roles are
# re-derived from geometry everywhere else here; do the same. Two facts separate an arm from a leg, and
# measured across 8 characters they hold with no overlap:
#   * an arm attaches at the SHOULDER — its top sits at the head's base and never rises above the head;
#   * an arm ends at the wrist MID-BODY — it does not reach the feet, whereas a leg runs to the floor.
# Both are needed: the shoulder test alone would also catch drill-hair a decomposer mislabels "leg" (it
# rises past the crown, so the "never above the head" clause rejects it); the foot test alone would catch
# a raised arm. Deliberately strict — a false positive gives a leg shoulder/elbow articulation.
_ARM_SHOULDER_MARGIN = 0.10  # the arm's top may sit at most this fraction of body height above the shoulder
_ARM_FOOT_CLEARANCE = 0.20   # ...and its bottom must clear the feet by at least this much of body height


def _leg_looks_like_arm(mesh: Mesh, *, head_box, body_box) -> bool:
    """Is this LEG-labelled part geometrically an arm (attaches at the shoulder, stops above the feet)?"""
    shoulder_y = head_box[1]                        # head bottom (y-up) — the shoulder line
    by0, by1 = body_box[1], body_box[3]
    height = max(by1 - by0, 1e-6)
    x0, y0, x1, y1 = _bbox(mesh.vertices)           # y1 = top (max y-up), y0 = bottom (min y-up)
    if (y1 - y0) <= (x1 - x0):
        return False                                # a limb is slender; a wide blob is a garment
    if y1 > shoulder_y + _ARM_SHOULDER_MARGIN * height:
        return False                                # rises above the shoulder/head — a leg reaches only
        #                                             the hip and drill-hair rises past the crown
    if y0 < by0 + _ARM_FOOT_CLEARANCE * height:
        return False                                # reaches down to the feet — that is a leg
    return True


def reassign_arm_mislabeled_as_leg(stack: LayerStack, meshes: list[Mesh]) -> list[str]:
    """Relabel ``leg_l``/``leg_r`` parts that are geometrically arms to ``arm_l``/``arm_r``. Mutates the
    layers' roles in ``stack``; returns the ids re-roled. Run before the pair/leg splitters so the arms
    flow through arm handling. A part keeps its own left/right side (the decomposer's L/R is position-
    consistent here); the real legs, if fused into clothing, are a separate problem this does not touch."""
    mesh_by_part = {m.part_id: m for m in meshes}
    all_verts = [v for m in meshes for v in m.vertices]
    if not all_verts:
        return []
    body_box = _bbox(all_verts)
    head = [m for ly in stack.layers if (m := mesh_by_part.get(ly.id))
            and ly.semantic_role in (SemanticRole.face_base, SemanticRole.neck)]
    if not head:
        return []                                   # no head to place the shoulder — don't guess
    head_box = _bbox([v for m in head for v in m.vertices])

    present = {ly.semantic_role for ly in stack.layers}
    changed: list[str] = []
    for layer in stack.layers:
        if layer.semantic_role not in (SemanticRole.leg_l, SemanticRole.leg_r):
            continue
        mesh = mesh_by_part.get(layer.id)
        if mesh is None or not _leg_looks_like_arm(mesh, head_box=head_box, body_box=body_box):
            continue
        new_role = (SemanticRole.arm_r if layer.semantic_role is SemanticRole.leg_r
                    else SemanticRole.arm_l)
        if new_role in present:
            continue                                # that side already has a real arm — don't duplicate
        layer.semantic_role = new_role
        present.add(new_role)
        changed.append(layer.id)
    return changed


def _leg_seam(mesh: Mesh, *, body_box) -> float | None:
    """The x of the seam between two fused legs — or ``None`` if this part is not a pair of legs.

    Walks the mesh's lattice rows from the hem upward looking for the gap between the legs. ``grid_mesh``
    drops transparent cells, so the space between two legs is literally a hole in the lattice: a row that
    straddles it has one interior gap far wider than its own grid step. Those rows must run *up from the
    hem* (legs open downward; a skirt is solid) and the gap must sit on the body's midline.
    """
    rows: dict[float, list[float]] = {}
    for x, y in mesh.vertices:
        rows.setdefault(round(y, 5), []).append(x)
    if len(rows) < 4:
        return None

    step = min((sorted(xs)[i + 1] - sorted(xs)[i]
                for xs in rows.values() if len(xs) > 1
                for i in range(len(sorted(xs)) - 1)), default=0.0)
    if step <= 0.0:
        return None

    ordered = sorted(rows)                              # bottom (hem) -> top
    centres: list[float] = []
    for y in ordered:
        xs = sorted(rows[y])
        gap, centre = max(((xs[i + 1] - xs[i], (xs[i] + xs[i + 1]) / 2.0)
                           for i in range(len(xs) - 1)), default=(0.0, 0.0))
        if gap < _SEAM_GAP_MIN * step:
            break                                       # the legs have fused: this is the crotch
        centres.append(centre)
    if len(centres) < _SEAM_MIN_ROWS_FRAC * len(ordered):
        return None                                     # no hole, or only a nick at the hem

    seam = sorted(centres)[len(centres) // 2]           # median: robust to a ragged hem
    bx0, by0, bx1, by1 = body_box
    if abs(seam - (bx0 + bx1) / 2.0) > _LEG_SEAM_CENTRED * (bx1 - bx0):
        return None                                     # off-centre: a slit or a fold, not a crotch
    ys = [y for _, y in mesh.vertices]
    if (max(ys) - min(ys)) < _LEG_MIN_HEIGHT_FRAC * (by1 - by0):
        return None                                     # too small to be a pair of legs
    return seam


def _cut_at_seam(mesh: Mesh, seam_x: float, ids: tuple[str, str]) -> tuple[Mesh, Mesh] | None:
    """Cut a mesh into ``(left, right)`` along a vertical seam.

    Assigns whole *triangles* by their centroid rather than splitting vertices across the line, so no
    triangle is dropped and no hole opens along the cut: every triangle is drawn exactly once, by one
    side or the other. Vertices on the seam are simply carried by both halves.
    """
    out = []
    for side, keep_left in zip(ids, (True, False)):
        tris = [t for t in mesh.triangles
                if (sum(mesh.vertices[i][0] for i in t) / 3.0 < seam_x) is keep_left]
        used = sorted({i for t in tris for i in t})
        sub = _sub_mesh(mesh, side, used)
        if sub is None:
            return None
        out.append(sub)
    return out[0], out[1]


def split_fused_legs(stack: LayerStack, meshes: list[Mesh]) -> list[str]:
    """Cut a part that is *both* legs fused at the hips into a left and a right leg.

    :func:`split_bundled_pairs` cannot do this: connected components only separate parts that are
    already disjoint, and the thighs touch, so both legs come back as one blob. The gap between the legs
    below the crotch is the handle — see :func:`_leg_seam`. Mutates ``stack`` and ``meshes``; returns the
    ids created.
    """
    mesh_by_part = {m.part_id: m for m in meshes}
    all_verts = [v for m in meshes for v in m.vertices]
    if not all_verts:
        return []
    body_box = _bbox(all_verts)

    created: list[str] = []
    for layer in list(stack.layers):
        mesh = mesh_by_part.get(layer.id)
        if mesh is None:
            continue
        role = layer.semantic_role
        if role not in _UNSORTED and role is not SemanticRole.leg_l:
            continue
        if SemanticRole.leg_r in {ly.semantic_role for ly in stack.layers}:
            continue                                    # a real right leg exists; leave this alone
        if len(mesh_components(mesh)) != 1:
            continue                                    # already separable: split_bundled_pairs owns it
        seam = _leg_seam(mesh, body_box=body_box)
        if seam is None:
            continue

        ids = (f"{layer.draw_order:02d}_{SemanticRole.leg_l.value}",
               f"{layer.draw_order:02d}_{SemanticRole.leg_r.value}")
        cut = _cut_at_seam(mesh, seam, ids)
        if cut is None:
            continue
        halves = [
            (Layer(id=ids[k], semantic_role=r, texture_path=layer.texture_path,
                   draw_order=layer.draw_order, width=layer.width, height=layer.height,
                   bbox=layer.bbox), cut[k])
            for k, r in enumerate((SemanticRole.leg_l, SemanticRole.leg_r))
        ]
        i = stack.layers.index(layer)
        stack.layers[i:i + 1] = [ly for ly, _ in halves]
        j = meshes.index(mesh)
        meshes[j:j + 1] = [m for _, m in halves]
        for ly, m in halves:
            mesh_by_part[ly.id] = m
            created.append(ly.id)
    return created


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
