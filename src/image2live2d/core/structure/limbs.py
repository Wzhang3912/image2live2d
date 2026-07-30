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
# Facial twins are level by anatomy, so the tolerance above is right for them. LIMBS are not: a pose
# raises one arm and leaves the other down, and the pair is still a pair. An absolute tolerance cannot
# express that (it means different things on a chibi and on an adult), so limbs get one measured
# against the lobes' OWN length: the mopping character's arms are 0.302 and 0.192 tall with centroids
# 0.139 apart, i.e. 0.55 of their mean length. Under the strict 0.10 they were not a pair at all.
# Swept on 13 characters: 0.40 and 0.50 miss the arms, 0.60 through 1.20 all give the identical result,
# so this sits in the middle of a plateau rather than on a threshold fitted to one character.
_PAIR_POSE_LEVEL_FRAC = 0.75  # ...as a fraction of the mean lobe height

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
# How much of a lobe must lie beside the head before "inside the head's column" condemns it as
# jewellery. Measured: an earring overlaps the head over 1.00 of its own height, a raised forearm 0.00.
# Nothing in between on 13 characters, and swept from 0.05 to 0.90 the result never changes.
_EARRING_HEAD_OVERLAP = 0.25

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


def _mirror_lobes(mesh: Mesh, midline_x: float, *, posed: bool = False) -> tuple[list[int], list[int]] | None:
    """The mesh's two lobes as ``(left, right)`` — or ``None`` if it isn't a mirrored pair.

    ``posed`` allows the two lobes to sit at different heights, for pairs a pose can move independently
    (limbs). Facial twins should leave it off: eyes and brows are level, and letting them drift would
    pair a feature with something that merely happens to be beside it.
    """
    comps = mesh_components(mesh)
    if len(comps) != 2:
        return None
    a, b = comps
    if min(len(a), len(b)) < _PAIR_SIZE_RATIO * max(len(a), len(b)):
        return None                                   # a speckle beside a blob is not a pair
    va = [mesh.vertices[i] for i in a]
    vb = [mesh.vertices[i] for i in b]
    ca, cb = _centroid(va), _centroid(vb)
    if posed:
        ha = _bbox(va)[3] - _bbox(va)[1]
        hb = _bbox(vb)[3] - _bbox(vb)[1]
        level_tol = max(_PAIR_LEVEL_TOL, _PAIR_POSE_LEVEL_FRAC * (ha + hb) / 2.0)
    else:
        level_tol = _PAIR_LEVEL_TOL
    if abs(ca[1] - cb[1]) > level_tol:
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
    hx0, hy0, hx1, hy1 = head_box[0], head_box[1], head_box[2], head_box[3]
    _, by0, _, _ = body_box
    height = max(body_box[3] - by0, 1e-6)
    # the waistline we require the arms to stay above: part-way down from the head to the feet
    floor_y = hy0 - _ARM_MIN_LEVEL_FRAC * (hy0 - by0)
    for lobe in lobes:
        verts = [mesh.vertices[i] for i in lobe]
        cx, _ = _centroid(verts)
        x0, y0, x1, y1 = _bbox(verts)
        # Jewellery is inside the head's column *at the head's height*. Below the chin that column is
        # the TORSO's column, and arms live there all the time — folded, crossed, or holding something
        # up in front of the body. Testing the column alone assumes arms hang clear of the body's
        # centre, which is only true of the arms-at-sides pose: on a mopping character the raised
        # forearm (x 0.466-0.572, entirely inside the head column 0.452-0.623) was rejected as an
        # earring, and she shipped with no arms at all. So the column only condemns a lobe that
        # actually reaches up alongside the head.
        overlap = min(y1, hy1) - max(y0, hy0)
        if hx0 <= cx <= hx1 and overlap > _EARRING_HEAD_OVERLAP * max(y1 - y0, 1e-6):
            return False                              # beside the head, at head height — jewellery
        if (y1 - y0) < _ARM_MIN_HEIGHT_FRAC * height:
            return False                              # too small to be a limb
        if y1 < floor_y:
            return False                              # hangs too low — that's a leg or a shoe
        # An arm attaches AT the shoulder, so it cannot start above one. This rule had a floor but no
        # ceiling, and a wide-brimmed hat clears every other test — its two lobes are a mirrored pair,
        # they sit outside the head's narrow column, they are tall enough, and they are nowhere near the
        # waist. So `blondedrills`' headwear was split into arm_l/arm_r up at the top of the canvas
        # (y 0.84-0.99), and the real arms then had to share a shoulder pivot with a hat. The sibling
        # rule `_leg_looks_like_arm` already tests exactly this; both now use the same shoulder line.
        if y1 > hy0 + _ARM_SHOULDER_MARGIN * height:
            return False                              # rises above the shoulder — headwear, not arms
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


_LIMB_SIDE_PAIRS = ((SemanticRole.arm_l, SemanticRole.arm_r),
                    (SemanticRole.leg_l, SemanticRole.leg_r))
_SIDE_MARGIN_FRAC = 0.03     # a part must sit this far (×body width) past the midline to count as a side


def reassign_mixed_limb_sides(stack: LayerStack, meshes: list[Mesh]) -> list[str]:
    """When a limb ROLE holds parts on BOTH sides of the body midline, snap each part in that pair to the
    role matching its own side (character-left = +x).

    The decomposer sometimes splits one physical arm's sub-parts (e.g. a shoulder puff vs the arm) across
    ``arm_l`` AND ``arm_r`` with the left/right swapped, so a single role ends up spanning the full width.
    The limb rigging then unions those opposite-side parts into one 'limb' whose shoulder lands on one
    side and wrist on the other — a full-width diagonal — and lower-body skirt tiers pass as 'wrist
    riders'/overlap-followers and rigidly rotate with the arm (the magicalgirl skirt-drag under
    arm-swing). Snapping each part to its geometric side keeps every limb role on one side.

    A *clean* full L/R swap (every part of a role on one side, just mirror-named) is left alone: swing
    direction is computed from geometry, not the label, so the swap is harmless and re-roling it is
    needless churn. Mutates roles in ``stack``; returns the ids re-roled."""
    mbp = {m.part_id: m for m in meshes}
    all_verts = [v for m in meshes for v in m.vertices]
    if not all_verts:
        return []
    x0, _, x1, _ = _bbox(all_verts)
    head = [m for ly in stack.layers if (m := mbp.get(ly.id))
            and ly.semantic_role in (SemanticRole.face_base, SemanticRole.neck)]
    mid = _centroid([v for m in head for v in m.vertices])[0] if head else 0.5 * (x0 + x1)
    margin = _SIDE_MARGIN_FRAC * max(x1 - x0, 1e-6)

    def side(m: Mesh) -> int:                        # +1 character-left (+x), -1 character-right, 0 ambiguous
        cx = _centroid(m.vertices)[0]
        return 1 if cx > mid + margin else (-1 if cx < mid - margin else 0)

    changed: list[str] = []
    for lrole, rrole in _LIMB_SIDE_PAIRS:
        members = [(ly, mbp[ly.id]) for ly in stack.layers
                   if ly.semantic_role in (lrole, rrole) and ly.id in mbp]
        # only act when a role straddles the midline (the mixed case that inflates the union)
        mixed = any({side(m) for ly, m in members if ly.semantic_role is role} >= {-1, 1}
                    for role in (lrole, rrole))
        if not mixed:
            continue
        for ly, m in members:
            want = lrole if side(m) > 0 else rrole if side(m) < 0 else ly.semantic_role
            if want is not ly.semantic_role:
                ly.semantic_role = want
                changed.append(ly.id)
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

    head = [m for ly in stack.layers if (m := mesh_by_part.get(ly.id))
            and ly.semantic_role in (SemanticRole.face_base, SemanticRole.neck)]
    head_box = _bbox([v for m in head for v in m.vertices]) if head else None

    # The midline decides which lobe is a left and which is a right, so it has to be the *character's*
    # centre — and the union of every part is not that. A character holding something (a mop, a staff,
    # a banner) has a part reaching far out to one side, which drags the union's centre off the body.
    # Measured on a mopping character: the mop layer spans x 0.10-0.91, pulling the midline to 0.503
    # while the body sat right of it, so BOTH arms (centroids 0.515 and 0.750) fell on the same side of
    # it — `_mirror_lobes` rejected them as "not a left and a right" and the character got no arms at
    # all. The head is on the midline by construction and cannot be pushed sideways by a held prop, so
    # prefer it; fall back to the union only when there is no head to measure.
    midline_x = ((head_box[0] + head_box[2]) / 2.0 if head_box
                 else (body_box[0] + body_box[2]) / 2.0)

    created: list[str] = []
    for layer in list(stack.layers):
        mesh = mesh_by_part.get(layer.id)
        if mesh is None:
            continue
        role = layer.semantic_role
        lobes = _mirror_lobes(mesh, midline_x)
        # A limb pair may be posed apart, so retry the junk drawer with the pose-aware tolerance — but
        # only if the result is recognisably the arms. Anything else off-level in the junk drawer stays
        # unpaired, and facial twins never get the loose tolerance at all.
        if lobes is None and role in _UNSORTED and head_box:
            posed = _mirror_lobes(mesh, midline_x, posed=True)
            if posed is not None and _looks_like_arms(mesh, posed, head_box=head_box, body_box=body_box):
                lobes = posed
        if lobes is None:
            continue

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


# --- FK segmentation (upper arm + forearm) ---------------------------------------------------------
# An arm handed back as one mesh can only rotate as one piece: a big shoulder swing shears the whole
# sheet, so the swing has to stay small and the limb reads as a stiff hinge. Cutting it at the elbow
# into two RIGID segments (upper arm + forearm) lets each rotate on its own joint without any
# intra-mesh shear — a real two-link FK chain — so the shoulder and elbow can move through a human
# range. The two segments overlap in a band straddling the elbow: the overlap is the same arm pixels
# in both halves, so as the forearm rotates the band covers the gap that would otherwise open at the
# joint. Legs get the same treatment at the knee.
_SEG_SUFFIX = ("_up", "_lo")     # upper (shoulder/hip side) / lower (wrist/ankle side) id suffixes
_SEG_OVERLAP_FRAC = 0.20         # overlap band half-height, as a fraction of the limb's height
_SEG_ELBOW_FRAC = 0.50           # the cut sits this fraction of the way down from the top: the elbow is
#                                  ~mid-arm (shoulder->wrist) and the knee is ~mid-leg (hip->sole), so
#                                  one fraction serves both. The 0.20 overlap band absorbs the variation.
_SEG_MIN_HEIGHT_FRAC = 0.10      # skip a limb shorter than this fraction of the character's height
# Arms AND legs both get a real two-link chain: an arm becomes upper-arm + forearm about the elbow, a
# leg becomes thigh + shin(+foot) about the knee. The authoring side is role-generic — it detects the
# ``_up``/``_lo`` segments by suffix and gives whichever role carries them the rigid FK treatment, with
# the per-role bend range (elbow vs knee) coming from the limb spec — so both flow through unchanged.
_SEG_ROLES: tuple[SemanticRole, ...] = (
    SemanticRole.arm_l, SemanticRole.arm_r, SemanticRole.leg_l, SemanticRole.leg_r,
)


def split_limb_segments(stack: LayerStack, meshes: list[Mesh]) -> list[str]:
    """Cut each arm and leg into an upper and a lower segment, overlapping at the joint (elbow / knee),
    for a real two-link FK chain. Both segments keep the limb's role and share its texture (UVs pick the
    region — nothing is written to disk); the lower segment is drawn just above the upper so its proximal
    overlap covers the joint seam. Mutates ``stack`` and ``meshes``; returns the ids created.

    Idempotent: a segment (id ending in ``_up``/``_lo``) is never re-split.
    """
    all_verts = [v for m in meshes for v in m.vertices]
    if not all_verts:
        return []
    fig_h = max(_bbox(all_verts)[3] - _bbox(all_verts)[1], 1e-6)
    mesh_by_part = {m.part_id: m for m in meshes}

    created: list[str] = []
    for layer in list(stack.layers):
        if layer.semantic_role not in _SEG_ROLES or layer.id.endswith(_SEG_SUFFIX):
            continue
        mesh = mesh_by_part.get(layer.id)
        if mesh is None:
            continue
        x0, y0, x1, y1 = _bbox(mesh.vertices)
        h = y1 - y0
        if h < _SEG_MIN_HEIGHT_FRAC * fig_h:
            continue                                     # too small to be a real, articulable limb
        cut = y1 - _SEG_ELBOW_FRAC * h                   # y-up: shoulder/hip at y1 (top), wrist/ankle at y0
        band = _SEG_OVERLAP_FRAC * h
        # upper keeps everything from the shoulder/hip down to just past the joint; lower keeps everything
        # from the wrist/ankle up to just past the joint — the [cut-band, cut+band] band is in both.
        upper = [i for i, (_, vy) in enumerate(mesh.vertices) if vy >= cut - band]
        fore = [i for i, (_, vy) in enumerate(mesh.vertices) if vy <= cut + band]
        segs = _segment_meshes(mesh, layer.id, upper, fore)
        if segs is None:
            continue                                     # a triangle spans past the band, or a degenerate half

        # lower segment drawn just above the upper (its proximal overlap hides the elbow/knee seam)
        halves = [
            Layer(id=f"{layer.id}{_SEG_SUFFIX[0]}", semantic_role=layer.semantic_role,
                  texture_path=layer.texture_path, draw_order=layer.draw_order,
                  width=layer.width, height=layer.height, bbox=layer.bbox),
            Layer(id=f"{layer.id}{_SEG_SUFFIX[1]}", semantic_role=layer.semantic_role,
                  texture_path=layer.texture_path, draw_order=layer.draw_order,
                  width=layer.width, height=layer.height, bbox=layer.bbox),
        ]
        i = stack.layers.index(layer)
        stack.layers[i:i + 1] = halves
        j = meshes.index(mesh)
        meshes[j:j + 1] = list(segs)
        for ly, m in zip(halves, segs):
            mesh_by_part[ly.id] = m
            created.append(ly.id)
    return created


def _segment_meshes(
    mesh: Mesh, base_id: str, upper: list[int], fore: list[int],
) -> tuple[Mesh, Mesh] | None:
    """Two overlapping sub-meshes (upper, lower) — or ``None`` if either is degenerate or a triangle
    falls outside both halves (which would leave a hole at the joint: the overlap band is too narrow)."""
    up = _sub_mesh(mesh, f"{base_id}{_SEG_SUFFIX[0]}", upper)
    lo = _sub_mesh(mesh, f"{base_id}{_SEG_SUFFIX[1]}", fore)
    if up is None or lo is None:
        return None
    # every source triangle must survive in at least one half — else the cut tears a hole in the limb
    up_set, fore_set = set(upper), set(fore)
    for a, b, c in mesh.triangles:
        tri = (a, b, c)
        if not (all(v in up_set for v in tri) or all(v in fore_set for v in tri)):
            return None
    return up, lo
