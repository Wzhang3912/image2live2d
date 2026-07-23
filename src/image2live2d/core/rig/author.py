"""Stage 4 — Rig authoring. The hard part: turn layers+meshes into deformers + parameters.

Approach = **template-retargeting** (CartoonAlive's proven playbook). The base rung of the quality
ladder is **template-transfer**: for each standard Live2D parameter we synthesize keyforms by
applying a canonical deformation to whichever parts are present, scaled to each part's own mesh
bounding box.

The Phase 3 rung is **landmark-corrected** (``landmarks`` argument): when per-character landmarks are
available (from ``core.landmark.extract_landmarks``), deformations fit *this* character instead of a
generic bbox — head turn pivots on the real face oval, blink collapses along the true lid line, the
mouth keys off real corners, pupils travel within the real eye, and limbs rotate about real joints.
Every landmark feature degrades gracefully: absent landmarks fall back to the bbox heuristics, so the
function works with ``landmarks=None``.

What we author for a portrait:
* ``ParamEyeLOpen`` / ``ParamEyeROpen`` — blink (collapse the eye group toward its centre line).
* ``ParamMouthOpenY`` — open the mouth into a lens-shaped cavity (lower lip drops, upper lip rises a
  little, corners anchored), not a flat jaw-slide.
* ``ParamAngleX`` / ``ParamAngleY`` — head turn via a single **shared pseudo-3D spherical warp**
  applied coherently to every head vertex (features near the turn axis shift most; the receding
  edge foreshortens). This is the effect a Live2D warp deformer produces, baked into keyforms so it
  stays backend-neutral instead of relying on a nijilive-specific deformer node.
* ``ParamAngleZ`` — head tilt via rigid rotation about the head centre.
* ``ParamEyeBallX`` / ``ParamEyeBallY`` — pupil look (translate pupils).
* ``ParamBrowLY`` / ``ParamBrowRY`` — brow raise.

Most motion is baked as per-vertex mesh offsets in keyforms, which keeps it backend-agnostic (the moc3
and nijilive runtimes read only the mesh offsets). The one exception is the head turn: it is *also*
authored as a WARP ``Deformer`` (``deformers`` is no longer empty), carried in the ``deformer_offsets``
channel that only the editable-project (.cmo3) backend consumes — the two runtime backends synthesize
their own head turn and ignore it, so the .moc3/.inp are unchanged. See the head-turn note below.
"""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass, field

from ..landmark import Landmarks
from ..structure.graph import (
    BODY,
    HEAD,
    build_rig_graph,
)
from ..structure.graph import BODY_ROLES as _BODY_ROLES
from ..structure.graph import HEAD_ROLES as _HEAD_ROLES
from ..structure.appendages import accessory_appendages, garment_appendages
from ..structure.skirt import skirt_cloth, skirt_zones
from ..structure.strands import hair_strands
from ..types import LayerStack
from ...irr.params import make_parameter
from ...irr.schema import Deformer, DeformerType, Keyform, Mesh, Parameter, SemanticRole, Vec2
from .head_rigidity import PROTECT as _PROTECT
from .head_rigidity import regions_from as _rigidity_regions
from .head_rigidity import rigidity_field as _rigidity_field

# --------------------------------------------------------------------------------------------------
# Template selection
# --------------------------------------------------------------------------------------------------


@dataclass
class Template:
    """A hand-rigged base rig (authored once in nijigenerate) used as a fitting target. For the
    template-transfer MVP the deformation logic lives in code, so ``path`` is nominal."""

    name: str
    path: str


def select_template(stack: LayerStack) -> Template:
    """Classify the input and pick the closest archetype template.

    Heuristic over which semantic parts are present: limbs/torso -> a full-body archetype, otherwise
    a portrait. author_rig adapts to whatever parts exist via presence-gating, so the archetype is
    mainly metadata + the future hook for template-specific tuning.
    """
    roles = {layer.semantic_role for layer in stack.layers}
    if roles & {SemanticRole.leg_l, SemanticRole.leg_r}:
        name = "fullbody"
    elif roles & {SemanticRole.torso, SemanticRole.arm_l, SemanticRole.arm_r}:
        name = "halfbody"
    else:
        name = "portrait_front"
    return Template(name=name, path=f"templates/{name}.inx")


def detect_landmarks(stack: LayerStack) -> dict[str, tuple[float, float]]:
    """Detect anime-face landmarks (and body pose, later) in model space.

    TODO(phase2): anime face landmark model to drive landmark-corrected keyforms. The
    template-transfer MVP in ``author_rig`` does not require this.
    """
    raise NotImplementedError("rig.detect_landmarks")


# --------------------------------------------------------------------------------------------------
# Authoring
# --------------------------------------------------------------------------------------------------

# The head/body role taxonomy now lives in core.structure.graph (imported above as _HEAD_ROLES /
# _BODY_ROLES) so the RigGraph owns it and this stage consumes it.

# Magnitudes at the extreme parameter value.
_YAW_MAX = math.radians(26.0)    # ParamAngleX rotation of the head sphere at its extreme
_PITCH_MAX = math.radians(20.0)  # ParamAngleY rotation of the head sphere at its extreme
_ANGLE_Z_DEG = 12.0   # head tilt degrees at its extreme
_BODY_TURN_X = math.radians(8.0)  # body sway at its extreme (range +-10)
_BODY_TURN_Y = math.radians(6.0)
_BODY_Z_DEG = 6.0     # body lean degrees at its extreme
# Head turn/tilt at the extreme, for the .cmo3 warp deformer only (moc3/nijilive synthesise their own).
# Magnitudes match the moc3 head-warp squash (HEAD_YAW/PITCH/ROLL) so the editable project turns like
# the runtime files.
_HEAD_TURN_X = 0.52   # yaw radians (horizontal squash) at full ParamAngleX
_HEAD_TURN_Y = 0.42   # pitch radians (vertical squash) at full ParamAngleY
_HEAD_Z_DEG = 20.0    # head roll degrees at full ParamAngleZ
_HEAD_WARP_ID = "deform_head_turn"
_HEAD_GRID = 4        # NxN control lattice over the head bbox (N-1 x N-1 warp segments)
# The head-turn squash is anchored on the FACE ball, not the whole head group: long hair inflates the
# union bbox and drops the pivot below the chin (the head then rotates about a point near the hair tips).
# So exclude hair when sizing the squash sphere — the grid still spans the head so hair is carried along.
# Mirrors the moc3 emitter's face-only pivot. See RIVAL_HARVEST_BACKLOG T1.
_HAIR_ROLES = frozenset({SemanticRole.hair_front, SemanticRole.hair_side, SemanticRole.hair_back})
_SWAY_TAPER = 2.0     # how the swing grows from root to tip, as depth**_SWAY_TAPER. Hair is a
#                       cantilever: it is *stiff where it is attached* and free at the ends, and a beam
#                       clamped at one end deflects quadratically along its length. The taper used to be
#                       linear (=1.0), which let the whole sheet shear sideways — a fringe slid bodily
#                       off the forehead and exposed the hairline, so the character read as balding. The
#                       real thing is far more tip-concentrated than linear: measured through the native
#                       core, a mid-strand vertex of Hiyori's bangs moves 0.11x its tip (a linear taper
#                       would give 0.5x). Squaring pins the roots to the scalp and puts the motion in the
#                       tips, where hair actually moves.
_HAIR_SWAY = 0.12     # hair-tip swing as fraction of strand length at +-1 (roots stay). Measured against
#                       the real thing: Hiyori's bang tip travels 0.0156 over a 0.161-long strand, i.e.
#                       ~10% of its own length. We were at 22% — twice a real rig — which swept the fringe
#                       far enough across the head to bare the hairline. Gentle, so hair reads as attached
#                       to the head rather than flying off it.
_HAIR_BOUNCE = 0.10   # hair-tip VERTICAL drop as fraction of strand length at +-1 (roots stay). A nod's
#                       secondary bob; slightly gentler than the horizontal sway so a nod reads as a
#                       settle, not a lurch. Driven by pitch through physics (see _hair_bounce).
_ACC_SWAY = 0.15      # accessory dangle: gentler than hair (an ornament sways subtly off its mount)
_GARMENT_SWAY = 0.20  # cape/sleeve dangle: between an ornament and a skirt hem (a bigger sheet of cloth)
_CLOTH_SWAY = 0.30    # skirt-hem swing as fraction of garment height at +-1 (waist stays)
_EYEBALL_FRAC = 0.25  # pupil shift as fraction of pupil bbox at +-1
_BROW_FRAC = 0.4      # brow shift as fraction of brow bbox height at +-1
_MOUTH_OPEN = 0.7     # lower-lip drop as fraction of mouth bbox height at 1
_UPPER_LIP_FRAC = 0.35  # upper-lip *rise* on open as a fraction of the lower-lip drop — the jaw does
#                         most of the opening, but a small upper-lip lift turns a jaw-slide into a
#                         lens-shaped cavity. Both lips taper to the (anchored) mouth corners.
_MOUTH_FORM = 0.35    # corner raise/lower as fraction of mouth height at +-1
_MOUTH_MIN_ASPECT = 0.55  # floor on the mouth's opening scale, as a fraction of its WIDTH. A decomposed
#                         closed mouth is a *stroke* — on a real character, 21x6 px — so its height is the
#                         line weight, not a mouth dimension. Scaling the open off that height parted the
#                         lips by 4 px: the mouth never visibly opened. Width is always the mouth's true
#                         extent, so it's the honest reference. The floor only binds when the layer is
#                         abnormally flat (exactly the broken case); a well-formed mouth blob keeps its
#                         own height and is unchanged.
_BLINK = 0.85         # collapse toward the lid line at 0 — *not* all the way. A full (1.0) collapse
#                       lands every vertex on the lid axis, so the triangles go zero-area and the eye
#                       does not merely close, it *vanishes* into blank skin: the lash line that a shut
#                       eye is actually drawn with disappears with it. Real rigs never degenerate the
#                       eye — measured through the native core, Hiyori's most-collapsed eye mesh still
#                       keeps 14.6% of its open height at ParamEyeLOpen=0 (several keep 90-100%). The
#                       residual here is what remains visible as the closed-eye lid line.

# Absolute displacement caps (model units, canvas ~= 1.0) that bound runaway warps on pathological
# silhouettes — far below QA's 0.6 runaway gate, far above normal motion (head-turn ~0.14, mouth
# ~0.05), so well-formed characters are untouched. See edge-case limit report (2026-06-30):
#  * head turn flew off on an asymmetric/occluded silhouette and on floor-length hair (huge head bbox)
#  * mouth-open scaled by an over-measured mouth-layer height (See-through mouths cover a big region)
_TURN_CAP = 0.25      # max per-vertex head/body-turn shift (uniform-scaled to preserve the warp shape)
_MOUTH_CAP = 0.10     # max per-vertex mouth open/form shift (a mouth never needs more)
# (skirt/footwear thresholds moved to core.structure.skirt, which owns the hem planner)
_DEFORM_CAP = 0.28    # final safety net: no keyform may shift any vertex more than this. Bounds the
#                       remaining magnitude runaways (blink/hair/cloth/brow) when See-through emits an
#                       oversized eye/hair layer, without touching well-formed motion (all < ~0.22).
_BREATH_SHIFT = 0.008 # crown's upward bob (model units) at breath 1; tapers to 0 at the feet (see _breath)
_ARM_DEG = 14.0       # arm swing degrees about the shoulder joint at the extreme
# Legs are close together (feet only ~0.08 of canvas apart), so a swing that would look fine on an arm
# converges the two feet past each other into a cross. A leg on a standing character barely moves
# anyway. Keep the swing small enough that opposite-phase legs stay short of crossing: 2*len*sin(deg)
# has to stay under the foot gap, which at ~0.4 leg length means well under ~6deg.
_LEG_DEG = 5.0        # leg swing degrees about the hip joint at the extreme
_ELBOW_DEG = 32.0     # forearm bend about the elbow at the extreme (lower segment ramps in)
_KNEE_DEG = 16.0      # lower-leg bend about the knee at the extreme (also swings the foot laterally)
_LIMB_BEND_BAND = 0.35  # fraction of the lower segment over which the bend ramps 0->full (a soft,
#                         gap-free fold at the joint rather than a hard crease on the continuous mesh)


@dataclass
class RigAuthoring:
    """Output of the authoring stage: deformers, populated parameters, and the part->deformer parenting
    (``part_deformers[part_id] = deformer_id``) for the IRR."""

    deformers: list[Deformer]
    parameters: list[Parameter]
    part_deformers: dict[str, str] = field(default_factory=dict)


def author_rig(
    stack: LayerStack,
    meshes: list[Mesh],
    template: Template,
    landmarks: Landmarks | None = None,
) -> RigAuthoring:
    """Synthesize standard-parameter keyforms for the parts present.

    Template-transfer by default; **landmark-corrected** when ``landmarks`` is supplied (each feature
    falls back to the bbox heuristic if its landmark is missing)."""
    lm = landmarks or Landmarks()
    mesh_by_part = {m.part_id: m for m in meshes}
    parts_by_role: dict[SemanticRole, list[str]] = defaultdict(list)
    for layer in stack.layers:
        if layer.id in mesh_by_part:
            parts_by_role[layer.semantic_role].append(layer.id)

    def members(*roles: SemanticRole) -> list[tuple[str, Mesh]]:
        out: list[tuple[str, Mesh]] = []
        for role in roles:
            for pid in parts_by_role.get(role, []):
                out.append((pid, mesh_by_part[pid]))
        return out

    params: list[Parameter] = []

    # --- Blink (per eye group) ------------------------------------------------------------------
    # Landmark-corrected: collapse the whole eye group toward the true lid axis (eye centroid y) so
    # lid + white + pupil close together along the real eye line, not each part's own bbox midline.
    left_eye = members(SemanticRole.eye_l, SemanticRole.eye_white_l, SemanticRole.pupil_l)
    if left_eye:
        axis_y = lm.eye_l.center[1] if lm.eye_l else None
        params.append(_blink("ParamEyeLOpen", left_eye,
                             members(SemanticRole.eye_closed_l), axis_y=axis_y))
    right_eye = members(SemanticRole.eye_r, SemanticRole.eye_white_r, SemanticRole.pupil_r)
    if right_eye:
        axis_y = lm.eye_r.center[1] if lm.eye_r else None
        params.append(_blink("ParamEyeROpen", right_eye,
                             members(SemanticRole.eye_closed_r), axis_y=axis_y))

    # --- Mouth open -----------------------------------------------------------------------------
    mouth = members(SemanticRole.mouth)
    cavity = members(SemanticRole.mouth_cavity)
    if mouth:
        if lm.mouth and lm.mouth.height > 1e-4:   # ignore a degenerate (collapsed) mouth landmark
            pivot_y = lm.mouth.center[1]
            height = max(lm.mouth.height, 1e-6)
            open_fn = lambda m: _open_lens(m, _MOUTH_OPEN, pivot_y, height)  # noqa: E731
        else:
            open_fn = lambda m: _drop_lower(m, _MOUTH_OPEN)                  # noqa: E731
        params.append(_mouth_open("ParamMouthOpenY", mouth, cavity, open_fn))
        params.append(_mouth_form("ParamMouthForm", mouth, mouth_lm=lm.mouth))

    # --- Classify accessories as head- vs body-mounted ------------------------------------------
    # Accessories aren't a fixed head/body role (a hair-bow vs a belt), so bind each to whichever
    # group its position implies: an ornament sitting in the head region must follow the head turn
    # (else it floats in place while the head turns away — the classic "detached bow"), while a
    # waist charm must follow the body. Reference is the FACE bbox expanded upward (headwear sits
    # above the forehead), NOT the full head bbox, so long back-hair draping to the waist doesn't
    # capture body accessories.
    # Kinematic parenting is read off the RigGraph now: each accessory rides whichever structural
    # group (head/body) its attachment point is nearest — the same rule as before, centralized there.
    graph = build_rig_graph(stack, meshes, landmarks)
    accessories = members(SemanticRole.accessory)
    head_acc = [(pid, m) for pid, m in accessories if graph.parent_of(pid) == HEAD]
    body_acc = [(pid, m) for pid, m in accessories if graph.parent_of(pid) == BODY]

    # --- Head turn & tilt (ParamAngleX/Y/Z) -----------------------------------------------------
    # Represented as a WARP DEFORMER over the head group, driven by ParamAngle*. The two runtime
    # backends (moc3 warp about the neck base, nijilive head-node rotation) synthesise their own head
    # turn and read neither Rig.deformers nor deformer_offsets, so this is invisible to them — the
    # ParamAngle* mesh_offsets stay empty and the .moc3/.inp are byte-identical. The editable-project
    # (.cmo3) backend, which has no synthesiser of its own, consumes this deformer. See _head_turn_warp.
    deformers: list[Deformer] = []
    part_deformers: dict[str, str] = {}
    head = members(*_HEAD_ROLES) + head_acc
    if head:
        # The grid spans the whole head (hair included); the squash is anchored on the face parts only.
        face = members(*(_HEAD_ROLES - _HAIR_ROLES))
        # Protected-region bboxes (eyes/nose/mouth): the warp keeps them rigid so they don't foreshorten
        # with the head — the same T5 fix the moc3 backend applies, sharing head_rigidity.PROTECT.
        prot = {}
        for _rn in _PROTECT:
            _pm = members(SemanticRole[_rn])
            if _pm:
                _pxs = [x for _, m in _pm for x, _ in m.vertices]
                _pys = [y for _, m in _pm for _, y in m.vertices]
                if _pxs:
                    prot[_rn] = (min(_pxs), min(_pys), max(_pxs), max(_pys))
        warp, turn_params = _head_turn_warp(
            [m for _, m in head], [m for _, m in face] or [m for _, m in head], prot)
        deformers.append(warp)
        params.extend(turn_params)
        for pid, _ in head:
            part_deformers[pid] = warp.id

    # --- Pupil look -----------------------------------------------------------------------------
    # Landmark-corrected: bound travel by the real eye size so pupils stay within the eye.
    pupils = members(SemanticRole.pupil_l, SemanticRole.pupil_r)
    if pupils:
        eye = lm.eye_l or lm.eye_r
        travel_x = _EYEBALL_FRAC * eye.width if eye else None
        travel_y = _EYEBALL_FRAC * eye.height if eye else None
        params.append(_eyeball("ParamEyeBallX", pupils, axis="x", travel=travel_x))
        params.append(_eyeball("ParamEyeBallY", pupils, axis="y", travel=travel_y))

    # --- Brow raise -----------------------------------------------------------------------------
    brow_l = members(SemanticRole.eyebrow_l)
    if brow_l:
        params.append(_brow("ParamBrowLY", brow_l))
    brow_r = members(SemanticRole.eyebrow_r)
    if brow_r:
        params.append(_brow("ParamBrowRY", brow_r))

    # --- Hair sway (physics OUTPUT params; the physics rig drives these) ------------------------
    # P2: one param per hair PART (strand), not one per role — so twin-tails / a ponytail + fringe
    # each swing on their own param (and their own pendulum). A single part of a role keeps the base
    # id, so single-strand characters are unchanged. See core.structure.strands.hair_strands.
    strands = hair_strands(stack, meshes)
    for spec in strands:
        params.append(_hair_sway(spec.param_id, spec.part_id,
                                 mesh_by_part[spec.part_id], spec.vertex_indices))

    # --- Hair BOUNCE (vertical), one param per hair ROLE (physics OUTPUT) ------------------------
    # The sway params above are horizontal-only, so a nod never bobs the hair through physics — it only
    # rides the head-turn deformation, which reads stiff. A vertical bounce param per role, driven by
    # pitch, lets the hair drop and settle on a nod. Per ROLE, not per strand (unlike sway): on a nod
    # every lobe of a role drops together, so the twin-tail welding that forced per-strand sway does not
    # apply — and it keeps this to at most three extra params (front/side/back). generate_physics wires
    # each to a ParamAngleY pendulum.
    for role in (SemanticRole.hair_front, SemanticRole.hair_side, SemanticRole.hair_back):
        role_strands = [s for s in strands if s.role is role]
        if role_strands:
            params.append(_hair_bounce(_hair_bounce_param(role), role_strands, mesh_by_part))

    # --- Cloth/skirt hem sway, L/C/R zones (physics OUTPUT params) -------------------------------
    # Which clothing is a swingable hem, and each zone's overlapping window, come from the shared
    # skirt planner (core.structure.skirt); the pendulum *material* is geometry-scaled there and read
    # by generate_physics. The sway keyform windows are unchanged, so this is byte-identical here.
    cloth = skirt_cloth(stack, meshes)
    for z in skirt_zones(stack, meshes):
        params.append(_skirt_zone(z.param_id, cloth, center_x=z.center_x, half_width=z.half_width))

    # --- Accessory dangle (physics OUTPUT params) -----------------------------------------------
    # Each accessory the graph bound to the head/body also gets a gentle pendulum, so a dangling
    # ornament swings as secondary motion (driven, in physics, by that same parent's turn/sway).
    for spec in accessory_appendages(stack, meshes, graph):
        params.append(_hair_sway(spec.param_id, spec.part_id,
                                 mesh_by_part[spec.part_id], None, amount=_ACC_SWAY))

    # --- Garment appendage sway (physics OUTPUT params) -----------------------------------------
    # A clothing part that hangs free (cape, long sleeve, coattail) — told from a rigid bodice by the
    # dynamics free-edge score — sways from its top edge like the skirt hem, but body-driven.
    for spec in garment_appendages(stack, meshes, graph):
        params.append(_hair_sway(spec.param_id, spec.part_id,
                                 mesh_by_part[spec.part_id], None, amount=_GARMENT_SWAY))

    # --- Body sway / lean (when a body is present) ----------------------------------------------
    body = members(*_BODY_ROLES) + body_acc
    if body:
        body_meshes = [m for _, m in body]
        bcenter = _union_center(body_meshes)
        bbox = _union_bbox(body_meshes)
        params.append(_head_turn("ParamBodyAngleX", body, bcenter, bbox, axis="x", amax=_BODY_TURN_X))
        params.append(_head_turn("ParamBodyAngleY", body, bcenter, bbox, axis="y", amax=_BODY_TURN_Y))
        params.append(_rotation("ParamBodyAngleZ", body, bcenter, deg=_BODY_Z_DEG))

    # --- Limb articulation (arms/legs swing about their joint) -----------------------------------
    # The joint comes from the limb's OWN split mesh, not from the landmark extractor. The de-cardboard
    # split leaves arm_l and arm_r sharing one full-canvas texture (both arms are in its alpha), so the
    # silhouette-based landmark reads the centroid of *both* arms — the body midline — and hands every
    # limb a pivot at its own inner edge. Rotating the left arm about a point at its far-right edge
    # swung the hand in a wide arc and tore it off the sleeve; legs flung their feet clean off. The mesh
    # carries only this side's triangles, so its bbox is the honest per-limb geometry — see _limb_joints.
    # Non-standard params (ParamArm*/ParamLeg*) — see irr.params.
    # Parts a limb might carry at its end (shoe, cuff): anything that isn't itself a limb. The geometry
    # filter in _limb_riders decides which limb, if any, actually claims each one.
    _LIMB_ROLES = frozenset({SemanticRole.arm_l, SemanticRole.arm_r,
                             SemanticRole.leg_l, SemanticRole.leg_r})
    all_limb_candidates = [(layer.id, mesh_by_part[layer.id]) for layer in stack.layers
                           if layer.id in mesh_by_part and layer.semantic_role not in _LIMB_ROLES]
    # garment parts a limb's overlapping region can ride (a jacket sleeve, a dress panel) — skinned to
    # the limb by _limb_follow so the sleeve rotates with the arm without splitting the clothing layer.
    clothing_candidates = [(layer.id, mesh_by_part[layer.id]) for layer in stack.layers
                           if layer.id in mesh_by_part and layer.semantic_role == SemanticRole.clothing]
    # The character's horizontal midline — the reference for "outward". A limb rotates the SAME absolute
    # direction for a given param sign, so without mirroring, driving both arms to +max lifts one and
    # drops the other: you literally cannot raise both arms at once. Negating the rotation for the limb
    # on the far side of the midline makes +param mean "lift/splay OUTWARD" on both sides — a body that
    # moves symmetrically, the way a real one does.
    _mid_x = _union_center(list(mesh_by_part.values()))[0] if mesh_by_part else 0.5
    for role, swing_id, bend_id, swing_deg, bend_deg in (
        (SemanticRole.arm_l, "ParamArmLA", "ParamArmLB", _ARM_DEG, _ELBOW_DEG),
        (SemanticRole.arm_r, "ParamArmRA", "ParamArmRB", _ARM_DEG, _ELBOW_DEG),
        (SemanticRole.leg_l, "ParamLegLA", "ParamLegLB", _LEG_DEG, _KNEE_DEG),
        (SemanticRole.leg_r, "ParamLegRA", "ParamLegRB", _LEG_DEG, _KNEE_DEG),
    ):
        limb = members(role)
        if not limb:
            continue
        joint, elbow, end = _limb_joints([m for _, m in limb])               # joint from the limb only
        # A limb has to carry whatever rides its distal end — the shoe at a foot, a cuff at a wrist —
        # or articulation moves the leg and leaves the shoe standing on the floor (the second half of
        # the leg-disconnect the render showed). Those parts are separate layers (footwear arrives as
        # its own "clothing"), so we find them by geometry: sitting at/below the limb's far end, within
        # its lateral span. They then swing and bend with the limb like the rest of it.
        riders = _limb_riders([m for _, m in limb], all_limb_candidates)
        rider_ids = {pid for pid, _ in riders}
        # A garment overlapping the limb (a jacket sleeve) rides its SWING with a tapered weight, so it
        # bends off the torso instead of the bare arm tearing out of a static sleeve.
        follow = _limb_follow([m for _, m in limb], joint, end, _mid_x,
                              [(pid, m) for pid, m in clothing_candidates if pid not in rider_ids])
        limb = limb + riders
        side = 1.0 if joint[0] >= _mid_x else -1.0    # +param lifts/splays OUTWARD on both sides
        params.append(_rotation(swing_id, limb, joint, deg=swing_deg * side, follow=follow))  # swing
        params.append(_limb_bend(bend_id, limb, elbow, end, deg=bend_deg * side))   # lower-segment bend

    # --- Breath (subtle whole-character bob) ----------------------------------------------------
    all_parts = [
        (layer.id, mesh_by_part[layer.id]) for layer in stack.layers if layer.id in mesh_by_part
    ]
    if all_parts:
        params.append(_breath("ParamBreath", all_parts))

    _cap_offsets(params, _DEFORM_CAP)  # final safety net against magnitude runaways (any param/part)
    return RigAuthoring(deformers=deformers, parameters=params, part_deformers=part_deformers)


def _cap_offsets(params: list[Parameter], cap: float) -> None:
    """Clamp every keyform's per-vertex offset magnitude to ``cap`` (in place), preserving direction.

    A blanket backstop so no parameter can fling a vertex across the canvas on a pathological
    silhouette (e.g. See-through's oversized eye/hair layers -> blink/sway runaway). Well-formed
    motion is far below ``cap`` and untouched; head-turn and mouth keep their own tighter caps.

    Limb params (ParamArm*/ParamLeg*) are EXEMPT: a full arm/leg swing or an elbow/knee bend
    legitimately moves the wrist/ankle far more than ``cap`` (that's the whole point), and each is
    already bounded by its own degree limit."""
    for p in params:
        if p.id.startswith("ParamArm") or p.id.startswith("ParamLeg"):
            continue
        for kf in p.keyforms:
            for pid, offs in kf.mesh_offsets.items():
                kf.mesh_offsets[pid] = [_cap_vec(dx, dy, cap) for dx, dy in offs]


def _cap_vec(dx: float, dy: float, cap: float) -> Vec2:
    mag = math.hypot(dx, dy)
    if mag > cap:
        s = cap / mag
        return (dx * s, dy * s)
    return (dx, dy)


# --------------------------------------------------------------------------------------------------
# Parameter builders
# --------------------------------------------------------------------------------------------------
def _set_keyforms(param_id: str, keyforms: list[Keyform]) -> Parameter:
    p = make_parameter(param_id)
    p.keyforms = keyforms
    return p


def _tri(param_id: str, at) -> Parameter:
    """Build a 3-keyform parameter at [min, default, max] from ``at(sign)`` where sign is -1/0/+1 and
    ``at(0)`` must yield zero offsets. Keyform values come from the parameter's own catalog range, so
    this works for ParamAngle* (+-30), ParamBodyAngle* (+-10), and +-1 params alike."""
    p = make_parameter(param_id)
    p.keyforms = [
        Keyform(value=p.min, mesh_offsets=at(-1.0)),
        Keyform(value=p.default, mesh_offsets=at(0.0)),
        Keyform(value=p.max, mesh_offsets=at(1.0)),
    ]
    return p


def _tri_deformer(param_id: str, deformer_id: str, at) -> Parameter:
    """Like :func:`_tri`, but the 3 keyforms carry ``deformer_offsets`` (per grid-vertex deltas for
    ``deformer_id``) instead of per-mesh offsets — for a parameter that drives a warp deformer."""
    p = make_parameter(param_id)
    p.keyforms = [
        Keyform(value=p.min, deformer_offsets={deformer_id: at(-1.0)}),
        Keyform(value=p.default, deformer_offsets={deformer_id: at(0.0)}),
        Keyform(value=p.max, deformer_offsets={deformer_id: at(1.0)}),
    ]
    return p


def _cap_cell(cell: list[Vec2]) -> list[Vec2]:
    """Uniform-scale a set of grid-vertex deltas so the largest is <= ``_TURN_CAP`` (preserves shape)."""
    mx = max((math.hypot(dx, dy) for dx, dy in cell), default=0.0)
    if mx > _TURN_CAP:
        s = _TURN_CAP / mx
        return [(dx * s, dy * s) for dx, dy in cell]
    return cell


def _head_turn_warp(
    head_meshes: list[Mesh], face_meshes: list[Mesh],
    protected: dict[str, tuple[float, float, float, float]] | None = None,
) -> tuple[Deformer, list[Parameter]]:
    """A head-turn WARP deformer for the .cmo3 backend: an ``N x N`` control lattice over the head bbox
    whose points move per ParamAngleX/Y/Z. Yaw/pitch are the same pivot-anchored pseudo-3D sphere squash
    as :func:`_head_turn` (features foreshorten around a fixed centre); roll is an in-plane rotation. The
    two runtime backends read neither this deformer nor its ``deformer_offsets``, so nothing changes for
    them (see the head-turn note in :func:`author_rig`).

    The lattice spans ``head_meshes`` (hair carried along) but the squash sphere is sized from
    ``face_meshes`` — anchoring the pivot on the face, not a hair-inflated union bbox (see ``_HAIR_ROLES``).
    ``protected`` (role -> bbox) keeps the eyes/nose/mouth rigid under the squash — the same T5 fix the
    moc3 warp applies, sharing :mod:`head_rigidity` so both backends foreshorten the face identically.
    """
    x0, y0, x1, y1 = _union_bbox(head_meshes)          # grid extent: the whole head
    fx0, fy0, fx1, fy1 = _union_bbox(face_meshes)       # squash anchor: the face ball only
    cx, cy = (fx0 + fx1) / 2.0, (fy0 + fy1) / 2.0
    n = _HEAD_GRID
    lattice: list[Vec2] = [
        (x0 + (x1 - x0) * c / (n - 1), y0 + (y1 - y0) * r / (n - 1))
        for r in range(n) for c in range(n)
    ]
    rx = max(cx - fx0, fx1 - cx, 1e-6)   # sphere radius = face half-extent (not the hair-inflated head)
    ry = max(cy - fy0, fy1 - cy, 1e-6)
    rigid = _rigidity_field(lattice, _rigidity_regions(protected or {}))

    def squash(a: float, axis: str) -> list[Vec2]:
        shift = (rx if axis == "x" else ry) * math.sin(a)   # anchor the pivot (see _head_turn)

        def delta(x: float, y: float) -> Vec2:
            if axis == "x":
                phi = math.asin(_clamp((x - cx) / rx, -1.0, 1.0))
                return (cx + rx * math.sin(phi + a) - x - shift, 0.0)
            psi = math.asin(_clamp((y - cy) / ry, -1.0, 1.0))
            return (0.0, cy + ry * math.sin(psi + a) - y - shift)

        cell: list[Vec2] = []
        for (x, y), (w, ccx, ccy) in zip(lattice, rigid):
            dx, dy = delta(x, y)
            if w > 0.0:
                # Rigid: the whole feature shifts by its centroid's OWN squash delta, so it never narrows.
                rdx, rdy = delta(ccx, ccy)
                dx, dy = dx * (1.0 - w) + rdx * w, dy * (1.0 - w) + rdy * w
            cell.append((dx, dy))
        return _cap_cell(cell)

    def roll(frac: float) -> list[Vec2]:
        ang = math.radians(_HEAD_Z_DEG) * frac
        ca, sa = math.cos(ang), math.sin(ang)
        cell = [(cx + (x - cx) * ca - (y - cy) * sa - x, cy + (x - cx) * sa + (y - cy) * ca - y)
                for x, y in lattice]
        return _cap_cell(cell)

    warp = Deformer(id=_HEAD_WARP_ID, type=DeformerType.warp, parent=None,
                    grid_rows=n, grid_cols=n, grid_vertices=lattice)
    params = [
        _tri_deformer("ParamAngleX", _HEAD_WARP_ID, lambda s: squash(s * _HEAD_TURN_X, "x")),
        _tri_deformer("ParamAngleY", _HEAD_WARP_ID, lambda s: squash(s * _HEAD_TURN_Y, "y")),
        _tri_deformer("ParamAngleZ", _HEAD_WARP_ID, roll),
    ]
    return warp, params


def _blink(param_id: str, group: list[tuple[str, Mesh]],
           closed_group: list[tuple[str, Mesh]] = (), *, axis_y: float | None = None) -> Parameter:
    # default (open) = 1.0 -> rest; 0.0 (closed) -> collapse to the lid line.
    # With a landmark lid axis (axis_y) the whole group collapses toward that shared y; otherwise
    # each part collapses toward its own bbox midline.
    #
    # When a synthesised closed-eye lash line exists (closed_group; see core.synth.eye) this becomes a
    # CROSSFADE: the open parts collapse AND fade out while the lash line fades in, so a shut eye is the
    # clean lash line, not the compressed sliver of iris+white a bare squash leaves (the _BLINK residual).
    # Needs the per-keyform opacity the moc3 emitter now honours. With no lash line it degrades to the
    # squash-only blink exactly as before (no opacity overrides emitted).
    if axis_y is None:
        collapsed = {pid: _collapse_vertical(m, _BLINK) for pid, m in group}
    else:
        collapsed = {pid: _collapse_to(m, _BLINK, axis_y) for pid, m in group}

    closed_off = dict(collapsed)
    open_off = {pid: _zeros(m) for pid, m in group}
    for pid, m in closed_group:                       # the lash line itself never moves — it only fades
        closed_off[pid] = _zeros(m)
        open_off[pid] = _zeros(m)

    closed_op: dict[str, float] = {}
    open_op: dict[str, float] = {}
    if closed_group:
        for pid, _ in group:                          # open parts: gone when shut, full when open
            closed_op[pid], open_op[pid] = 0.0, 1.0
        for pid, _ in closed_group:                   # lash line: full when shut, gone when open
            closed_op[pid], open_op[pid] = 1.0, 0.0

    closed_kf = Keyform(value=0.0, mesh_offsets=closed_off, opacity_overrides=closed_op)
    open_kf = Keyform(value=1.0, mesh_offsets=open_off, opacity_overrides=open_op)
    return _set_keyforms(param_id, [closed_kf, open_kf])


def _mouth_open(
    param_id: str,
    lips: list[tuple[str, Mesh]],
    cavity: list[tuple[str, Mesh]],
    open_fn,
) -> Parameter:
    """Open the mouth: the lips part into a lens *and* the synthesised cavity behind them grows from
    nothing into the gap they leave.

    The two groups move oppositely, which is why this can't be a ``_two_pose``. The cavity is collapsed
    **completely** when the mouth is shut — the degeneracy a blink must avoid is exactly what we want
    here, because a closed mouth has to render as the bare lip line it always was, with no trace of the
    part we painted behind it (see core.synth.mouth).
    """
    shut = Keyform(value=0.0, mesh_offsets={
        **{pid: _zeros(m) for pid, m in lips},
        **{pid: _collapse_vertical(m, 1.0) for pid, m in cavity},
    })
    opened = Keyform(value=1.0, mesh_offsets={
        **{pid: open_fn(m) for pid, m in lips},
        **{pid: _zeros(m) for pid, m in cavity},
    })
    return _set_keyforms(param_id, [shut, opened])


def _two_pose(
    param_id: str, rest_v: float, active_v: float, group: list[tuple[str, Mesh]], fn
) -> Parameter:
    rest = Keyform(value=rest_v, mesh_offsets={pid: _zeros(m) for pid, m in group})
    active = Keyform(value=active_v, mesh_offsets={pid: fn(m) for pid, m in group})
    return _set_keyforms(param_id, [rest, active])


def _head_turn(
    param_id: str,
    head: list[tuple[str, Mesh]],
    center: Vec2,
    bbox: tuple[float, float, float, float],
    *,
    axis: str,
    amax: float,
    neck: list[tuple[str, Mesh]] | None = None,
) -> Parameter:
    """A single shared pseudo-3D rotation, baked coherently into every vertex of the group.

    The group is modelled as a sphere centred on it; each vertex's signed offset from centre on the
    turn axis maps to an angle on that sphere, rotated by ``amax`` at the extreme. Vertices near the
    axis shift most and the receding edge foreshortens — the look a warp deformer gives, but
    backend-neutral. One shared centre/radius keeps parts coherent. Used for head (ParamAngleX/Y)
    and body (ParamBodyAngleX/Y).

    ``neck`` parts (optional) get a **tapered follow-through**: each neck vertex shifts by the head's
    displacement evaluated at the chin (``cx``, bbox-bottom), scaled by a vertical weight that is 1 at
    the neck top (so it stays joined to the head) and 0 at the shoulders (so the body still anchors
    it). Closes the head/neck seam on strong turns without detaching the shoulders. The reference is
    taken on-sphere (at the chin) so it is exactly zero at rest and never clamps.
    """
    cx, cy = center
    x0, y0, x1, y1 = bbox
    # Radius must contain every vertex *relative to the pivot center*, so no vertex falls outside the
    # sphere and gets clamped (which would yank far parts violently). When center is the bbox midpoint
    # this equals the half-extent; when the pivot is offset (e.g. a landmark face-oval center that sits
    # away from the parts' midline), the far side governs — keeping the warp bounded and artifact-free.
    if axis == "x":
        radius = max(cx - x0, x1 - cx, 1e-6)
    else:
        radius = max(cy - y0, y1 - cy, 1e-6)

    if neck:
        ny0 = min(y for _, m in neck for _, y in m.vertices)
        ny1 = max(y for _, m in neck for _, y in m.vertices)
        nspan = max(ny1 - ny0, 1e-6)

    def at(sign: float) -> dict[str, list[Vec2]]:
        a = sign * amax
        # Anchor the pivot: a bare sphere warp shifts the centre itself by radius·sin(a), so the whole
        # head *slides* — and an asymmetric silhouette or floor-length hair inflates `radius`, sliding
        # it right off the body. Subtracting the centre's own shift makes the head **rotate in place**
        # (centre stays put; features foreshorten around it), which is both more natural and removes
        # the slide that scaled with radius. Zero at rest (a=0 -> sin=0).
        center_shift = radius * math.sin(a)
        offs: dict[str, list[Vec2]] = {}
        for pid, m in head:
            cell: list[Vec2] = []
            for x, y in m.vertices:
                if axis == "x":
                    phi = math.asin(_clamp((x - cx) / radius, -1.0, 1.0))
                    cell.append((cx + radius * math.sin(phi + a) - x - center_shift, 0.0))
                else:
                    psi = math.asin(_clamp((y - cy) / radius, -1.0, 1.0))
                    cell.append((0.0, cy + radius * math.sin(psi + a) - y - center_shift))
            offs[pid] = cell
        if neck:
            if axis == "x":
                ref = 0.0                                                    # pivot anchored: no x slide
            else:
                psi0 = math.asin(_clamp((y0 - cy) / radius, -1.0, 1.0))
                ref = cy + radius * math.sin(psi0 + a) - y0 - center_shift   # residual chin dy
            for pid, m in neck:
                cell = []
                for x, y in m.vertices:
                    w = _clamp((y - ny0) / nspan, 0.0, 1.0)                  # 0 shoulders -> 1 top
                    cell.append((ref * w, 0.0) if axis == "x" else (0.0, ref * w))
                offs[pid] = cell
        # Bound runaway: a head sphere sized to a huge group (long hair) or skewed by an asymmetric
        # silhouette can fling vertices off. Uniform-scale the whole warp so its largest shift is
        # <= _TURN_CAP, preserving the warp's shape (relative motion) while capping its magnitude.
        mx = max((math.hypot(dx, dy) for cell in offs.values() for dx, dy in cell), default=0.0)
        if mx > _TURN_CAP:
            s = _TURN_CAP / mx
            offs = {pid: [(dx * s, dy * s) for dx, dy in cell] for pid, cell in offs.items()}
        return offs

    return _tri(param_id, at)


def _rotation(param_id: str, head: list[tuple[str, Mesh]], center: Vec2, *, deg: float,
              neck: list[tuple[str, Mesh]] | None = None,
              follow: list[tuple[str, Mesh, list[float]]] | None = None) -> Parameter:
    """Rigid roll about ``center`` by ``deg`` at the extreme (ParamAngleZ / ParamBodyAngleZ).

    ``neck`` parts (optional) get a **tapered twist**: each neck vertex rotates about the same centre
    by ``deg`` scaled by a vertical weight (1 at the neck top → follows the head roll, 0 at the
    shoulders → stays), so the head stays joined to the neck when it tilts (matching the head-turn
    neck follow).

    ``follow`` parts (optional) get a **per-vertex weighted** rotation about the same centre — used to
    make an overlapping garment (a jacket sleeve, a dress panel) ride a limb without splitting the
    layer: weight 1 over the limb rotates it rigidly with the arm, tapering to 0 at the torso seam so
    the continuous garment mesh bends there instead of tearing."""
    cx, cy = center
    if neck:
        ny0 = min(y for _, m in neck for _, y in m.vertices)
        ny1 = max(y for _, m in neck for _, y in m.vertices)
        nspan = max(ny1 - ny0, 1e-6)

    def _weighted(m: Mesh, weights: list[float], theta: float) -> list[Vec2]:
        cell = []
        for (x, y), w in zip(m.vertices, weights):
            a = theta * w
            c, s = math.cos(a), math.sin(a)
            rx, ry = x - cx, y - cy
            cell.append((cx + rx * c - ry * s - x, cy + rx * s + ry * c - y))
        return cell

    def at(sign: float) -> dict[str, list[Vec2]]:
        theta = sign * math.radians(deg)
        offs = {pid: _rotate(m, center, theta) for pid, m in head}
        if neck:
            for pid, m in neck:
                weights = [_clamp((y - ny0) / nspan, 0.0, 1.0) for _, y in m.vertices]
                offs[pid] = _weighted(m, weights, theta)
        if follow:
            for pid, m, weights in follow:
                offs[pid] = _weighted(m, weights, theta)
        return offs

    return _tri(param_id, at)


def _limb_bend(param_id: str, limb: list[tuple[str, Mesh]], elbow: Vec2, end: Vec2, *,
               deg: float) -> Parameter:
    """Bend the LOWER segment of a limb about its ``elbow``/knee joint (elbow->wrist / knee->ankle).

    Unlike ``_rotation`` (rigid whole-limb swing about the shoulder/hip), this rotates only vertices
    *below* the joint, by an angle weighted 0 above the joint -> full over the segment. The limb is a
    single continuous mesh, so a weighted rotation folds it at the joint with **no gap** (splitting it
    into two rigid cut-outs would tear). ``elbow``/``end`` are model-space (y-up); the ramp runs along
    the vertical limb axis, which fits a hanging arm/leg."""
    ex, ey = elbow
    _, wy = end
    span = max(ey - wy, 1e-6)                       # elbow(top) -> wrist(bottom), y-up so ey > wy
    band = max(_LIMB_BEND_BAND * span, 1e-6)

    def at(sign: float) -> dict[str, list[Vec2]]:
        theta = sign * math.radians(deg)
        offs: dict[str, list[Vec2]] = {}
        for pid, m in limb:
            cell: list[Vec2] = []
            for x, y in m.vertices:
                w = _clamp((ey - y) / band, 0.0, 1.0)   # 0 at/above the joint -> 1 down the segment
                a = theta * w
                c, s = math.cos(a), math.sin(a)
                rx, ry = x - ex, y - ey
                cell.append((ex + rx * c - ry * s - x, ey + rx * s + ry * c - y))
            offs[pid] = cell
        return offs

    return _tri(param_id, at)


def _hair_sway(param_id: str, part_id: str, mesh: Mesh, indices: list[int] | None = None,
               *, amount: float = _HAIR_SWAY) -> Parameter:
    """Pendulum OUTPUT param for one hair strand: tips swing horizontally, roots (top) stay. The
    physics rig drives this from head/body motion; it is also a normal driveable parameter.

    ``indices`` (a strand's own vertices, e.g. one lobe of a shared mesh) restricts the sway to those
    vertices — the rest of the part is held at zero — so lobes of one part swing independently and the
    strand hangs from its *own* top. ``None`` sways the whole mesh (the single-lobe case).

    The swing grows as ``depth ** _SWAY_TAPER`` — see that constant. A linear taper let the whole sheet
    shear, which slid a fringe off the forehead."""
    verts = mesh.vertices
    owned = range(len(verts)) if indices is None else indices
    _, bottom, _, top = _bbox([verts[i] for i in owned])
    length = max(top - bottom, 1e-6)

    def at(sign: float) -> dict[str, list[Vec2]]:
        cell: list[Vec2] = [(0.0, 0.0)] * len(verts)
        for vi in owned:
            _, y = verts[vi]
            depth = (top - y) / length                  # 0 at the root, 1 at the tip
            cell[vi] = (sign * amount * length * depth ** _SWAY_TAPER, 0.0)
        return {part_id: cell}

    return _tri(param_id, at)


_HAIR_BOUNCE_PARAM = {
    SemanticRole.hair_front: "ParamHairFrontV",
    SemanticRole.hair_side: "ParamHairSideV",
    SemanticRole.hair_back: "ParamHairBackV",
}


def _hair_bounce_param(role: SemanticRole) -> str:
    return _HAIR_BOUNCE_PARAM[role]


def _hair_bounce(param_id: str, strands, mesh_by_part, *, amount: float = _HAIR_BOUNCE) -> Parameter:
    """Per-role VERTICAL hair-bounce OUTPUT param: on a nod the tips drop straight down and settle,
    roots (top) stay. The sibling of ``_hair_sway`` on the Y axis — same tip-weighted taper, but the
    offset is ``(0, dy)`` instead of ``(dx, 0)`` — so pitch physics can bob the hair, which a horizontal
    sway param cannot express. Covers every strand of the role in one param (a nod drops them together).

    +1 drops the tips (−y is down in this y-up model); the pendulum drives it both ways, so the sign is
    only a convention. Each strand droops from its *own* top, using its owned vertices when the part is a
    shared multi-lobe mesh."""
    def at(sign: float) -> dict[str, list[Vec2]]:
        cells: dict[str, list[Vec2]] = {}
        for spec in strands:
            mesh = mesh_by_part[spec.part_id]
            verts = mesh.vertices
            owned = range(len(verts)) if spec.vertex_indices is None else spec.vertex_indices
            _, bottom, _, top = _bbox([verts[i] for i in owned])
            length = max(top - bottom, 1e-6)
            cell = cells.setdefault(spec.part_id, [(0.0, 0.0)] * len(verts))
            for vi in owned:
                _, y = verts[vi]
                depth = (top - y) / length                 # 0 at the root, 1 at the tip
                cell[vi] = (0.0, -sign * amount * length * depth ** _SWAY_TAPER)
        return cells

    return _tri(param_id, at)


def _skirt_zone(
    param_id: str, group: list[tuple[str, Mesh]], *, center_x: float, half_width: float
) -> Parameter:
    """One skirt-hem zone OUTPUT param: the hem swings horizontally, weighted by a triangular window
    centred on ``center_x`` (width ``2*half_width``) so only this zone's strip moves and adjacent
    zones blend. Roots (top/waist) stay; tips (hem) swing most. Driven by the physics rig from the
    nearest lower-body motion."""
    hw = max(half_width, 1e-6)

    def at(sign: float) -> dict[str, list[Vec2]]:
        offs: dict[str, list[Vec2]] = {}
        for pid, m in group:
            _, _, _, top = _bbox(m.vertices)
            offs[pid] = [
                (sign * _CLOTH_SWAY * (top - y) * max(0.0, 1.0 - abs(x - center_x) / hw), 0.0)
                for x, y in m.vertices
            ]
        return offs

    return _tri(param_id, at)


def _mouth_form(param_id: str, group: list[tuple[str, Mesh]], *, mouth_lm=None) -> Parameter:
    """Smile/frown: raise the mouth corners at +1, lower them at -1. Vertical offset grows with a
    vertex's horizontal distance from the mouth centre, so the corners move and the centre stays.

    With a mouth landmark, the centre x and the corner span come from the real corners; otherwise
    from each part's bbox."""
    def at(sign: float) -> dict[str, list[Vec2]]:
        offs: dict[str, list[Vec2]] = {}
        for pid, m in group:
            if mouth_lm is not None:
                cx = mouth_lm.center[0]
                half = max(mouth_lm.width / 2.0, 1e-6)
                height = max(mouth_lm.height, 1e-6)
            else:
                x0, y0, x1, y1 = _bbox(m.vertices)
                cx = (x0 + x1) / 2.0
                half = max((x1 - x0) / 2.0, 1e-6)
                height = y1 - y0
            offs[pid] = [
                (0.0, sign * min(_MOUTH_FORM * height * abs(x - cx) / half, _MOUTH_CAP))
                for x, _ in m.vertices
            ]
        return offs

    return _tri(param_id, at)


def _breath(param_id: str, group: list[tuple[str, Mesh]]) -> Parameter:
    """Subtle breathing: the upper body rises at breath=1 while the FEET stay planted.

    A uniform whole-character translate lifted the feet off the ground — a float/hop, not a breath (a
    full-body character visibly levitated on every inhale). Real breathing expands the chest and barely
    lifts the head; the ground contact never moves. So weight the upward shift by height over the
    figure's own vertical extent: 0 at the feet, full at the crown. (Amplitude also pulled back toward
    the ~0.1%-of-canvas a real rig like Hiyori uses; the old 1.2% read as a hop.)"""
    ys = [y for _, m in group for _, y in m.vertices]
    rest = Keyform(value=0.0, mesh_offsets={pid: _zeros(m) for pid, m in group})
    if not ys:
        return _set_keyforms(param_id, [rest])
    y0, span = min(ys), max(max(ys) - min(ys), 1e-6)
    inhale = Keyform(value=1.0, mesh_offsets={
        pid: [(0.0, _BREATH_SHIFT * _clamp((y - y0) / span, 0.0, 1.0)) for _, y in m.vertices]
        for pid, m in group})
    return _set_keyforms(param_id, [rest, inhale])


def _eyeball(
    param_id: str, pupils: list[tuple[str, Mesh]], *, axis: str, travel: float | None = None
) -> Parameter:
    """Pupil look. ``travel`` (model units) is the absolute shift at the extreme; when given (from
    the real eye size) pupils stay inside the eye. Falls back to a fraction of the pupil bbox when no
    usable eye size is given — including a *degenerate* (~0) landmark eye, which would otherwise leave
    the pupils dead (caught by the all-params audit on a character whose eye landmark collapsed)."""
    def at(sign: float) -> dict[str, list[Vec2]]:
        offs: dict[str, list[Vec2]] = {}
        for pid, m in pupils:
            if travel and travel > 1e-6:
                d = sign * travel
            else:
                x0, y0, x1, y1 = _bbox(m.vertices)
                extent = (x1 - x0) if axis == "x" else (y1 - y0)
                d = sign * extent * _EYEBALL_FRAC
            offs[pid] = _translate(m, d if axis == "x" else 0.0, d if axis == "y" else 0.0)
        return offs

    return _tri(param_id, at)


def _brow(param_id: str, group: list[tuple[str, Mesh]]) -> Parameter:
    def at(sign: float) -> dict[str, list[Vec2]]:
        offs: dict[str, list[Vec2]] = {}
        for pid, m in group:
            _, y0, _, y1 = _bbox(m.vertices)
            offs[pid] = _translate(m, 0.0, sign * (y1 - y0) * _BROW_FRAC)
        return offs

    return _tri(param_id, at)


# --------------------------------------------------------------------------------------------------
# Geometry primitives — each returns offsets aligned to mesh.vertices order
# --------------------------------------------------------------------------------------------------
def _bbox(verts: list[Vec2]) -> tuple[float, float, float, float]:
    xs = [x for x, _ in verts]
    ys = [y for _, y in verts]
    return min(xs), min(ys), max(xs), max(ys)


def _zeros(m: Mesh) -> list[Vec2]:
    return [(0.0, 0.0)] * len(m.vertices)


def _translate(m: Mesh, dx: float, dy: float) -> list[Vec2]:
    return [(dx, dy)] * len(m.vertices)


def _collapse_vertical(m: Mesh, amount: float) -> list[Vec2]:
    _, y0, _, y1 = _bbox(m.vertices)
    cy = (y0 + y1) / 2.0
    return [(0.0, (cy - y) * amount) for _, y in m.vertices]


def _collapse_to(m: Mesh, amount: float, cy: float) -> list[Vec2]:
    """Collapse vertices vertically toward a shared line ``cy`` (the landmark lid axis)."""
    return [(0.0, (cy - y) * amount) for _, y in m.vertices]


def _open_lens(m: Mesh, amount: float, pivot_y: float, height: float) -> list[Vec2]:
    """A **lens-shaped** mouth open about the lip line ``pivot_y``: the lower lip drops and the upper lip
    rises a smaller amount (``_UPPER_LIP_FRAC``), both tapering to zero at the mouth *corners* so the
    opening reads as a cavity, not a jaw-slide. The horizontal taper uses the mesh's own bbox (anchoring
    its true corners), so it's robust to a degenerate-width landmark; ``pivot_y``/``height`` come from the
    landmark (or the bbox mid/extent in the fallback). Every shift is capped at ``_MOUTH_CAP`` to bound an
    over-measured See-through mouth layer.

    The opening scale is floored against the mouth's **width** (see ``_MOUTH_MIN_ASPECT``) because a
    decomposed *closed* mouth is a stroke — its height is the line weight, not the mouth.
    """
    x0, _, x1, _ = _bbox(m.vertices)
    cx = (x0 + x1) / 2.0
    half = max((x1 - x0) / 2.0, 1e-6)
    scale = max(height, _MOUTH_MIN_ASPECT * (x1 - x0))
    out: list[Vec2] = []
    for x, y in m.vertices:
        taper = max(0.0, 1.0 - abs(x - cx) / half)          # 1 at centre -> 0 at/beyond the corners
        if y < pivot_y:                                     # lower lip: drops
            factor = min(1.0, (pivot_y - y) / scale)
            out.append((0.0, -min(amount * scale * factor * taper, _MOUTH_CAP)))
        else:                                               # upper lip: smaller rise
            factor = min(1.0, (y - pivot_y) / scale)
            out.append((0.0, min(_UPPER_LIP_FRAC * amount * scale * factor * taper, _MOUTH_CAP)))
    return out


def _drop_lower(m: Mesh, amount: float) -> list[Vec2]:
    """Lens-shaped open with no landmark — pivot at the bbox mid-line, scale from the bbox height."""
    _, y0, _, y1 = _bbox(m.vertices)
    cy = (y0 + y1) / 2.0
    return _open_lens(m, amount, pivot_y=cy, height=max(y1 - y0, 1e-6))


def _rotate(m: Mesh, center: Vec2, theta: float) -> list[Vec2]:
    cx, cy = center
    c, s = math.cos(theta), math.sin(theta)
    out: list[Vec2] = []
    for x, y in m.vertices:
        rx, ry = x - cx, y - cy
        nx = cx + rx * c - ry * s
        ny = cy + rx * s + ry * c
        out.append((nx - x, ny - y))
    return out


def _union_bbox(meshes: list[Mesh]) -> tuple[float, float, float, float]:
    boxes = [_bbox(m.vertices) for m in meshes]
    return (
        min(b[0] for b in boxes),
        min(b[1] for b in boxes),
        max(b[2] for b in boxes),
        max(b[3] for b in boxes),
    )


def _union_center(meshes: list[Mesh]) -> Vec2:
    x0, y0, x1, y1 = _union_bbox(meshes)
    return ((x0 + x1) / 2.0, (y0 + y1) / 2.0)


# How much of the limb's height is sampled around a joint to locate the limb's axis there. The result
# is insensitive to this: every band from 0.00 (the single top row) to 0.30 puts all 34 measured
# shoulders inside their limb. 0.10 is picked for robustness at both ends — an exact top row rides on
# whichever stray vertex happens to be highest, and a wide band drags the shoulder down toward the elbow.
_JOINT_BAND = 0.10


def _limb_joints(meshes: list[Mesh]) -> tuple[Vec2, Vec2, Vec2]:
    """``(shoulder/hip, elbow/knee, wrist/ankle)`` down the limb's own **silhouette axis**.

    A hanging arm or leg attaches at the top of its silhouette and dangles down, so the joints sit at
    the top, middle and bottom of it. Taking them from the limb's *mesh* (which holds only this side's
    triangles after the de-cardboard split) is what keeps the left arm pivoting on its own shoulder
    instead of the body midline — the landmark silhouette can't, because the split halves share one
    two-armed texture.

    Their **x** used to be the limb's bbox centre, and that is wrong for any limb that is not a vertical
    bar. An arm attaches at the shoulder and hangs down and *outward*, so the bbox centre is somewhere
    out in the middle of the arc — measured on the 8 real characters, **16 of 34 limb pivots landed
    outside the limb they rotate**, including both arms on 7 of 8. A rotation about a point in empty
    space swings the limb wide instead of turning it in place, which is the same family as the
    shoe-and-leg disconnect the render oracle caught earlier.

    So each joint takes the centroid of the limb's own cross-section at that height (a band of
    ``_JOINT_BAND`` of the limb's height) — the silhouette's medial axis, sampled where the joint
    actually is. That puts every shoulder/hip inside its limb (16 outside -> 0) and most wrists/ankles
    too (16 -> 4); the elbow/knee was already inside, since a limb's middle is near its bbox centre.
    """
    verts = [v for m in meshes for v in m.vertices]
    x0, y0, x1, y1 = _union_bbox(meshes)
    top, bot = y1, y0                                   # y-up: shoulder/hip at the top, wrist/ankle low
    height = max(top - bot, 1e-9)

    def axis_x(y: float) -> float:
        band = [x for x, vy in verts if abs(vy - y) <= _JOINT_BAND * height]
        return sum(band) / len(band) if band else (x0 + x1) / 2.0

    mid = (top + bot) / 2.0
    return (axis_x(top), top), (axis_x(mid), mid), (axis_x(bot), bot)


# A rider (shoe/cuff) attaches at the limb's distal END: its top sits in a band around the limb's
# bottom — a little above (it overlaps the ankle) down to a quarter of the limb below it. That the top
# must be *near* the end, not merely below it, is what stops a shoe from also attaching to the arm two
# body-lengths up: the shoe is below the arm, but nowhere near the wrist.
_RIDER_END_FRAC = 0.25      # how far below the limb's end a rider's top may start
_RIDER_OVERLAP = 0.15       # how far it may reach up into the limb (overlap at the ankle)


def _limb_riders(
    limb_meshes: list[Mesh], candidates: list[tuple[str, Mesh]],
) -> list[tuple[str, Mesh]]:
    """The parts that hang off a limb's far end and must move with it (a shoe at a foot, a cuff at a
    wrist). Chosen purely by geometry so it doesn't depend on a footwear role the decomposer may not
    label: the part's top sits at the limb's distal end, laterally within the limb's own column."""
    lx0, ly0, lx1, ly1 = _union_bbox(limb_meshes)
    h = ly1 - ly0
    hi = ly0 + _RIDER_END_FRAC * h                      # top may start this far below the end...
    lo = ly0 - _RIDER_OVERLAP * h                       # ...up to this far into the limb (the ankle)
    riders: list[tuple[str, Mesh]] = []
    for pid, m in candidates:
        cx0, cy0, cx1, cy1 = _bbox(m.vertices)
        ccx = (cx0 + cx1) / 2.0
        if lo <= cy1 <= hi and lx0 <= ccx <= lx1:      # top at the limb's end, and in its column
            riders.append((pid, m))
    return riders


# A garment (jacket sleeve, dress panel) rides a limb where it OVERLAPS it. Unlike a rider (which sits
# at the distal end and moves rigidly), a sleeve overlaps the whole limb, so it gets a per-vertex weight:
# 1 over the limb (rotates with the arm), tapering to 0 in a band at the shoulder/hip seam so the
# continuous garment mesh bends there instead of tearing off the torso. This is how a jacket sleeve
# follows its arm without splitting the clothing layer.
_FOLLOW_SEAM_BAND = 0.35    # taper zone at the joint, as a fraction of the limb's height
_FOLLOW_X_OUT = 1.1         # widen the limb's column OUTWARD (away from midline) by this ×its width —
_FOLLOW_X_IN = 0.15         # a puffy sleeve is wider than the bare arm; the INNER edge stays tight (torso)


def _limb_follow(
    limb_meshes: list[Mesh], joint: Vec2, end: Vec2, mid_x: float,
    candidates: list[tuple[str, Mesh]],
) -> list[tuple[str, Mesh, list[float]]]:
    """Per-vertex weights so garment parts overlapping a limb ride its swing. Returns
    ``[(pid, mesh, weights)]`` for parts with meaningful overlap; a torso-only part gets all-zero
    weights and is dropped."""
    lx0, _, lx1, _ = _union_bbox(limb_meshes)
    lw = max(lx1 - lx0, 1e-6)
    jy = joint[1]                                       # shoulder/hip (top, y-up)
    _, ey = end                                         # wrist/ankle (bottom)
    span = max(jy - ey, 1e-6)
    band = max(_FOLLOW_SEAM_BAND * span, 1e-6)
    # widen the column outward (the sleeve flares away from the body), keep the inner edge tight (torso)
    if (lx0 + lx1) / 2.0 >= mid_x:                      # limb on the right of the body -> flare right
        gx0, gx1 = lx0 - _FOLLOW_X_IN * lw, lx1 + _FOLLOW_X_OUT * lw
    else:
        gx0, gx1 = lx0 - _FOLLOW_X_OUT * lw, lx1 + _FOLLOW_X_IN * lw
    ylo, yhi = ey - 0.10 * span, jy + band
    out: list[tuple[str, Mesh, list[float]]] = []
    for pid, m in candidates:
        weights: list[float] = []
        mw, n = 0.0, 0
        for x, y in m.vertices:
            if gx0 <= x <= gx1 and ylo <= y <= yhi:
                w = _clamp((jy - y) / band, 0.0, 1.0)   # 0 at the seam -> 1 over the limb
            else:
                w = 0.0
            weights.append(w)
            if w > 0.05:
                n += 1
            mw = max(mw, w)
        if mw > 0.3 and n >= 3:
            out.append((pid, m, weights))
    return out


def _clamp(v: float, lo: float, hi: float) -> float:
    return lo if v < lo else hi if v > hi else v
