"""IRR ``Rig`` -> nijilive puppet JSON.

Field names, key order, value shapes, and sentinels here are taken verbatim from the nijilive
source (``core/puppet.d``, ``nodes/node.d``, ``nodes/drawable.d``, ``core/meshdata.d``,
``nodes/part/package.d``, ``param/package.d``, ``param/binding.d``). The loader matches by key, so
order is cosmetic, but shapes are strict: a deform binding's ``values`` grid must be
``[xAxisPoints][yAxisPoints][vertexCount][2]`` and match ``axis_points`` exactly, or nijilive's
loader throws.

Coordinate handling: IRR geometry is in normalized model space with **y up** (Live2D-native, so
Route A stays natural). nijilive's camera is orthographic with ``top=0, bottom=height`` — i.e.
**y down** — so this emitter negates y on vertex positions and deform offsets. Positions/offsets are
scaled by ``scale``; UVs are left in [0, 1] (v already down, matching the texture).
"""

from __future__ import annotations

import struct
import zlib
from dataclasses import dataclass
from pathlib import Path

from ...core.structure.graph import HEAD_ROLES as _HEAD_ROLES
from ...irr.schema import Mesh, Parameter, Part, Rig, SemanticRole

NO_TEXTURE = 4294967295  # uint.max — empty texture slot / thumbnail sentinel
DEFAULT_SCALE = 1000.0  # pixels per normalized unit

# Auto-physics tuning (SimplePhysics is transform-driven; an anchor node moves with the driver
# param and its pendulum drives the hair output param).
#
# FEEL-PARITY (P0) — keep this pendulum in step with the Cubism one in
# backends/live2d/physics3.py::_vertices. The two runtimes parameterize the pendulum differently
# (Cubism: Mobility/Delay/Acceleration; nijilive: gravity/length/angle-damping), so tools/feel_parity.py
# is the oracle: it simulates THIS pendulum (natural frequency, damping ratio, settle) and range-checks
# the Cubism vertices against real pro physics3.json. As tuned, this pendulum swings at ~1.2-1.3 Hz with
# hair lightly damped (zeta ~0.24) and skirt more damped (~0.38, settles ~1s vs hair ~2s) — the correct
# feel ordering. The absolute swing magnitude across the two runtimes is the one thing the oracle can't
# self-confirm; it's eyeballed side by side in nijigenerate + Cubism Viewer.
_ANCHOR_SHIFT = 60.0          # px the anchor translates at the driver param's extreme
_PHYS_GRAVITY = 9.8           # SimplePhysics scalar gravity
_PHYS_LENGTH_SCALE = 120.0    # IRR pendulum length (~1) -> px
_PHYS_OUTPUT_SCALE = 3.0      # pendulum angle -> output param gain
_PHYS_LENGTH_DAMP = 0.5

# IRR PhysicsModel value -> nijilive enum member name (PascalCase, for nijigenerate beta2).
_MODEL_NAMES = {"pendulum": "Pendulum", "spring_pendulum": "SpringPendulum"}


@dataclass
class PuppetBuild:
    """Result of mapping a Rig: the puppet dict plus the PNG bytes for each texture slot (in
    order)."""

    puppet: dict
    textures: list[bytes]


def build_puppet(rig: Rig, *, asset_root: str | Path | None = None, scale: float = DEFAULT_SCALE) -> PuppetBuild:
    root = Path(asset_root) if asset_root is not None else None
    uuids = _UuidAllocator()

    # Texture slots: index by position in rig.textures; load bytes (or synthesize a placeholder).
    slot_of: dict[str, int] = {t.id: i for i, t in enumerate(rig.textures)}
    texture_blobs: list[bytes] = [_load_texture(t, root) for t in rig.textures]

    # Split parts into the head group (moves as one rigid unit) and everything else.
    drawn = [(p, rig.mesh_for(p.id)) for p in rig.parts_in_draw_order() if rig.mesh_for(p.id)]
    head_ids = head_group_ids(drawn)
    head_parts = [(p, m) for p, m in drawn if p.id in head_ids]
    body_parts = [(p, m) for p, m in drawn if p.id not in head_ids]
    body_ids = {p.id for p, _ in body_parts}

    def _nij_pivot(parts):  # bottom-centre in nijilive coords (y-down): (centre-x, bottom-y)
        nx = [x * scale for _, m in parts for x, _ in m.vertices]
        ny = [-y * scale for _, m in parts for _, y in m.vertices]
        return ((min(nx) + max(nx)) / 2.0, max(ny))

    # Head rotates about the neck; the body sways about the feet (ground) so the feet stay planted and
    # the whole figure moves as one. Head is nested UNDER the body node so body sway carries it too.
    head_pivot = _nij_pivot(head_parts) if head_parts else (0.0, 0.0)
    body_pivot = _nij_pivot(drawn) if drawn else (0.0, 0.0)  # feet/ground, from the whole figure

    part_uuid: dict[str, int] = {}
    head_children: list[dict] = []
    body_children: list[dict] = []
    for part, mesh in drawn:
        uuid = uuids.next()
        part_uuid[part.id] = uuid
        if part.id in head_ids:
            head_children.append(_part_node(uuid, part, mesh, slot_of, scale, offset=head_pivot))
        else:
            body_children.append(_part_node(uuid, part, mesh, slot_of, scale, offset=body_pivot))

    def _group(uuid, name, trans, children):
        return {"uuid": uuid, "name": name, "type": "Node", "enabled": True, "zsort": 0.0,
                "transform": {"trans": [trans[0], trans[1], 0.0], "rot": [0.0, 0.0, 0.0],
                              "scale": [1.0, 1.0]},
                "lockToRoot": False, "pinToMesh": False, "children": children}

    head_group_uuid = uuids.next() if head_children else None
    body_group_uuid = uuids.next() if body_children else None

    root_children: list[dict] = []
    if body_group_uuid is not None:
        children = list(body_children)
        if head_group_uuid is not None:
            # head is a child of body -> its transform is relative to the body pivot
            hp = (head_pivot[0] - body_pivot[0], head_pivot[1] - body_pivot[1])
            children.append(_group(head_group_uuid, "head", hp, head_children))
        root_children.append(_group(body_group_uuid, "body", body_pivot, children))
    elif head_group_uuid is not None:                     # head-only (portrait, no body)
        root_children.append(_group(head_group_uuid, "head", head_pivot, head_children))

    # Allocate parameter uuids up front: physics nodes reference output-param uuids, and the driver
    # param needs a transform binding to the (about-to-be-created) anchor node.
    param_uuid: dict[str, int] = {p.id: uuids.next() for p in rig.parameters}
    anchor_nodes, extra_bindings = _build_physics(rig, uuids, param_uuid)

    root_node = {
        "uuid": uuids.next(reserve_root=True),
        "name": "Root",
        "type": "Node",
        "enabled": True,
        "zsort": 0.0,
        "transform": _identity_transform(),
        "lockToRoot": False,
        "pinToMesh": False,
        "children": root_children + anchor_nodes,
    }

    groups = []
    if head_group_uuid is not None:
        groups.append({"ids": head_ids, "uuid": head_group_uuid, "rot": _HEAD_ROT})
    if body_group_uuid is not None:
        groups.append({"ids": body_ids, "uuid": body_group_uuid, "rot": _BODY_ROT})
    params = []
    for p in rig.parameters:
        pd = _parameter(param_uuid[p.id], p, rig, part_uuid, scale, groups=groups)
        pd["bindings"].extend(extra_bindings.get(p.id, []))
        params.append(pd)

    puppet = {
        "meta": {
            "name": rig.meta.name,
            "version": "1.0-alpha",
            "thumbnailId": NO_TEXTURE,
            "preservePixels": False,
        },
        "physics": {"pixelsPerMeter": 1000.0, "gravity": 9.8},
        "nodes": root_node,
        "param": params,
        "automation": [],
        "animations": _build_animations(rig, param_uuid),
    }
    return PuppetBuild(puppet=puppet, textures=texture_blobs)


# --------------------------------------------------------------------------------------------------
# Animations (idle: blink / breath / sway) — nijilive Animation[string] dict
# --------------------------------------------------------------------------------------------------
def _build_animations(rig: Rig, param_uuid: dict[str, int]) -> dict:
    """Map IRR animations to nijilive's ``animations`` dict (keyed by name).

    Schema from ``core/animation/animation.d``: each Animation has ``timestep`` (= 1/fps), ``length``
    (frames), ``leadIn``/``leadOut`` (-1 = none), and ``lanes``; each lane targets a parameter by
    ``uuid`` on ``target`` axis 0 (our params are 1D) with ``keyframes`` {frame,value,tension}.
    ``interpolation`` and ``merge_mode`` serialize by member name (Linear/Cubic, Forced)."""
    out: dict = {}
    for anim in rig.animations:
        lanes = []
        for lane in anim.lanes:
            uuid = param_uuid.get(lane.param_id)
            if uuid is None:
                continue
            lanes.append({
                "interpolation": lane.interpolation.value,
                "uuid": uuid,
                "target": 0,  # axis 0 (X) — all authored params are scalar
                "keyframes": [
                    {"frame": kf.frame, "value": kf.value, "tension": kf.tension}
                    for kf in lane.keyframes
                ],
                # Forced: the idle owns these params outright (blink must close to 0, not add).
                "merge_mode": "Forced",
            })
        if not lanes:
            continue
        out[anim.name] = {
            "timestep": 1.0 / anim.fps,
            "additive": False,
            "animationWeight": 1.0,
            "length": anim.length,
            "leadIn": -1,
            "leadOut": -1,
            "lanes": lanes,
        }
    return out


# --------------------------------------------------------------------------------------------------
# Node / Part
# --------------------------------------------------------------------------------------------------
def _identity_transform() -> dict:
    return {"trans": [0.0, 0.0, 0.0], "rot": [0.0, 0.0, 0.0], "scale": [1.0, 1.0]}


def _part_node(uuid: int, part: Part, mesh: Mesh, slot_of: dict[str, int], scale: float,
               offset: tuple[float, float] = (0.0, 0.0)) -> dict:
    slot = slot_of[part.texture_id]
    return {
        "uuid": uuid,
        "name": part.id,
        "type": "Part",
        "enabled": True,
        # IRR uses "higher draw_order = on top"; nijilive draws *lowest* zsort last (on top) and
        # sorts children descending, so we negate to preserve intended layering.
        "zsort": float(-part.draw_order),
        "transform": _identity_transform(),
        "lockToRoot": False,
        "pinToMesh": False,
        "mesh": _mesh_data(mesh, scale, offset),
        # albedo + (empty) emission + (empty) bumpmap slots
        "textures": [slot, NO_TEXTURE, NO_TEXTURE],
        "blend_mode": "Normal",
        "tint": [1.0, 1.0, 1.0],
        "screenTint": [0.0, 0.0, 0.0],
        "emissionStrength": 1.0,
        "mask_threshold": 0.5,
        "opacity": part.opacity,
    }


def _mesh_data(mesh: Mesh, scale: float, offset: tuple[float, float] = (0.0, 0.0)) -> dict:
    ox, oy = offset  # subtract the parent-group origin so verts are local to a re-centered group node
    verts: list[float] = []
    for x, y in mesh.vertices:
        verts.extend((x * scale - ox, -y * scale - oy))  # negate y: IRR y-up -> nijilive y-down
    uvs: list[float] = []
    for u, v in mesh.uvs:
        uvs.extend((u, v))
    indices: list[int] = []
    for tri in mesh.triangles:
        indices.extend(tri)
    return {"verts": verts, "uvs": uvs, "indices": indices, "origin": [0.0, 0.0]}


# --------------------------------------------------------------------------------------------------
# Parameters / deform bindings
# --------------------------------------------------------------------------------------------------
def _parameter(uuid: int, param: Parameter, rig: Rig, part_uuid: dict[str, int], scale: float,
               groups: list | None = None) -> dict:
    keyforms = sorted(param.keyforms, key=lambda k: k.value)
    span = param.max - param.min
    if span <= 0:
        axis_x = [0.0]
        keyforms = keyforms[:1] or keyforms
    else:
        axis_x = [(kf.value - param.min) / span for kf in keyforms] or [0.0, 1.0]

    # Group-turn params (head / body) drive that GROUP node's rotation (one rigid unit). Their
    # per-vertex offsets on the group's parts are dropped (the transform replaces them); offsets on
    # OTHER parts (e.g. the neck follow-through under a head-turn) stay as normal deform bindings.
    exclude: set = set()
    rot_bindings: list[dict] = []
    for g in (groups or []):
        if param.id in g["rot"]:
            exclude |= g["ids"]
            channel, max_rad = g["rot"][param.id]
            if keyforms:
                rot_bindings.append(_rot_binding(g["uuid"], param, channel, max_rad))
    bindings = _deform_bindings(keyforms, rig, part_uuid, scale, exclude=exclude) if keyforms else []
    bindings.extend(rot_bindings)
    # Opacity fades (a part that appears/disappears across the param — e.g. a synthesised closed-eye lash
    # line crossfading in on ParamEyeOpen). Independent of the deform/transform split above, so exclude
    # does not apply. Without this the .inp shipped these parts at their static opacity — the lash line was
    # painted over the OPEN eye at rest (see core.synth.eye; the moc3 path got this via opacity keyforms).
    if keyforms:
        bindings.extend(_opacity_bindings(keyforms, rig, part_uuid))

    return {
        "uuid": uuid,
        "name": param.id,
        "is_vec2": False,
        "min": [param.min, 0.0],
        "max": [param.max, 0.0],
        "defaults": [param.default, 0.0],
        "axis_points": [axis_x, [0.0]],
        "merge_mode": "Additive",
        "bindings": bindings,
    }


def _deform_bindings(keyforms, rig: Rig, part_uuid: dict[str, int], scale: float,
                     exclude: set | None = None) -> list[dict]:
    exclude = exclude or set()
    # Which parts are deformed by any keyform of this parameter? (skip parts driven by a group transform)
    affected: list[str] = []
    for kf in keyforms:
        for pid in kf.mesh_offsets:
            if pid not in affected and pid in part_uuid and pid not in exclude:
                affected.append(pid)

    bindings: list[dict] = []
    for pid in affected:
        mesh = rig.mesh_for(pid)
        if mesh is None:
            continue
        vcount = len(mesh.vertices)
        zero_cell = [[0.0, 0.0] for _ in range(vcount)]
        # values[x][y]; y axis has a single point -> one cell per keyform.
        values: list[list[list[list[float]]]] = []
        for kf in keyforms:
            offsets = kf.mesh_offsets.get(pid)
            if offsets is None:
                cell = [list(p) for p in zero_cell]
            else:
                cell = [[dx * scale, -dy * scale] for (dx, dy) in offsets]  # y-up -> y-down
            values.append([cell])
        is_set = [[True] for _ in keyforms]
        bindings.append(
            {
                "node": part_uuid[pid],
                "param_name": "deform",
                "values": values,
                "isSet": is_set,
                "interpolate_mode": "Linear",
            }
        )
    return bindings


# --------------------------------------------------------------------------------------------------
# Physics (SimplePhysics nodes) — make hair animate itself
# --------------------------------------------------------------------------------------------------
def _build_physics(rig: Rig, uuids: "_UuidAllocator", param_uuid: dict[str, int]):
    """Emit a non-drawn anchor node per driver param (its transform.t.x bound to that param) with a
    child SimplePhysics pendulum per hair output param. When the driver moves, the anchor moves, the
    pendulum swings (with momentum), and it drives the output param — so hair animates itself.

    Returns ``(anchor_nodes, extra_bindings)`` where ``extra_bindings[param_id]`` are transform
    bindings to splice into that parameter.
    """
    anchor_nodes: list[dict] = []
    extra_bindings: dict[str, list[dict]] = {}
    if not rig.physics:
        return anchor_nodes, extra_bindings

    params_by_id = {p.id: p for p in rig.parameters}
    # Group rigs by their full (present) driver set so rigs sharing the same drivers share one anchor.
    # Multi-driver rigs (e.g. a skirt zone driven by body sway + the near leg) get an anchor bound to
    # every driver, whose transform contributions sum -> all those motions excite the pendulum.
    groups: dict[tuple, list] = {}
    for ph in rig.physics:
        if ph.output_param not in param_uuid:
            continue
        drivers = tuple(d for d in ph.all_drivers() if d in param_uuid)
        if not drivers:
            continue
        groups.setdefault(drivers, []).append(ph)

    for drivers, rigs in groups.items():
        anchor_uuid = uuids.next()
        phys_nodes = [
            _simple_physics_node(uuids.next(), param_uuid[ph.output_param], ph) for ph in rigs
        ]
        anchor_nodes.append(
            _anchor_node(anchor_uuid, "physics_anchor_" + "_".join(drivers), phys_nodes)
        )
        for d in drivers:
            extra_bindings.setdefault(d, []).append(
                _transform_binding(anchor_uuid, params_by_id[d], channel=_DRIVER_CHANNEL.get(d, "transform.t.x"))
            )
    return anchor_nodes, extra_bindings


# Which anchor transform channel each driver moves. Pitch (head/body Y) -> vertical, so a nod bobs
# hair/cloth; everything else (yaw/roll/legs/lean) -> horizontal sway. The pendulum reads the anchor's
# 2D motion, so binding different drivers to t.x vs t.y makes the response directional, not 1-axis.
_DRIVER_CHANNEL = {
    "ParamAngleY": "transform.t.y",
    "ParamBodyAngleY": "transform.t.y",
}


def _anchor_node(uuid: int, name: str, children: list[dict]) -> dict:
    return {
        "uuid": uuid,
        "name": name,
        "type": "Node",
        "enabled": True,
        "zsort": 0.0,
        "transform": _identity_transform(),
        "lockToRoot": False,
        "pinToMesh": False,
        "children": children,
    }


def _simple_physics_node(uuid: int, output_param_uuid: int, ph) -> dict:
    angle_damping = min(1.5, max(0.1, 0.4 + ph.drag))
    return {
        "uuid": uuid,
        "name": f"phys_{ph.output_param}",
        "type": "SimplePhysics",
        "enabled": True,
        "zsort": 0.0,
        "transform": _identity_transform(),
        "lockToRoot": False,
        "pinToMesh": False,
        "children": [],
        "param": output_param_uuid,
        # PascalCase enum names: nijilive serializes PhysicsModel/ParamMapMode by member name (the
        # lowercase "pendulum" values landed in nijilive 2026-06-06, AFTER nijigenerate beta2).
        "model_type": _MODEL_NAMES.get(getattr(ph.model, "value", ph.model), "Pendulum"),
        "map_mode": "AngleLength",
        "gravity": _PHYS_GRAVITY,
        "length": max(0.1, ph.length) * _PHYS_LENGTH_SCALE,
        "frequency": 1.0,
        "angle_damping": angle_damping,
        "length_damping": _PHYS_LENGTH_DAMP,
        "output_scale": [_PHYS_OUTPUT_SCALE, 1.0],
        "local_only": False,
    }


# --------------------------------------------------------------------------------------------------
# Head-group hierarchy — parent the head parts under one node and drive its rotation, so the whole
# head (face + hair + eyes + ...) moves as ONE rigid unit instead of each part slipping on its own.
# --------------------------------------------------------------------------------------------------
# _HEAD_ROLES is imported from core.structure.graph at the top of the module (single source of truth).

# Head-turn params -> which node-rotation channel drives the head group, and the head rotation (rad)
# at the param's extreme. inochi2d applies rot.y/rot.x as a pseudo-3D horizontal/vertical squash
# (yaw/pitch), rot.z as in-plane roll — all about the group's origin (the neck pivot).
_HEAD_ROT = {
    "ParamAngleX": ("transform.r.y", 0.52),   # yaw  -> horizontal squash
    "ParamAngleY": ("transform.r.x", 0.42),   # pitch-> vertical squash
    "ParamAngleZ": ("transform.r.z", 0.35),   # roll -> in-plane tilt
}
# Body sway rotates the whole figure about the feet (gentler than the head). ParamBodyAngle range is
# +-10, so the whole standing figure leans subtly as one unit, feet planted.
_BODY_ROT = {
    "ParamBodyAngleX": ("transform.r.y", 0.10),
    "ParamBodyAngleY": ("transform.r.x", 0.08),
    "ParamBodyAngleZ": ("transform.r.z", 0.09),
}


def head_group_ids(drawn) -> set:
    """Part ids that belong to the head group: head-role parts + head-region accessories (hairpins,
    earrings, headwear). Shared by the emitter and the live preview so both group identically.
    ``drawn`` is a list of (part, mesh)."""
    head = [(p, m) for p, m in drawn if p.semantic_role in _HEAD_ROLES]
    ids = {p.id for p, _ in head}
    if head:
        hx = [x for _, m in head for x, _ in m.vertices]
        hy = [y for _, m in head for _, y in m.vertices]
        hx0, hx1, hy0, hy1 = min(hx), max(hx), min(hy), max(hy)
        hw, hh = max(hx1 - hx0, 1e-6), max(hy1 - hy0, 1e-6)
        for p, m in drawn:
            if p.semantic_role is SemanticRole.accessory and p.id not in ids:
                cx = sum(x for x, _ in m.vertices) / len(m.vertices)
                cy = sum(y for _, y in m.vertices) / len(m.vertices)
                if hx0 - 0.3 * hw <= cx <= hx1 + 0.3 * hw and hy0 - 0.2 * hh <= cy <= hy1 + 0.3 * hh:
                    ids.add(p.id)
    return ids


def _opacity_bindings(keyforms, rig: Rig, part_uuid: dict[str, int]) -> list[dict]:
    """One ``opacity`` value-binding per part whose opacity this parameter keys (``opacity_overrides``).
    A part with no override at a given keyform holds its base opacity there, so a fade that only touches
    the extremes still interpolates from the resting value. Scalar ``values[x][y]`` like the transform
    bindings; ordered by ascending keyform value to match the param's ``axis_points``."""
    base = {p.id: p.opacity for p in rig.parts}
    affected: list[str] = []
    for kf in keyforms:
        for pid in kf.opacity_overrides:
            if pid not in affected and pid in part_uuid:
                affected.append(pid)
    bindings: list[dict] = []
    for pid in affected:
        values = [[kf.opacity_overrides.get(pid, base.get(pid, 1.0))] for kf in keyforms]
        bindings.append({
            "node": part_uuid[pid], "param_name": "opacity",
            "values": values, "isSet": [[True] for _ in keyforms], "interpolate_mode": "Linear",
        })
    return bindings


def _rot_binding(node_uuid: int, param, channel: str, max_rad: float) -> dict:
    """Bind a head-turn param to the head-group node's rotation ``channel`` (radians at the extreme).
    Uniform for the whole group -> every head part shares the exact transform -> no inter-part slip."""
    keyforms = sorted(param.keyforms, key=lambda k: k.value)
    denom = max(abs(param.min), abs(param.max), 1e-6)
    values = [[(kf.value / denom) * max_rad] for kf in keyforms]
    return {
        "node": node_uuid, "param_name": channel,
        "values": values, "isSet": [[True] for _ in keyforms], "interpolate_mode": "Linear",
    }


def _transform_binding(node_uuid: int, param, *, channel: str = "transform.t.x") -> dict:
    """Bind a parameter to an anchor node's transform ``channel`` (e.g. ``transform.t.x`` or
    ``transform.t.y``). Scalar values[x][y] (one per axis point), matching nijilive's binding grid;
    ordered by ascending keyform value like the param's axis_points."""
    keyforms = sorted(param.keyforms, key=lambda k: k.value)
    denom = max(abs(param.min), abs(param.max), 1e-6)
    values = [[(kf.value / denom) * _ANCHOR_SHIFT] for kf in keyforms]
    is_set = [[True] for _ in keyforms]
    return {
        "node": node_uuid,
        "param_name": channel,
        "values": values,
        "isSet": is_set,
        "interpolate_mode": "Linear",
    }


# --------------------------------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------------------------------
class _UuidAllocator:
    """Deterministic uuid allocation (stable output for the same rig). Root is reserved as 1."""

    def __init__(self) -> None:
        self._next = 2

    def next(self, *, reserve_root: bool = False) -> int:
        if reserve_root:
            return 1
        u = self._next
        self._next += 1
        return u


def _load_texture(texture, root: Path | None) -> bytes:
    if root is not None:
        path = root / texture.path
        if path.is_file():
            return path.read_bytes()
    # No asset on disk -> synthesize a valid placeholder PNG at the declared size so the .inp is
    # self-contained and renderable. (Phase 0 uses this for the example rig.)
    return solid_png(texture.width, texture.height)


def solid_png(width: int, height: int, rgba: tuple[int, int, int, int] = (180, 180, 200, 255)) -> bytes:
    """Encode a solid-color RGBA PNG using only the stdlib (zlib)."""
    r, g, b, a = rgba
    row = bytes([0]) + bytes([r, g, b, a]) * width  # filter byte 0 + pixels
    raw = row * height
    compressed = zlib.compress(raw, 9)

    def chunk(tag: bytes, data: bytes) -> bytes:
        return (
            struct.pack(">I", len(data))
            + tag
            + data
            + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)
        )

    sig = b"\x89PNG\r\n\x1a\n"
    ihdr = struct.pack(">IIBBBBB", width, height, 8, 6, 0, 0, 0)  # 8-bit RGBA
    return sig + chunk(b"IHDR", ihdr) + chunk(b"IDAT", compressed) + chunk(b"IEND", b"")
