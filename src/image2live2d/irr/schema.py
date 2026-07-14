"""Intermediate Rig Representation (IRR).

The IRR is the keystone of image2live2d: a format-neutral, JSON-serializable description of a
complete 2D puppet rig. Everything *before* the IRR in the pipeline is shared; everything *after*
it is a thin, backend-specific emitter (nijilive ``.inp`` for Route B, Live2D ``.moc3`` for
Route A). Getting this schema right is what makes "Route A is mostly free after Route B" true.

Design conventions
------------------
* **Standard Live2D parameter IDs** (see ``params.py``) are used verbatim so the same motion
  clips, ARKit face-tracking, and TTS lip-sync drive *both* backends unchanged.
* A rig is a flat list of typed objects referenced by string ``id``; relationships are expressed
  by id references (``parent``, ``texture_id``, ``part_id``) and validated for integrity.
* Geometry uses normalized model-space coordinates; emitters map to each backend's conventions.
"""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field, model_validator

IRR_VERSION = "0.1.0"

# A 2D point in normalized model space. Serializes to/from a JSON array [x, y].
Vec2 = tuple[float, float]
Tri = tuple[int, int, int]


class SemanticRole(str, Enum):
    """What a part *is*, independent of how it's drawn. Drives template fitting, physics
    assignment, and parameter binding."""

    # Head / face
    face_base = "face_base"
    hair_front = "hair_front"
    hair_side = "hair_side"
    hair_back = "hair_back"
    eyebrow_l = "eyebrow_l"
    eyebrow_r = "eyebrow_r"
    eye_l = "eye_l"
    eye_r = "eye_r"
    eye_white_l = "eye_white_l"
    eye_white_r = "eye_white_r"
    pupil_l = "pupil_l"
    pupil_r = "pupil_r"
    nose = "nose"
    mouth = "mouth"
    mouth_cavity = "mouth_cavity"  # synthesised interior behind the lips — see core.synth.mouth
    ear_l = "ear_l"
    ear_r = "ear_r"
    blush = "blush"
    # Body
    neck = "neck"
    torso = "torso"
    arm_l = "arm_l"
    arm_r = "arm_r"
    hand_l = "hand_l"
    hand_r = "hand_r"
    leg_l = "leg_l"
    leg_r = "leg_r"
    # Extras
    clothing = "clothing"
    accessory = "accessory"
    background = "background"
    other = "other"


class DeformerType(str, Enum):
    """Deformer kinds shared by Live2D and nijilive."""

    warp = "warp"  # grid (bezier/lattice) deformation
    rotation = "rotation"  # rigid rotation/scale about a pivot


class Texture(BaseModel):
    """A separated, inpainted layer image produced by decomposition."""

    id: str
    path: str  # relative path to the PNG (RGBA)
    width: int = Field(gt=0)
    height: int = Field(gt=0)


class Mesh(BaseModel):
    """Triangulated geometry for a single part, in model space.

    ``uvs`` are in [0, 1] referencing the part's texture. ``triangles`` index into ``vertices``.
    """

    part_id: str
    vertices: list[Vec2]
    uvs: list[Vec2]
    triangles: list[Tri]

    @model_validator(mode="after")
    def _check_geometry(self) -> "Mesh":
        n = len(self.vertices)
        if n < 3:
            raise ValueError(f"mesh for part {self.part_id!r} needs >= 3 vertices, got {n}")
        if len(self.uvs) != n:
            raise ValueError(
                f"mesh for part {self.part_id!r}: uvs ({len(self.uvs)}) must match vertices ({n})"
            )
        for tri in self.triangles:
            for idx in tri:
                if not 0 <= idx < n:
                    raise ValueError(
                        f"mesh for part {self.part_id!r}: triangle index {idx} out of range [0,{n})"
                    )
        return self


class Part(BaseModel):
    """A drawable layer. Bound to a texture, placed in draw order, optionally parented to a
    deformer so it inherits that deformer's motion."""

    id: str
    semantic_role: SemanticRole
    texture_id: str
    draw_order: int  # higher = drawn on top
    parent_deformer: str | None = None
    opacity: float = Field(default=1.0, ge=0.0, le=1.0)


class Deformer(BaseModel):
    """A node that transforms its children (parts and/or other deformers).

    For ``warp``: a ``grid_rows`` x ``grid_cols`` lattice of control points (``grid_vertices``).
    For ``rotation``: a ``pivot`` about which children rotate/scale.
    """

    id: str
    type: DeformerType
    parent: str | None = None  # another deformer id, or None (root)
    # warp-only
    grid_rows: int | None = None
    grid_cols: int | None = None
    grid_vertices: list[Vec2] | None = None
    # rotation-only
    pivot: Vec2 | None = None

    @model_validator(mode="after")
    def _check_kind(self) -> "Deformer":
        if self.type is DeformerType.warp:
            if not (self.grid_rows and self.grid_cols and self.grid_vertices):
                raise ValueError(f"warp deformer {self.id!r} requires grid_rows/cols/vertices")
            expected = self.grid_rows * self.grid_cols
            if len(self.grid_vertices) != expected:
                raise ValueError(
                    f"warp deformer {self.id!r}: expected {expected} grid vertices, "
                    f"got {len(self.grid_vertices)}"
                )
        elif self.type is DeformerType.rotation and self.pivot is None:
            raise ValueError(f"rotation deformer {self.id!r} requires a pivot")
        return self


class Keyform(BaseModel):
    """The deformed state of the rig at one parameter ``value``.

    A parameter interpolates between its keyforms. Offsets are *deltas* from the rest pose.
    * ``mesh_offsets[part_id]`` -> per-vertex Vec2 delta (len must equal that part's vertex count).
    * ``deformer_offsets[deformer_id]`` -> per-grid-vertex Vec2 delta (warp) or [pivot delta]
      (rotation, len 1).
    * ``opacity_overrides[part_id]`` -> absolute opacity at this keyform.
    """

    value: float
    mesh_offsets: dict[str, list[Vec2]] = Field(default_factory=dict)
    deformer_offsets: dict[str, list[Vec2]] = Field(default_factory=dict)
    opacity_overrides: dict[str, float] = Field(default_factory=dict)


class Parameter(BaseModel):
    """A driveable parameter (e.g. ``ParamMouthOpenY``) with keyforms describing how the rig
    deforms across its range. Use standard Live2D ids from ``params.py``."""

    id: str
    min: float
    max: float
    default: float = 0.0
    keyforms: list[Keyform] = Field(default_factory=list)

    @model_validator(mode="after")
    def _check_range(self) -> "Parameter":
        if self.min > self.max:
            raise ValueError(f"parameter {self.id!r}: min ({self.min}) > max ({self.max})")
        if not self.min <= self.default <= self.max:
            raise ValueError(f"parameter {self.id!r}: default {self.default} outside [{self.min},{self.max}]")
        for kf in self.keyforms:
            if not self.min <= kf.value <= self.max:
                raise ValueError(
                    f"parameter {self.id!r}: keyform value {kf.value} outside [{self.min},{self.max}]"
                )
        seen = sorted(kf.value for kf in self.keyforms)
        if len(set(seen)) != len(seen):
            raise ValueError(f"parameter {self.id!r}: duplicate keyform values")
        return self


class PhysicsModel(str, Enum):
    """Pendulum models shared with nijilive's SimplePhysics (member names match its enum)."""

    pendulum = "pendulum"               # rigid pendulum
    spring_pendulum = "spring_pendulum"  # springy pendulum — bouncier, more cloth-like


class PhysicsRig(BaseModel):
    """A pendulum that converts driver parameters (e.g. body sway + leg motion) into an output
    parameter (e.g. ``ParamSkirtL``) via a spring/damper sim. Procedurally generated from hair/cloth
    parts.

    ``driver_param`` is the primary driver; ``extra_drivers`` are additional parameters whose motion
    *also* excites this pendulum (their contributions sum) — e.g. a skirt zone driven by both body
    sway and the near leg, so all lower-body motion affects the cloth."""

    id: str
    driver_param: str
    output_param: str
    extra_drivers: list[str] = Field(default_factory=list)
    model: PhysicsModel = PhysicsModel.pendulum
    mass: float = 1.0
    drag: float = 0.2
    length: float = 1.0
    gravity: Vec2 = (0.0, -1.0)
    wind: Vec2 = (0.0, 0.0)

    def all_drivers(self) -> list[str]:
        """Primary + extra drivers, de-duplicated, primary first."""
        seen, out = set(), []
        for d in (self.driver_param, *self.extra_drivers):
            if d not in seen:
                seen.add(d)
                out.append(d)
        return out


class InterpolateMode(str, Enum):
    """Keyframe interpolation modes shared by Live2D and nijilive (member names match nijilive's
    ``InterpolateMode`` enum, which serializes by name)."""

    nearest = "Nearest"
    linear = "Linear"
    stepped = "Stepped"
    cubic = "Cubic"
    bezier = "Bezier"


class AnimKeyframe(BaseModel):
    """One keyframe on an animation lane: a parameter ``value`` at integer ``frame``."""

    frame: int = Field(ge=0)
    value: float
    tension: float = 0.5


class AnimationLane(BaseModel):
    """A timeline of keyframes driving one parameter over an animation."""

    param_id: str
    keyframes: list[AnimKeyframe]
    interpolation: InterpolateMode = InterpolateMode.linear

    @model_validator(mode="after")
    def _check(self) -> "AnimationLane":
        if not self.keyframes:
            raise ValueError(f"animation lane for {self.param_id!r} has no keyframes")
        return self


class Animation(BaseModel):
    """A named, optionally-looping animation (e.g. an idle: blink + breath + sway). Frames are at
    ``fps``; ``length`` is the total frame count (loop point)."""

    name: str
    fps: float = Field(default=60.0, gt=0)
    length: int = Field(gt=0)
    loop: bool = True
    lanes: list[AnimationLane] = Field(default_factory=list)


class Meta(BaseModel):
    name: str
    source_image: str | None = None
    archetype: str | None = None  # which template was used (e.g. "portrait_front")
    irr_version: str = IRR_VERSION


class Rig(BaseModel):
    """The complete Intermediate Rig Representation. Root object emitters consume."""

    meta: Meta
    textures: list[Texture] = Field(default_factory=list)
    parts: list[Part] = Field(default_factory=list)
    meshes: list[Mesh] = Field(default_factory=list)
    deformers: list[Deformer] = Field(default_factory=list)
    parameters: list[Parameter] = Field(default_factory=list)
    physics: list[PhysicsRig] = Field(default_factory=list)
    animations: list[Animation] = Field(default_factory=list)

    # ---- convenience lookups ---------------------------------------------------------------
    def texture_ids(self) -> set[str]:
        return {t.id for t in self.textures}

    def part_ids(self) -> set[str]:
        return {p.id for p in self.parts}

    def deformer_ids(self) -> set[str]:
        return {d.id for d in self.deformers}

    def parameter_ids(self) -> set[str]:
        return {p.id for p in self.parameters}

    def mesh_for(self, part_id: str) -> Mesh | None:
        return next((m for m in self.meshes if m.part_id == part_id), None)

    def parts_in_draw_order(self) -> list[Part]:
        return sorted(self.parts, key=lambda p: p.draw_order)

    # ---- structural integrity (hard errors) -----------------------------------------------
    @model_validator(mode="after")
    def _integrity(self) -> "Rig":
        _require_unique([t.id for t in self.textures], "texture")
        _require_unique([p.id for p in self.parts], "part")
        _require_unique([d.id for d in self.deformers], "deformer")
        _require_unique([p.id for p in self.parameters], "parameter")

        tex_ids, part_ids, def_ids, param_ids = (
            self.texture_ids(),
            self.part_ids(),
            self.deformer_ids(),
            self.parameter_ids(),
        )

        for p in self.parts:
            if p.texture_id not in tex_ids:
                raise ValueError(f"part {p.id!r} references missing texture {p.texture_id!r}")
            if p.parent_deformer is not None and p.parent_deformer not in def_ids:
                raise ValueError(f"part {p.id!r} references missing deformer {p.parent_deformer!r}")

        vertex_counts: dict[str, int] = {}
        for m in self.meshes:
            if m.part_id not in part_ids:
                raise ValueError(f"mesh references missing part {m.part_id!r}")
            vertex_counts[m.part_id] = len(m.vertices)

        for d in self.deformers:
            if d.parent is not None and d.parent not in def_ids:
                raise ValueError(f"deformer {d.id!r} references missing parent {d.parent!r}")
        _check_no_deformer_cycles(self.deformers)

        for param in self.parameters:
            for kf in param.keyforms:
                for pid, offsets in kf.mesh_offsets.items():
                    if pid not in part_ids:
                        raise ValueError(f"{param.id!r} keyform offsets unknown part {pid!r}")
                    if pid in vertex_counts and len(offsets) != vertex_counts[pid]:
                        raise ValueError(
                            f"{param.id!r} keyform mesh_offsets[{pid!r}] has {len(offsets)} "
                            f"deltas but mesh has {vertex_counts[pid]} vertices"
                        )
                for did in kf.deformer_offsets:
                    if did not in def_ids:
                        raise ValueError(f"{param.id!r} keyform offsets unknown deformer {did!r}")
                for pid in kf.opacity_overrides:
                    if pid not in part_ids:
                        raise ValueError(f"{param.id!r} opacity override unknown part {pid!r}")

        for phys in self.physics:
            for ref in (*phys.all_drivers(), phys.output_param):
                if ref not in param_ids:
                    raise ValueError(f"physics {phys.id!r} references missing parameter {ref!r}")

        for anim in self.animations:
            for lane in anim.lanes:
                if lane.param_id not in param_ids:
                    raise ValueError(
                        f"animation {anim.name!r} lane references missing parameter {lane.param_id!r}"
                    )
                for kf in lane.keyframes:
                    if kf.frame > anim.length:
                        raise ValueError(
                            f"animation {anim.name!r}: keyframe frame {kf.frame} exceeds length "
                            f"{anim.length}"
                        )
        return self


def _require_unique(ids: list[str], kind: str) -> None:
    seen: set[str] = set()
    for i in ids:
        if i in seen:
            raise ValueError(f"duplicate {kind} id {i!r}")
        seen.add(i)


def _check_no_deformer_cycles(deformers: list[Deformer]) -> None:
    parent = {d.id: d.parent for d in deformers}
    for start in parent:
        seen: set[str] = set()
        cur: str | None = start
        while cur is not None:
            if cur in seen:
                raise ValueError(f"deformer parent cycle involving {start!r}")
            seen.add(cur)
            cur = parent.get(cur)
