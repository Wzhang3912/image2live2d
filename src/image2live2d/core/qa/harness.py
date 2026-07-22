"""Stage 7' — QA harness. "Does it look good in motion?"

Automated checks sweep every parameter min->max, (eventually) render frames via a backend runtime,
and flag artifacts. For now it runs the IRR lint and parameter-coverage checks; rendering hooks in
once a backend emitter + runtime adapter exist.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from ..motion import MIN_SWING_FRAC
from ...irr.schema import Parameter, Rig, SemanticRole, Vec2
from ...irr.validate import Issue, Severity, lint

# A deformed vertex shifting more than this (model-space units, canvas ~= 1.0 wide) almost
# certainly signals a runaway keyform rather than intended motion.
MAX_DISPLACEMENT = 0.6


@dataclass
class ParamSweep:
    """A planned sweep of one parameter for artifact detection."""

    param_id: str
    samples: list[float]


def plan_sweeps(rig: Rig, *, steps: int = 9) -> list[ParamSweep]:
    """Plan min->max sweeps for every parameter with keyforms."""
    sweeps: list[ParamSweep] = []
    for p in rig.parameters:
        if not p.keyforms:
            continue
        sweeps.append(ParamSweep(p.id, _linspace(p, steps)))
    return sweeps


def _linspace(p: Parameter, steps: int) -> list[float]:
    if steps < 2:
        return [p.default]
    span = p.max - p.min
    return [p.min + span * i / (steps - 1) for i in range(steps)]


def _offsets_at(param: Parameter, value: float, part_id: str, vcount: int) -> list[Vec2]:
    """Interpolate a part's per-vertex offsets at ``value`` from a parameter's keyforms.

    Values outside the keyform range clamp to the nearest keyform. A part absent from a keyform is
    treated as zero offset there.
    """
    kfs = sorted(param.keyforms, key=lambda k: k.value)
    zero = [(0.0, 0.0)] * vcount

    def offs(kf) -> list[Vec2]:
        return kf.mesh_offsets.get(part_id, zero)

    if not kfs:
        return zero
    if value <= kfs[0].value:
        return offs(kfs[0])
    if value >= kfs[-1].value:
        return offs(kfs[-1])
    for lo, hi in zip(kfs, kfs[1:]):
        if lo.value <= value <= hi.value:
            span = hi.value - lo.value
            t = 0.0 if span == 0 else (value - lo.value) / span
            a, b = offs(lo), offs(hi)
            return [(ax + (bx - ax) * t, ay + (by - ay) * t) for (ax, ay), (bx, by) in zip(a, b)]
    return zero


def deform_at(rig: Rig, param_id: str, value: float) -> dict[str, list[Vec2]]:
    """Absolute deformed vertex positions (rest + interpolated offset) for each part this parameter
    moves at ``value``."""
    param = next((p for p in rig.parameters if p.id == param_id), None)
    if param is None:
        raise KeyError(f"no parameter {param_id!r}")
    moved = {pid for kf in param.keyforms for pid in kf.mesh_offsets}
    result: dict[str, list[Vec2]] = {}
    for pid in moved:
        mesh = rig.mesh_for(pid)
        if mesh is None:
            continue
        offs = _offsets_at(param, value, pid, len(mesh.vertices))
        result[pid] = [(x + dx, y + dy) for (x, y), (dx, dy) in zip(mesh.vertices, offs)]
    return result


@dataclass
class SweepReport:
    """Result of the numeric (render-free) param sweep."""

    frames: int = 0
    issues: list[Issue] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return not any(i.severity is Severity.warning for i in self.issues)


def sweep_report(rig: Rig, *, steps: int = 9) -> SweepReport:
    """Sweep every parameter min->max and flag numeric artifacts (NaN/inf deforms, runaway
    displacement). This is the automated half of the Phase 1 quality gate; visual believability
    (does it *look* like a blink / head-turn?) still needs the nijilive runtime and a human eye.
    """
    report = SweepReport()
    for sweep in plan_sweeps(rig, steps=steps):
        param = next(p for p in rig.parameters if p.id == sweep.param_id)
        for value in sweep.samples:
            report.frames += 1
            for pid, positions in deform_at(rig, param.id, value).items():
                rest = rig.mesh_for(pid)
                for (x, y), (rx, ry) in zip(positions, rest.vertices if rest else positions):
                    if not (math.isfinite(x) and math.isfinite(y)):
                        report.issues.append(
                            Issue(Severity.warning, "nan_deform",
                                  f"{param.id}={value:g}: non-finite vertex in {pid!r}")
                        )
                        break
                    if math.hypot(x - rx, y - ry) > MAX_DISPLACEMENT:
                        report.issues.append(
                            Issue(Severity.warning, "runaway_deform",
                                  f"{param.id}={value:g}: vertex in {pid!r} moved > "
                                  f"{MAX_DISPLACEMENT} model units")
                        )
                        break
    return report


def motion_issues(rig: Rig) -> list[Issue]:
    """Does the rig's own motion actually exercise the rig?

    A well-formed rig whose motion never moves it is exactly as good as no rig, and until now nothing
    caught that: the shipped idle drove 6 of 31 parameters and left five of eight physics chains with a
    flat-lined driver, so hair physics we had carefully tuned never swung once. These three checks are
    the ones that would have caught it. Measured over the *natural* motion only — the diagnostic
    ``sweep`` clip drives everything by construction and would paper over the very gap we're looking
    for.
    """
    from ..motion import SWEEP_NAME, motion_coverage

    # A rig with no clips at all is not a coverage failure — it is a different kind of model. A Live2D
    # puppet for VTube Studio often ships with no motion3 at all, because face tracking drives
    # ParamAngleX from a webcam. This check is about motion that *exists* and doesn't reach the rig.
    if not rig.animations:
        return []

    natural = [a for a in rig.animations if a.name != SWEEP_NAME]
    cov = motion_coverage(rig.parameters, rig.physics, natural)
    issues = [
        Issue(Severity.warning, "undriven_param",
              f"no animation moves {pid!r} — it will never be seen unless a human drags the slider")
        for pid in cov.undriven
    ]
    issues += [
        Issue(Severity.warning, "unexcited_physics",
              f"physics on {pid!r} can never swing: no driver moves more than "
              f"{MIN_SWING_FRAC:.0%} of its range in any clip")
        for pid in cov.unexcited
    ]
    issues += [
        Issue(Severity.warning, "keyed_physics_output",
              f"an animation keys {pid!r}, which the pendulum writes — the lane fights physics, "
              f"and physics wins")
        for pid in cov.keyed_outputs
    ]
    return issues


# A front-facing character's face has features on BOTH sides. If every face feature sits on one side and
# the other side is entirely empty, it is a profile view or a mis-decomposition — either way this rig
# can't drive both sides (blink one eye, turn a half-face), so warn. Judged by SIDE, not pair-by-pair:
# a character may carry a single combined eyelash (one eye_l) yet split its eye-whites L/R, so it is
# bilateral even though the eye_l/eye_r pair alone looks one-sided. Only the case where a whole side is
# empty is a red flag. Needs this many side-features present before judging, so a minimal fixture with
# one feature isn't condemned; the one real scene input had 5 on one side and 0 on the other.
_MIN_FACE_SIDE_FEATURES = 3
# NOTE: a "cluttered_input" fill-ratio signal (Σ part-bbox area / union-bbox area) was tried and
# REMOVED. It flagged the one scene input at 4.2×, but running seven diverse characters through it
# showed elaborate legitimate ones measure *higher* — a floor-length gown at 4.4×, huge drill-curls at
# 7.3× — so no threshold separates a scene from an ornate character; the signal only false-flagged busy
# outfits. The scene is still caught by `one_sided_face` (it was 5-features-on-one-side). Don't
# reintroduce fill-ratio without a signal that actually discriminates.

# L/R face-feature roles that a front-facing character carries on both sides.
_FACE_PAIRS = (
    (SemanticRole.eye_l, SemanticRole.eye_r),
    (SemanticRole.eye_white_l, SemanticRole.eye_white_r),
    (SemanticRole.eyebrow_l, SemanticRole.eyebrow_r),
    (SemanticRole.pupil_l, SemanticRole.pupil_r),
    (SemanticRole.ear_l, SemanticRole.ear_r),
)


# Mouth-region envelope, measured on the 8 real decomposed characters (model space, face_base bbox as
# the reference frame). The spread is remarkably tight — width 0.133-0.237 of the face, centre 0.808-
# 0.881 of the way down it, and horizontally centred to within 0.9% of face width on every one — so
# these bounds are set several times wider than the observed spread. They are here to catch a mouth
# region that is *grossly* wrong (on the forehead, spanning the whole face, off to one side), not to
# police style; a stylised character must never trip them.
#
# This exists because we could not otherwise tell. `synth.mouth` sizes the whole cavity from the mouth
# layer's WIDTH, so a bad mouth region silently produces a bad cavity, and nothing downstream complains
# — the same shape of silent-but-wrong that hid the head-turn squash and the clamped skirt. It is also
# the trigger for backlog T9 (SAM-mouth-from-source): the rival's premise is that See-through's mouth
# region is unreliable, and after the T8 denoise we could not reproduce that on any character. If a real
# character trips this check, that premise is back and the heavy SAM path is justified; until one does,
# it is not.
_MOUTH_MIN_WIDTH_FRAC = 0.05   # narrower than this and it is a speck, not a mouth (measured >= 0.133)
_MOUTH_MAX_WIDTH_FRAC = 0.50   # wider and the region has swallowed the jaw     (measured <= 0.237)
_MOUTH_MIN_DEPTH_FRAC = 0.50   # above mid-face is the nose/eyes, not a mouth   (measured >= 0.808)
_MOUTH_MAX_DEPTH_FRAC = 1.10   # below this it has fallen off the chin          (measured <= 0.881)
_MOUTH_MAX_OFFSET_FRAC = 0.20  # off-centre by more than this is a mis-region   (measured <= 0.009)


def _role_bbox(rig: Rig, role: SemanticRole) -> tuple[float, float, float, float] | None:
    """Model-space ``(x0, y0, x1, y1)`` over every mesh of every part carrying ``role`` (y up)."""
    ids = {p.id for p in rig.parts if p.semantic_role is role}
    verts = [v for m in rig.meshes if m.part_id in ids for v in m.vertices]
    if not verts:
        return None
    xs = [x for x, _ in verts]
    ys = [y for _, y in verts]
    return min(xs), min(ys), max(xs), max(ys)


def mouth_region_issues(rig: Rig) -> list[Issue]:
    """Flag a mouth whose region is implausible relative to the face (see the envelope above).

    Silent when either the mouth or the face is absent — ``no_face`` already covers a faceless rig, and
    a character with no mouth layer has no mouth region to judge.
    """
    mouth = _role_bbox(rig, SemanticRole.mouth)
    face = _role_bbox(rig, SemanticRole.face_base)
    if mouth is None or face is None:
        return []
    fw = face[2] - face[0]
    fh = face[3] - face[1]
    if fw <= 0.0 or fh <= 0.0:
        return []

    width = (mouth[2] - mouth[0]) / fw
    depth = (face[3] - (mouth[1] + mouth[3]) / 2.0) / fh      # y up -> distance below the face's top
    offset = ((mouth[0] + mouth[2]) / 2.0 - (face[0] + face[2]) / 2.0) / fw

    issues: list[Issue] = []
    if not _MOUTH_MIN_WIDTH_FRAC <= width <= _MOUTH_MAX_WIDTH_FRAC:
        issues.append(Issue(Severity.warning, "implausible_mouth_width",
                            f"the mouth spans {width:.2f} of the face's width (expected "
                            f"{_MOUTH_MIN_WIDTH_FRAC}-{_MOUTH_MAX_WIDTH_FRAC}) — the decomposed mouth "
                            "region looks wrong, and the synthesised cavity is sized from it"))
    if not _MOUTH_MIN_DEPTH_FRAC <= depth <= _MOUTH_MAX_DEPTH_FRAC:
        issues.append(Issue(Severity.warning, "misplaced_mouth",
                            f"the mouth sits {depth:.2f} of the way down the face (expected "
                            f"{_MOUTH_MIN_DEPTH_FRAC}-{_MOUTH_MAX_DEPTH_FRAC}) — that is not where a "
                            "mouth goes, so the mouth rig is driving the wrong region"))
    if abs(offset) > _MOUTH_MAX_OFFSET_FRAC:
        issues.append(Issue(Severity.warning, "off_centre_mouth",
                            f"the mouth is {offset:+.2f} of the face's width off its centre line "
                            f"(expected within {_MOUTH_MAX_OFFSET_FRAC}) — a mis-decomposed region, or "
                            "a profile view the front-facing mouth rig can't drive"))
    return issues


def plausibility_issues(rig: Rig) -> list[Issue]:
    """Is the input a single front-facing character this pipeline can actually rig?

    The rig math assumes one bilateral, front-facing figure — split limbs L/R, blink both eyes, turn a
    head that has two sides. Fed something else (a profile crop, or a whole *scene* — a decomposition of
    a girl-at-a-bar-table came through as 17 parts with a one-sided face and furniture fragments), the
    pipeline does not crash: it emits a rig whose motion is nonsense (a head-turn that barely moves, a
    blink with one eye). These checks flag that up front instead of shipping a confident-looking dud.
    They are warnings, not a hard stop — a rig is still produced; the caller decides what to do with it.
    """
    roles = [p.semantic_role for p in rig.parts]
    role_set = set(roles)
    issues: list[Issue] = []

    if SemanticRole.face_base not in role_set:
        issues.append(Issue(Severity.warning, "no_face",
                            "no face detected among the parts — this may not be a riggable character "
                            "(the head turn, blink and gaze rig have nothing to attach to)"))

    left = [lr[0] for lr in _FACE_PAIRS if lr[0] in role_set]
    right = [lr[1] for lr in _FACE_PAIRS if lr[1] in role_set]
    if len(left) + len(right) >= _MIN_FACE_SIDE_FEATURES and (not left or not right):
        have, empty = ("left", "right") if left else ("right", "left")
        issues.append(Issue(Severity.warning, "one_sided_face",
                            f"every face feature is on the {have} ({len(left) + len(right)} features, "
                            f"none on the {empty}) — a profile view or a mis-decomposition, which the "
                            "front-facing rig can't drive on both sides"))

    issues.extend(mouth_region_issues(rig))
    return issues


def check(rig: Rig) -> list[Issue]:
    """Full static QA pass: structural lint + numeric param sweep + motion coverage + input
    plausibility. Render-based artifact detection (tearing/holes via the nijilive runtime) is added
    in Phase 2.
    """
    return [*lint(rig), *sweep_report(rig).issues, *motion_issues(rig), *plausibility_issues(rig)]


# --------------------------------------------------------------------------------------------------
# Pass-rate harness (Phase 2 exit gate)
# --------------------------------------------------------------------------------------------------
@dataclass
class RigReport:
    """Per-rig QA result: structural lint warnings + the numeric sweep."""

    name: str
    parts: int
    params: int
    physics: int
    lint_warnings: list[Issue]
    sweep: SweepReport
    landmark_warnings: list[str] = field(default_factory=list)
    motion_warnings: list[Issue] = field(default_factory=list)
    plausibility_warnings: list[Issue] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return (self.sweep.passed and not self.lint_warnings and not self.landmark_warnings
                and not self.motion_warnings and not self.plausibility_warnings)

    @property
    def reasons(self) -> list[str]:
        out = [f"lint:{i.code}" for i in self.lint_warnings]
        out += [f"sweep:{i.code}" for i in self.sweep.issues if i.severity is Severity.warning]
        out += [f"landmark:{c}" for c in self.landmark_warnings]
        out += [f"motion:{i.code}" for i in self.motion_warnings]
        out += [f"input:{i.code}" for i in self.plausibility_warnings]
        return out


@dataclass
class BatchReport:
    """Aggregate QA over a set of rigs."""

    items: list[RigReport] = field(default_factory=list)

    @property
    def total(self) -> int:
        return len(self.items)

    @property
    def passed(self) -> int:
        return sum(1 for r in self.items if r.passed)

    @property
    def pass_rate(self) -> float:
        return self.passed / self.total if self.items else 0.0

    def format(self) -> str:
        lines = [f"{'RESULT':6}  {'name':20}  parts params phys  notes"]
        for r in self.items:
            status = "PASS" if r.passed else "FAIL"
            notes = "ok" if r.passed else ", ".join(r.reasons) or "fail"
            lines.append(
                f"{status:6}  {r.name[:20]:20}  {r.parts:5} {r.params:6} {r.physics:4}  {notes}"
            )
        pct = self.pass_rate * 100.0
        lines.append(f"\npass-rate: {self.passed}/{self.total} ({pct:.0f}%)")
        return "\n".join(lines)


def evaluate(
    rig: Rig, name: str = "rig", *, steps: int = 9, landmark_warnings: list[str] | None = None
) -> RigReport:
    """Run the static QA pass on one rig and return a pass/fail report.

    ``landmark_warnings`` (optional, from ``core.landmark.landmark_warnings``) folds per-character
    landmark sanity checks into the gate — the caller computes them since they need the layer images.
    """
    warnings = [i for i in lint(rig) if i.severity is Severity.warning]
    return RigReport(
        name=name,
        parts=len(rig.parts),
        params=len(rig.parameters),
        physics=len(rig.physics),
        lint_warnings=warnings,
        sweep=sweep_report(rig, steps=steps),
        landmark_warnings=list(landmark_warnings or []),
        motion_warnings=motion_issues(rig),
        plausibility_warnings=plausibility_issues(rig),
    )


def batch(named_rigs, *, steps: int = 9) -> BatchReport:
    """Evaluate many rigs. ``named_rigs`` is a mapping ``{name: Rig}`` or an iterable of
    ``(name, Rig)`` pairs."""
    pairs = named_rigs.items() if isinstance(named_rigs, dict) else named_rigs
    return BatchReport(items=[evaluate(rig, name, steps=steps) for name, rig in pairs])
