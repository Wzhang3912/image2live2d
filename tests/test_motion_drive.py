"""The drive sheet + motion coverage — "does the motion actually move the rig we built?"

These exist because of a bug that shipped: the idle drove 6 of a real character's 31 parameters and
never touched ``ParamAngleX/Y/Z``, so five of eight physics chains had a flat-lined driver and the hair
physics we had tuned against a pro rig never swung once. Every check below is either a regression test
for that, or a guard against the next version of it.
"""

from __future__ import annotations

from image2live2d.core.motion import (
    DRIVE_NAMES,
    MIN_SWING_FRAC,
    SWEEP_NAME,
    generate_all,
    generate_drives,
    generate_idle,
    generate_sweep,
    motion_coverage,
)
from image2live2d.core.motion.drive import _HOLD, _SNAP
from image2live2d.irr.schema import (
    AnimKeyframe,
    Animation,
    AnimationLane,
    InterpolateMode,
    Parameter,
    PhysicsRig,
)


def _param(pid: str, lo: float, hi: float, default: float = 0.0) -> Parameter:
    return Parameter(id=pid, name=pid, min=lo, max=hi, default=default)


def _full_body_params() -> list[Parameter]:
    """The parameter shape of a real rigged character: face + head + body + limbs + physics outputs."""
    return [
        _param("ParamEyeLOpen", 0.0, 1.0, 1.0), _param("ParamEyeROpen", 0.0, 1.0, 1.0),
        _param("ParamMouthOpenY", 0.0, 1.0), _param("ParamMouthForm", -1.0, 1.0),
        _param("ParamAngleX", -30.0, 30.0), _param("ParamAngleY", -30.0, 30.0),
        _param("ParamAngleZ", -30.0, 30.0),
        _param("ParamEyeBallX", -1.0, 1.0), _param("ParamEyeBallY", -1.0, 1.0),
        _param("ParamBrowLY", -1.0, 1.0), _param("ParamBrowRY", -1.0, 1.0),
        _param("ParamBodyAngleX", -10.0, 10.0), _param("ParamBodyAngleY", -10.0, 10.0),
        _param("ParamBodyAngleZ", -10.0, 10.0),
        _param("ParamArmLA", -10.0, 10.0), _param("ParamArmLB", -10.0, 10.0),
        _param("ParamArmRA", -10.0, 10.0), _param("ParamArmRB", -10.0, 10.0),
        _param("ParamLegLA", -10.0, 10.0), _param("ParamLegLB", -10.0, 10.0),
        _param("ParamLegRA", -10.0, 10.0), _param("ParamLegRB", -10.0, 10.0),
        _param("ParamBreath", 0.0, 1.0),
        # physics OUTPUTS — written by the pendulum, never keyed by a clip
        _param("ParamHairFront", -1.0, 1.0), _param("ParamHairBack", -1.0, 1.0),
        _param("ParamCloth0", -1.0, 1.0), _param("ParamAcc0", -1.0, 1.0),
    ]


def _physics() -> list[PhysicsRig]:
    """Hair + accessories hang off the head; cloth off the body — the real wiring."""
    head = dict(driver_param="ParamAngleX", extra_drivers=["ParamAngleY", "ParamAngleZ"])
    return [
        PhysicsRig(id="p1", output_param="ParamHairFront", **head),
        PhysicsRig(id="p2", output_param="ParamHairBack", **head),
        PhysicsRig(id="p3", output_param="ParamAcc0", **head),
        PhysicsRig(id="p4", output_param="ParamCloth0", driver_param="ParamBodyAngleX",
                   extra_drivers=["ParamBodyAngleY", "ParamBodyAngleZ"]),
    ]


# --------------------------------------------------------------------------------------------------
# The regression: idle has to excite the physics, or the physics is decorative
# --------------------------------------------------------------------------------------------------
def test_idle_turns_the_head():
    """The original bug in one line: the idle never drove the head, and the head is what drives hair."""
    lanes = {ln.param_id for a in generate_idle(_full_body_params()) for ln in a.lanes}
    assert {"ParamAngleX", "ParamAngleY", "ParamAngleZ"} <= lanes


def test_idle_alone_excites_every_physics_chain():
    """A character standing still must still have moving hair. This is the check that would have caught
    the shipped bug: before the fix, 5 of 8 chains had a driver the idle never moved."""
    params, phys = _full_body_params(), _physics()
    cov = motion_coverage(params, phys, generate_idle(params))
    assert cov.unexcited == [], f"idle leaves physics dead: {cov.unexcited}"


def test_idle_head_motion_has_a_quick_move_in_it():
    """Turning the head is necessary but *not sufficient* — the move also has to be quick.

    A pendulum responds to acceleration, not position. Drift the head through a slow six-second sine and
    the hair just hangs straight down from wherever the head currently is, so the relative angle the
    physics output measures stays near zero: simulated against the real Cubism model, a 12-degree slow
    drift moves ParamHairFront by 1% of its range. Adding the accent — one brief glance per loop — took
    that to 10%. This test is the guard: it fails if someone smooths the idle back into a pure drift.
    """
    idle = generate_idle(_full_body_params())[0]
    yaw = next(ln for ln in idle.lanes if ln.param_id == "ParamAngleX")
    rates = [abs(b.value - a.value) / max(b.frame - a.frame, 1)
             for a, b in zip(yaw.keyframes, yaw.keyframes[1:])]
    # degrees per frame at the sharpest point of the glance — a pure 6-degree sine over 360 frames peaks
    # at about 0.1 deg/frame, which is what left the hair frozen
    assert max(rates) > 0.25, f"idle head motion is too slow to swing anything: {max(rates):.3f} deg/frame"


def test_idle_loops_without_a_twitch():
    """A discontinuity at the loop point would kick every pendulum once per loop — a visible jerk."""
    for anim in generate_idle(_full_body_params()):
        for lane in anim.lanes:
            first, last = lane.keyframes[0], lane.keyframes[-1]
            assert last.frame == anim.length
            assert abs(first.value - last.value) < 1e-6, f"{lane.param_id} does not close its loop"


# --------------------------------------------------------------------------------------------------
# The drive sheet
# --------------------------------------------------------------------------------------------------
def test_every_clip_is_authored_for_a_full_body_rig():
    names = [a.name for a in generate_drives(_full_body_params(), _physics())]
    assert names == list(DRIVE_NAMES)


def test_clips_are_present_gated():
    """A bare portrait has no legs to swing — it gets the face clips and nothing else."""
    face = [_param("ParamEyeLOpen", 0.0, 1.0, 1.0), _param("ParamMouthOpenY", 0.0, 1.0),
            _param("ParamAngleX", -30.0, 30.0)]
    names = {a.name for a in generate_drives(face, [])}
    assert "head_yaw" in names and "talk" in names and "blink" in names
    assert "legs_swing" not in names and "arms_swing" not in names and "body_sway" not in names


def test_no_clip_keys_a_physics_output():
    """A keyframe on ParamHairFront is a lane arguing with the pendulum that writes it — and losing."""
    params, phys = _full_body_params(), _physics()
    outputs = {r.output_param for r in phys}
    for anim in generate_all(params, phys):
        keyed = {ln.param_id for ln in anim.lanes} & outputs
        assert not keyed, f"{anim.name} keys physics output(s) {keyed}"


def test_a_clip_snaps_then_holds_then_settles():
    """The shape is the whole point: a pendulum is excited by *velocity*, so the drive has to be a snap
    (fast enough to swing the hair) followed by a hold and a settle (still enough to watch it swing)."""
    yaw = next(a for a in generate_drives(_full_body_params(), _physics()) if a.name == "head_yaw")
    lane = yaw.lanes[0]
    frames = {kf.frame: kf.value for kf in lane.keyframes}

    assert frames[0] == 0.0                       # starts neutral
    assert frames[_SNAP] == 30.0                  # ...and is at full extent 8 frames later: the impulse
    assert frames[_SNAP + _HOLD] == 30.0          # held there — this is where the lag is visible
    assert frames[lane.keyframes[-1].frame] == 0.0  # home at the end...
    # ...and *stays* home for a settle tail: driver at rest while the physics is still moving. Without
    # this, the clip ends the instant the head stops and you never see the follow-through.
    home = [kf.frame for kf in lane.keyframes if kf.value == 0.0]
    assert home[-1] - home[-2] >= 30


def test_drives_reach_the_extremes_of_every_range():
    """The cardboard-arm stretch was found at the END of ParamAngleX's travel, not in the middle. Clips
    that stop short of the extremes would not have found it."""
    params = _full_body_params()
    by_id = {p.id: p for p in params}
    # a parameter is driven by several clips (head_yaw takes ParamAngleX to its extremes; head_shake
    # only to 80% of them, on purpose) — the sheet reaches the extreme if ANY clip does
    swings: dict[str, list[float]] = {}
    for anim in generate_drives(params, _physics()):
        for lane in anim.lanes:
            swings.setdefault(lane.param_id, []).extend(kf.value for kf in lane.keyframes)
    for pid in ("ParamAngleX", "ParamAngleY", "ParamAngleZ", "ParamBodyAngleX"):
        assert max(swings[pid]) == by_id[pid].max
        assert min(swings[pid]) == by_id[pid].min


def test_limb_clips_move_left_and_right_in_opposite_phase():
    """The de-cardboard check. If the L/R split failed, both arms move as one sheet — and a clip that
    drove them in the SAME direction could never show it."""
    swing = next(a for a in generate_drives(_full_body_params(), _physics()) if a.name == "arms_swing")
    peaks = {ln.param_id: max(kf.value for kf in ln.keyframes) for ln in swing.lanes}
    troughs = {ln.param_id: min(kf.value for kf in ln.keyframes) for ln in swing.lanes}
    # left peaks positive as right troughs negative — they are never at the same place at the same time
    assert peaks["ParamArmLA"] > 0 > troughs["ParamArmLA"]
    assert peaks["ParamArmRA"] > 0 > troughs["ParamArmRA"]
    lane = {ln.param_id: {kf.frame: kf.value for kf in ln.keyframes} for ln in swing.lanes}
    at_snap_l = lane["ParamArmLA"][_SNAP]
    at_snap_r = lane["ParamArmRA"][_SNAP]
    assert at_snap_l == -at_snap_r != 0.0


def test_asymmetric_ranges_drive_to_the_right_end():
    """ParamEyeLOpen is 0..1 with a default of 1: 'drive it negative' means shut, and there is no
    positive direction to go. A clip that assumed a range centred on its default would drive it out of
    bounds or nowhere."""
    blink = next(a for a in generate_drives(_full_body_params(), _physics()) if a.name == "blink")
    values = [kf.value for ln in blink.lanes for kf in ln.keyframes]
    assert min(values) == 0.0 and max(values) == 1.0    # shut and open, never outside [0, 1]


# --------------------------------------------------------------------------------------------------
# The sweep
# --------------------------------------------------------------------------------------------------
def test_sweep_drives_every_parameter_that_is_not_a_physics_output():
    params, phys = _full_body_params(), _physics()
    outputs = {r.output_param for r in phys}
    sweep = generate_sweep(params, phys)[0]
    assert sweep.name == SWEEP_NAME
    driven = {ln.param_id for ln in sweep.lanes}
    assert driven == {p.id for p in params} - outputs


def test_sweep_moves_one_parameter_at_a_time():
    """So that whatever you are looking at is attributable to exactly one parameter. Overlapping lanes
    would make a defect impossible to pin on a param without bisecting by hand."""
    params, phys = _full_body_params(), _physics()
    sweep = generate_sweep(params, phys)[0]

    def active(lane: AnimationLane) -> tuple[int, int]:
        moving = [kf.frame for kf in lane.keyframes if kf.value != _default(params, lane.param_id)]
        return min(moving), max(moving)

    spans = sorted(active(ln) for ln in sweep.lanes)
    for (_, end), (start, _) in zip(spans, spans[1:]):
        assert start >= end, "two parameters move at the same time in the sweep"


def _default(params: list[Parameter], pid: str) -> float:
    return next(p.default for p in params if p.id == pid)


# --------------------------------------------------------------------------------------------------
# Coverage
# --------------------------------------------------------------------------------------------------
def test_coverage_is_clean_for_the_shipped_motion_set():
    params, phys = _full_body_params(), _physics()
    natural = [a for a in generate_all(params, phys) if a.name != SWEEP_NAME]
    cov = motion_coverage(params, phys, natural)
    assert cov.ok, cov.format()


def test_coverage_catches_the_flat_lined_driver():
    """Reconstruct the actual shipped bug — a body-only idle — and prove the check fails it."""
    params, phys = _full_body_params(), _physics()
    body_only = Animation(
        name="idle", fps=60.0, length=360, loop=True,
        lanes=[AnimationLane(
            param_id="ParamBodyAngleX",
            keyframes=[AnimKeyframe(frame=0, value=0.0), AnimKeyframe(frame=180, value=0.5),
                       AnimKeyframe(frame=360, value=0.0)],
            interpolation=InterpolateMode.cubic)],
    )
    cov = motion_coverage(params, phys, [body_only])
    assert not cov.ok
    # every head-driven chain is dead, and the body chain too: 0.5 deg of a +/-10 range is 5% — under
    # the threshold, and on screen it is nothing at all
    assert set(cov.unexcited) == {"ParamHairFront", "ParamHairBack", "ParamAcc0", "ParamCloth0"}


def test_a_driver_that_barely_moves_does_not_count_as_exciting_anything():
    params, phys = _full_body_params(), _physics()
    tiny = MIN_SWING_FRAC * 0.5 * 60.0 / 2.0   # half the threshold, as a +/- amplitude on a 60-wide range
    anim = Animation(
        name="idle", fps=60.0, length=120, loop=True,
        lanes=[AnimationLane(
            param_id="ParamAngleX",
            keyframes=[AnimKeyframe(frame=0, value=-tiny), AnimKeyframe(frame=120, value=tiny)],
            interpolation=InterpolateMode.cubic)],
    )
    cov = motion_coverage(params, phys, [anim])
    assert "ParamAngleX" in cov.driven          # it does move...
    assert "ParamHairFront" in cov.unexcited    # ...but not enough to swing anything


def test_coverage_catches_a_lane_fighting_a_pendulum():
    params, phys = _full_body_params(), _physics()
    anim = Animation(
        name="bad", fps=60.0, length=60, loop=False,
        lanes=[AnimationLane(
            param_id="ParamHairFront",   # the pendulum writes this every frame; the lane is dead
            keyframes=[AnimKeyframe(frame=0, value=0.0), AnimKeyframe(frame=60, value=1.0)],
            interpolation=InterpolateMode.linear)],
    )
    cov = motion_coverage(params, phys, [anim])
    assert cov.keyed_outputs == ["ParamHairFront"] and not cov.ok


def test_a_flat_lane_is_not_motion():
    """Holding a parameter at one value for a whole clip drives nothing — it must not count as coverage."""
    params, phys = _full_body_params(), _physics()
    anim = Animation(
        name="held", fps=60.0, length=60, loop=False,
        lanes=[AnimationLane(
            param_id="ParamAngleX",
            keyframes=[AnimKeyframe(frame=0, value=5.0), AnimKeyframe(frame=60, value=5.0)],
            interpolation=InterpolateMode.linear)],
    )
    cov = motion_coverage(params, phys, [anim])
    assert "ParamAngleX" not in cov.driven
    assert "ParamAngleX" in cov.undriven
