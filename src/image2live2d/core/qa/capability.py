"""Rig **capability** report — an honest per-character summary of what the puppet can and cannot do.

The pipeline never crashes on a hard input; it emits *whatever rig the parts allow*. A gowned character
whose arms were never separated from the dress gets a rig with no arm articulation; a mouth decomposed
as a bare lip line with no synthesised interior can't actually open. Those are not bugs — they are the
honest ceiling of the parts we were handed — but today they ship **silently**, so a caller can't tell a
fully-articulated puppet from a degraded one without loading it into a viewer.

This module reads the finished ``Rig`` and answers "what does this puppet actually do?": which axes
articulate (head/body/arms/legs), which face features are riggable on both sides, whether the mouth can
open, how many physics chains are live, and which motion clips were dropped for lack of parts. It is a
pure function of the IRR — no runtime, no render — so it runs in CI and in the batch report.

It is deliberately *descriptive, not a gate*: `plausibility_issues` already decides "is this even a
riggable character?"; this says "given that it is, here's the puppet you got." Capability gaps surface
as ``notes`` (e.g. "no arm articulation", "mouth cannot open"), not failures.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from ...irr.schema import Rig, SemanticRole
from ..motion.coverage import motion_coverage
from ..motion.drive import DRIVE_NAMES, SWEEP_NAME


@dataclass
class Capability:
    """One articulation axis: is it present, and a one-line human detail."""

    name: str
    present: bool
    detail: str = ""


@dataclass
class CapabilityReport:
    """What a finished rig can and cannot do. Pure function of the IRR (see ``rig_capabilities``)."""

    name: str
    parts: int
    params: int
    capabilities: list[Capability] = field(default_factory=list)
    physics_chains: int = 0
    physics_excited: int = 0
    clips_shipped: list[str] = field(default_factory=list)
    clips_suppressed: list[str] = field(default_factory=list)   # drive clips dropped for missing params
    notes: list[str] = field(default_factory=list)              # honest degradations worth surfacing

    def has(self, name: str) -> bool:
        return any(c.present for c in self.capabilities if c.name == name)

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "parts": self.parts,
            "params": self.params,
            "capabilities": {c.name: c.present for c in self.capabilities},
            "physics": {"chains": self.physics_chains, "excited": self.physics_excited},
            "clips": {"shipped": self.clips_shipped, "suppressed": self.clips_suppressed},
            "notes": self.notes,
        }

    def format(self) -> str:
        lines = [f"{self.name}: {self.parts} parts, {self.params} params"]
        for c in self.capabilities:
            mark = "✓" if c.present else "✗"
            lines.append(f"  {mark} {c.name}" + (f"  ({c.detail})" if c.detail else ""))
        lines.append(f"  physics: {self.physics_excited}/{self.physics_chains} chains live")
        lines.append(f"  clips: {len(self.clips_shipped)} shipped"
                     + (f", {len(self.clips_suppressed)} suppressed "
                        f"({', '.join(self.clips_suppressed)})" if self.clips_suppressed else ""))
        for n in self.notes:
            lines.append(f"  ⚠ {n}")
        return "\n".join(lines)


def rig_capabilities(rig: Rig) -> CapabilityReport:
    """Summarise what ``rig`` can do, from its parts, parameters, physics and clips."""
    param_ids = {p.id for p in rig.parameters}
    roles = {p.semantic_role for p in rig.parts}
    has_p = param_ids.__contains__
    has_r = roles.__contains__

    caps: list[Capability] = []

    # Head turn (yaw/pitch/roll) is synthesised by the emitters from a warp grid / group rotation, NOT
    # by IRR keyform offsets — so it is present iff the params exist and a head is there to turn. Don't
    # test mesh offsets here or a real, working head turn reads as dead (the offsets are empty by design).
    caps.append(Capability("head_turn", has_p("ParamAngleX") and has_p("ParamAngleY")
                           and has_p("ParamAngleZ") and has_r(SemanticRole.face_base),
                           "yaw+pitch+roll"))
    caps.append(Capability("body_sway", has_p("ParamBodyAngleX") and has_p("ParamBodyAngleZ")))

    # Limbs are keyform-driven and only authored when the part was separated, so param presence is the
    # honest signal — and it's per-side, because a decomposer can hand us one arm and not the other.
    caps.append(Capability("arm_left", has_p("ParamArmLA")))
    caps.append(Capability("arm_right", has_p("ParamArmRA")))
    caps.append(Capability("leg_left", has_p("ParamLegLA")))
    caps.append(Capability("leg_right", has_p("ParamLegRA")))

    caps.append(Capability("blink_left", has_p("ParamEyeLOpen")))
    caps.append(Capability("blink_right", has_p("ParamEyeROpen")))
    caps.append(Capability("gaze", has_p("ParamEyeBallX") and has_p("ParamEyeBallY")))
    caps.append(Capability("brow_left", has_p("ParamBrowLY")))
    caps.append(Capability("brow_right", has_p("ParamBrowRY")))

    # A mouth can open only if it has an interior behind the lips (decomposed or synthesised). The lip
    # param on its own just parts a stroke over skin — see core.synth.mouth.
    mouth_open = has_p("ParamMouthOpenY")
    has_cavity = has_r(SemanticRole.mouth_cavity)
    caps.append(Capability("mouth_open", mouth_open and has_cavity,
                           "has interior" if has_cavity else "lip line only"))
    caps.append(Capability("mouth_form", has_p("ParamMouthForm")))

    expr = {"smile", "surprise", "sad", "angry"}
    shipped = [a.name for a in rig.animations]
    caps.append(Capability("expressions", bool(expr & set(shipped)),
                           ", ".join(sorted(expr & set(shipped)))))

    cov = motion_coverage(rig.parameters, rig.physics,
                          [a for a in rig.animations if a.name != SWEEP_NAME])
    excited = len(rig.physics) - len(cov.unexcited)

    # A drive clip is suppressed when the character lacks its parameters (generate_drives drops it), so
    # the gap between the standard drive set and what shipped is the honest "what this puppet can't do".
    suppressed = [n for n in DRIVE_NAMES if n not in shipped]

    notes: list[str] = []
    if not has_p("ParamArmLA") and not has_p("ParamArmRA"):
        notes.append("no arm articulation — arms were not separated from the body/clothing")
    elif has_p("ParamArmLA") != has_p("ParamArmRA"):
        notes.append("only one arm is riggable — the other was not separated")
    if not has_p("ParamLegLA") and not has_p("ParamLegRA"):
        notes.append("no leg articulation — legs were not separated")
    if mouth_open and not has_cavity:
        notes.append("mouth cannot open — no interior behind the lips (talk only parts the lip line)")
    if has_r(SemanticRole.eyebrow_l) != has_r(SemanticRole.eyebrow_r):
        notes.append("brows are one-sided — only one eyebrow was decomposed")

    return CapabilityReport(
        name=rig.meta.name if rig.meta else "rig",
        parts=len(rig.parts),
        params=len(rig.parameters),
        capabilities=caps,
        physics_chains=len(rig.physics),
        physics_excited=excited,
        clips_shipped=shipped,
        clips_suppressed=suppressed,
        notes=notes,
    )
