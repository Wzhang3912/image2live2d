"""IRR -> Cubism ``.cdi3.json`` (Cubism Display Info, open JSON).

Display names + folder grouping for parameters and parts, shown in the Cubism Editor / some viewers.
Purely cosmetic metadata, but cheap to author from the IRR and expected by a complete model bundle.
"""

from __future__ import annotations

from ...irr.schema import Rig
from . import mapping

CDI_VERSION = 3


def cdi3(rig: Rig) -> dict:
    """Build the ``.cdi3.json`` document for ``rig``."""
    parameters = [
        {"Id": p.id, "GroupId": mapping.group_id(p.id), "Name": mapping.display_name(p.id)}
        for p in rig.parameters
    ]
    parameter_groups = [
        {"Id": g, "GroupId": "", "Name": g} for g in mapping.used_groups(rig)
    ]
    parts = [
        {"Id": part.id, "Name": mapping.part_display_name(part.semantic_role)}
        for part in rig.parts
    ]
    return {
        "Version": CDI_VERSION,
        "Parameters": parameters,
        "ParameterGroups": parameter_groups,
        "Parts": parts,
    }
