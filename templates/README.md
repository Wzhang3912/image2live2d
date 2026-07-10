# Templates

Hand-rigged base rigs (authored once in **nijigenerate**), used as fitting targets by the rig
authoring stage (`core/rig`). Each template defines the canonical part set, deformer hierarchy,
standard parameter set, and keyforms for one archetype.

Planned templates:

| Name | Archetype | Phase |
|---|---|---|
| `portrait_front` | front-facing head/shoulders | 1 |
| `portrait_3q` | three-quarter portrait | 2 |
| `chibi_fullbody` | full-body chibi | 3 |

Store nijilive `.inx` sources here; Route A (Phase 4) adds matching `.moc3`/`.cmo3` templates whose
slot layout mirrors these.
