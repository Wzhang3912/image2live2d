"""The in-browser live runtime (``app/index.html``) is a *third* renderer of the same rig, and it had
no tests — which is exactly how it drifted: it skipped every parameter sitting at 0, so the closed eye
(``ParamEyeLOpen`` default 1, **min 0**) never deformed and blink was a silent no-op, while the mouth
cavity (whose *0* keyform is what squashes it shut) hung open in every frame including rest.

The native Cubism core always evaluates every parameter at its default. These tests run the *real* JS
out of index.html under Node and pin that same contract, so the preview can't quietly disagree with
the .moc3 again.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pytest

INDEX = Path(__file__).resolve().parents[1] / "src" / "image2live2d" / "app" / "index.html"


def _extract(src: str, name: str) -> str:
    """Pull `function <name>(...){...}` out of the page by brace matching (no JS parser needed)."""
    i = src.index(f"function {name}(")
    depth = 0
    for k in range(src.index("{", i), len(src)):
        if src[k] == "{":
            depth += 1
        elif src[k] == "}":
            depth -= 1
            if depth == 0:
                return src[i : k + 1]
    raise AssertionError(f"unbalanced braces extracting {name}()")


def _run(rig: dict, body: str):
    """Run the page's own interp/markNeutralKeyforms/accum against `rig` in Node."""
    node = shutil.which("node")
    if node is None:
        pytest.skip("node not installed — browser-runtime JS not exercised")
    fns = "\n".join(_extract(INDEX.read_text(encoding="utf-8"), n)
                    for n in ("interp", "markNeutralKeyforms", "accum"))
    prog = (
        f"const RIG = {json.dumps(rig)};\n"
        "const HTURN = {}, BTURN = {};\n"
        "const HEADSET = new Set(), BODYSET = new Set();\n"
        f"{fns}\n{body}\n"
    )
    r = subprocess.run([node, "-e", prog], capture_output=True, text=True)
    assert r.returncode == 0, f"node failed:\n{r.stderr}"
    return json.loads(r.stdout)


def _kf(value: float, part: str, dy: float):
    return {"value": value, "offsets": {part: [[0.0, dy], [0.0, dy]]}}


def _rig() -> dict:
    """The three parameter shapes that matter, mirroring what the authoring stage really emits."""
    return {"params": [
        # the eye: neutral is the MAX (open); the 0 end is the closed lid that must deform
        {"id": "ParamEyeLOpen", "min": 0.0, "max": 1.0, "default": 1.0,
         "keyforms": [_kf(0.0, "0", -0.011), _kf(1.0, "0", 0.0)]},
        # the mouth cavity: neutral IS 0, and the 0 keyform is what collapses it shut
        {"id": "ParamMouthOpenY", "min": 0.0, "max": 1.0, "default": 0.0,
         "keyforms": [_kf(0.0, "1", -0.005), _kf(1.0, "1", 0.0)]},
        # an ordinary param: neutral 0 with a genuinely zero keyform (the skippable majority)
        {"id": "ParamArmLA", "min": -10.0, "max": 10.0, "default": 0.0,
         "keyforms": [_kf(-10.0, "2", -0.02), _kf(0.0, "2", 0.0), _kf(10.0, "2", 0.02)]},
    ]}


def test_closed_eye_actually_deforms_the_lid():
    """ParamEyeLOpen=0 is the *fully closed* eye — the single most important pose for that param, and
    the exact value the old `v===0` skip threw away. Blink rendered as a no-op for it."""
    out = _run(_rig(), """
        markNeutralKeyforms();
        const acc = accum({ParamEyeLOpen: 0});
        console.log(JSON.stringify({parts: Object.keys(acc), dy: acc["0"] ? acc["0"][0][1] : 0}));
    """)
    assert "0" in out["parts"], "closed eye deformed nothing — blink is dead in the live view"
    assert out["dy"] == pytest.approx(-0.011)


def test_undriven_mouth_still_collapses_its_cavity():
    """Nothing drives the mouth in the idle loop, but Cubism still evaluates it at its default — and
    this param's default keyform is what squashes the cavity shut. Skipping undriven params left the
    mouth hanging open in every frame, rest included."""
    out = _run(_rig(), """
        markNeutralKeyforms();
        const acc = accum({});                     // nothing driven at all
        console.log(JSON.stringify({parts: Object.keys(acc), dy: acc["1"] ? acc["1"][0][1] : 0}));
    """)
    assert "1" in out["parts"], "mouth cavity never collapsed — the mouth hangs open at rest"
    assert out["dy"] == pytest.approx(-0.005)


def test_a_param_whose_default_is_zero_offset_is_still_skipped():
    """The skip is a real optimization — keep it for the params it's *sound* for, so the rasterizer
    doesn't interpolate 30 no-op keyforms a frame. Only the default-deforms case must evaluate."""
    out = _run(_rig(), """
        markNeutralKeyforms();
        const flags = {};
        for (const p of RIG.params) flags[p.id] = !!p._neutralZero;
        const acc = accum({ParamArmLA: 0});
        console.log(JSON.stringify({flags, armParts: Object.keys(acc).includes("2")}));
    """)
    assert out["flags"]["ParamArmLA"] is True          # ordinary param: safe to skip at neutral
    assert out["flags"]["ParamEyeLOpen"] is True       # its default (open) keyform is zero-offset
    assert out["flags"]["ParamMouthOpenY"] is False    # its default keyform genuinely deforms
    assert out["armParts"] is False, "a zero-offset neutral should contribute nothing"


def test_driven_params_still_deform():
    """Guard the obvious: the fix must not break the params that already worked."""
    out = _run(_rig(), """
        markNeutralKeyforms();
        const acc = accum({ParamArmLA: 10, ParamMouthOpenY: 1});
        console.log(JSON.stringify({arm: acc["2"][0][1], mouth: acc["1"][0][1]}));
    """)
    assert out["arm"] == pytest.approx(0.02)
    assert out["mouth"] == pytest.approx(0.0)   # cavity open at MouthOpenY=1
