# image2live2d

Turn character art into a **reusable, riggable 2D puppet** that animates interactively — blink,
lip-sync, gaze, head-turn, brows, limbs, and self-animating hair/cloth physics — at near-zero
per-frame cost. Instead of baking a video, you get a rig you can drive live.

![Upload an image → automatic decompose, rig & physics → live render in the browser](media/demo.gif)

*Drop a character image, and it's automatically decomposed, rigged, and rendered live in the browser
— idling on its own and following the cursor.*

The pipeline builds an internal **rig representation** from separated art layers, then emits it to
standard animation formats:

- **nijilive `.inp`** — open format; opens directly in [nijigenerate](https://github.com/nijigenerate/nijigenerate).
- **Live2D `.moc3` bundle** — `.moc3` + `.model3.json` / `.physics3.json` / `.motion3.json` / `.cdi3.json`,
  targeting the Cubism Viewer and standard Live2D runtimes (compatibility still being hardened — see
  [Work in progress](#work-in-progress)).

```
ingest → preprocess → decompose → mesh → rig authoring → physics → motion
                                                             ├── nijilive emitter → .inp
                                                             └── live2d emitter   → .moc3 bundle
```

## Quick start

```bash
pip install -e ".[decompose]"            # Pillow + psd-tools (PSD / layer input)

# convert a folder of {order}_{role}.png layers → animatable .inp
python -m image2live2d path/to/layers -o character.inp

# or a layered PSD
python -m image2live2d --psd hero.psd -o hero.inp

# also emit a Live2D bundle alongside the .inp
python -m image2live2d --psd hero.psd -o hero.inp --live2d hero_live2d/

# try a built-in sample (face or full-body)
python -m image2live2d --sample --fullbody -o sample.inp

# batch a whole roster with a pass-rate report
python -m image2live2d --batch path/to/roster -o out/

# local web app: drag-drop a PSD / zip of layers, preview and download
python -m image2live2d --serve            # http://127.0.0.1:8000
```

```python
from image2live2d import convert_psd

result = convert_psd("hero.psd", "out/", live2d=True)   # nijilive .inp + Live2D bundle
print(result.inp_path, result.passed)
```

The generated rig animates blink, mouth, per-character head turn, eye gaze, brows, and limbs, plus
procedural hair/cloth physics and a looping idle.

## Input

The converter takes **already-separated layers** — a layered PSD, or a folder/zip of
`{order}_{role}.png` files (e.g. `00_hair_back.png`, `13_face_base.png`). Layer names drive role
detection (face, eyes, hair, limbs, clothing, …), meshing, and rig authoring.

To start from a **single flat image**, the pipeline calls a [See-through](#credits) decompose service
(GPU) that splits the illustration into ordered layers first — this is what the demo above shows. See
[`service/seethrough/`](service/seethrough) for the service, and [Work in progress](#work-in-progress)
for the licensing caveat.

## Layout

```
src/image2live2d/
  irr/        # rig representation — the schema everything is built around
  core/       # pipeline stages: decompose, mesh, landmark, rig, physics, motion, qa
  backends/   # emitters: nijilive (.inp) and Live2D (.moc3 bundle)
  api.py      # public API: convert_layers / convert_psd
  batch.py    # batch conversion + aggregate QA
  app/        # local web app
tests/
```

## Work in progress

- **Live2D `.moc3` export** is functional and renders in the Cubism Viewer, but broader Cubism Editor /
  runtime and VTube Studio compatibility is still being hardened. The nijilive `.inp` path is the more
  mature, fully-headless target today.
- Richer expression presets, more body/clothing archetypes, and improved automatic landmark/pose
  detection.

Contributions and issues welcome.

## Credits

Single-image layer decomposition is powered by **See-through**, the automatic anime-character layer
decomposer that turns one flat illustration into ordered layers — which this project then meshes, rigs,
and animates.

> Jian Lin, Chengze Li, Haoyun Qin, Kwun Wang Chan, Yanghua Jin, Hanyuan Liu, Stephen Chun Wang Choy,
> Xueting Liu. **_See-through: Single-image Layer Decomposition for Anime Characters._** SIGGRAPH 2026.
> [arXiv:2602.03749](https://arxiv.org/abs/2602.03749) ·
> [code](https://github.com/shitagaki-lab/see-through) (Apache-2.0)

This project also builds on [nijilive / nijigenerate](https://github.com/nijigenerate/nijigenerate)
(the open 2D-puppet format and editor) and the [Live2D Cubism](https://www.live2d.com/) `.moc3` format.
See-through and nijilive are independent projects and are not affiliated with this repository.

## Development

```bash
pip install -e ".[dev]"
python -m pytest -q
```

## License

See [LICENSE](LICENSE). Third-party components (See-through, nijilive, Live2D Cubism) are governed by
their own licenses.
