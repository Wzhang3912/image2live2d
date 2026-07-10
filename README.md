# image2live2d

Turn character art into a **reusable, riggable 2D puppet** that animates interactively — blink,
lip-sync, gaze, head-turn, brows, limbs, and self-animating hair/cloth physics — at near-zero
per-frame cost. Instead of baking a video, you get a rig you can drive live.

The pipeline builds an internal **rig representation** from separated art layers, then emits it to
standard animation formats:

- **nijilive `.inp`** — open format; opens directly in [nijigenerate](https://github.com/nijigenerate/nijigenerate).
- **Live2D `.moc3` bundle** — `.moc3` + `.model3.json` / `.physics3.json` / `.motion3.json` / `.cdi3.json`,
  loadable in the Cubism Viewer and standard Live2D runtimes.

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

## Development

```bash
pip install -e ".[dev]"
python -m pytest -q
```

## License

See [LICENSE](LICENSE).
