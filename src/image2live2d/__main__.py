"""Command-line entry point: image (layer dir) -> riggable ``.inp``.

Examples
--------
Generate a throwaway sample face and convert it (one command, drop the result in nijigenerate)::

    python -m image2live2d --sample -o sample.inp

Convert an existing folder of ``{order}_{role}.png`` layers::

    python -m image2live2d path/to/layers -o character.inp
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .backends.nijilive import NijiliveEmitter
from .core import decompose
from .core.qa import sweep_report
from .irr.validate import Severity, lint
from .pipeline import rig_from_stack


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="image2live2d",
        description="Convert a folder of decomposed character layers into a nijilive .inp puppet.",
    )
    parser.add_argument(
        "layer_dir",
        nargs="?",
        help="directory of '{order}_{role}.png' layers (omit when using --sample)",
    )
    parser.add_argument(
        "--sample",
        nargs="?",
        const="sample_layers",
        metavar="DIR",
        help="generate a throwaway sample face into DIR (default ./sample_layers), then convert",
    )
    parser.add_argument("--fullbody", action="store_true", help="with --sample, generate a full-body character")
    parser.add_argument("--psd", metavar="FILE", help="a layered PSD (e.g. See-through output)")
    parser.add_argument(
        "--image", metavar="FILE",
        help="a single flat image; requires --decompose-url (remote See-through GPU service)",
    )
    parser.add_argument(
        "--decompose-url", metavar="URL",
        help="See-through decompose service base URL (see docs/DECOMPOSE_SERVICE.md)",
    )
    parser.add_argument("--decompose-token", metavar="TOKEN", help="auth token for the decompose service")
    parser.add_argument(
        "--gcp-instance", metavar="NAME",
        help="with --image: GCP VM to auto-start for decomposition and auto-stop when done",
    )
    parser.add_argument("--gcp-zone", metavar="ZONE", help="GCP zone of --gcp-instance")
    parser.add_argument("--gcp-port", type=int, default=8000, help="decompose service port (default 8000)")
    parser.add_argument("--gcp-project", metavar="PROJECT", help="GCP project (default: gcloud default)")
    parser.add_argument(
        "--gcp-keep-running", action="store_true",
        help="don't stop the GCP VM after inference (leave it warm for more images)",
    )
    parser.add_argument(
        "--work-dir", metavar="DIR", help="where --psd/--image extract layer PNGs (default ./<name>_layers)"
    )
    parser.add_argument("-o", "--out", help="output .inp path (default ./<name>.inp)")
    parser.add_argument("-n", "--name", help="model name (default: derived from output/layer dir)")
    parser.add_argument("--grid", type=int, default=10, help="mesh grid resolution (default 10)")
    parser.add_argument(
        "--live2d",
        nargs="?",
        const="",
        metavar="DIR",
        help="also emit a Live2D (Route A) bundle into DIR (default ./<name>_live2d). JSON-only "
        "(model3/physics3/motion3/cdi3) until a .moc3 template is supplied — see docs/PHASE4_PLAN.md",
    )
    parser.add_argument(
        "--landmarks",
        nargs="?",
        const="",
        metavar="OVERLAY_PNG",
        help="extract silhouette landmarks; print a summary + warnings, and (if a path is given) "
        "write a debug overlay PNG over the composited character",
    )
    parser.add_argument(
        "--qa",
        nargs="*",
        metavar="LAYER_DIR",
        help="run the QA pass-rate harness over the given layer dirs (or built-in samples if none) "
        "and exit; non-zero exit if pass-rate < 100%%",
    )
    parser.add_argument(
        "--batch",
        metavar="ROOT",
        help="convert every PSD / layer dir under ROOT to a .inp (into -o dir, default ./batch_out) "
        "and exit; prints an aggregate QA report; non-zero exit on any error or QA fail",
    )
    parser.add_argument(
        "--serve",
        nargs="?",
        const=8000,
        type=int,
        metavar="PORT",
        help="launch the local web app (default port 8000): upload a PSD / zip of layers, get a .inp",
    )
    args = parser.parse_args(argv)

    if args.serve is not None:
        from .app import serve

        serve(port=args.serve)
        return 0

    if args.batch is not None:
        return _run_batch(args.batch, args.out, live2d=args.live2d is not None)

    if args.qa is not None:
        return _run_qa(args.qa)

    # Resolve the layer source: --image (remote decompose), --psd, --sample, or a layer dir.
    layer_dir: Path | None = None
    psd_path: Path | None = None
    image_path: Path | None = None
    if args.image:
        if not args.decompose_url and not args.gcp_instance:
            parser.error("--image requires --decompose-url OR --gcp-instance (+ --gcp-zone)")
        if args.gcp_instance and not args.gcp_zone:
            parser.error("--gcp-instance requires --gcp-zone")
        image_path = Path(args.image)
    elif args.psd:
        psd_path = Path(args.psd)
    elif args.sample is not None:
        try:
            from .samples import make_sample_fullbody, make_sample_layers
        except ImportError:  # pragma: no cover - Pillow missing
            print("error: --sample needs Pillow (pip install pillow)", file=sys.stderr)
            return 2
        gen = make_sample_fullbody if args.fullbody else make_sample_layers
        layer_dir = gen(args.sample)
        print(f"generated sample layers -> {layer_dir}")
    elif args.layer_dir:
        layer_dir = Path(args.layer_dir)
    else:
        parser.error("provide a layer_dir, --sample, or --psd")

    # Resolve name + output path.
    if args.name:
        name = args.name
    elif args.out:
        name = Path(args.out).stem
    elif args.sample is not None:
        name = "sample"
    elif image_path is not None:
        name = image_path.stem
    elif psd_path is not None:
        name = psd_path.stem
    else:
        name = Path(layer_dir).name
    out = Path(args.out) if args.out else Path.cwd() / f"{name}.inp"
    out.parent.mkdir(parents=True, exist_ok=True)

    # Build the rig (stages 2-6).
    try:
        if image_path is not None:
            work = Path(args.work_dir) if args.work_dir else out.parent / f"{name}_layers"
            if args.gcp_instance:
                from .gpu import decompose_managed

                print(f"starting GCP VM {args.gcp_instance} ({args.gcp_zone}) for decomposition…")
                stack = decompose_managed(
                    image_path, work, instance=args.gcp_instance, zone=args.gcp_zone,
                    port=args.gcp_port, token=args.decompose_token, project=args.gcp_project,
                    stop_on_finish=not args.gcp_keep_running,
                )
                print(f"VM {'left running' if args.gcp_keep_running else 'stopped'}")
            else:
                print(f"decomposing {image_path} via {args.decompose_url} (remote GPU)…")
                stack = decompose.from_service(
                    image_path, args.decompose_url, work, token=args.decompose_token
                )
            asset_root: Path = work
            print(f"got {len(stack.layers)} layers -> {work}")
        elif psd_path is not None:
            work = Path(args.work_dir) if args.work_dir else out.parent / f"{name}_layers"
            stack = decompose.from_psd(psd_path, work)
            asset_root = work
            print(f"extracted {len(stack.layers)} layers from {psd_path} -> {work}")
        else:
            stack = decompose.from_layer_dir(layer_dir)
            asset_root = layer_dir
        rig = rig_from_stack(stack, name=name, source=str(image_path or psd_path or layer_dir))
    except ImportError as exc:
        print(f"error: --psd/--image need the decompose extra (pip install psd-tools pillow): {exc}",
              file=sys.stderr)
        return 2
    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    written = NijiliveEmitter(asset_root=asset_root).emit(rig, out.parent)
    if written != out:  # name/out stem mismatch -> place it exactly where asked
        written.replace(out)
        written = out

    # Report.
    warnings = [i for i in lint(rig) if i.severity is Severity.warning]
    report = sweep_report(rig)
    print(f"wrote {written}")
    print(
        f"  parts={len(rig.parts)}  params={len(rig.parameters)}  "
        f"lint_warnings={len(warnings)}  sweep={'PASS' if report.passed else 'FAIL'}"
    )
    for w in warnings:
        print(f"  [warn] {w.code}: {w.message}")

    if args.landmarks is not None:
        _report_landmarks(stack, overlay_path=args.landmarks or None)

    if args.live2d is not None:
        from .backends.live2d import Live2DEmitter

        l2d_dir = Path(args.live2d) if args.live2d else out.parent / f"{name}_live2d"
        bundle = Live2DEmitter(asset_root=asset_root).build(rig, l2d_dir)
        moc = "yes" if bundle.moc_written else "stub (needs a .moc3 template — see docs/PHASE4_PLAN.md)"
        print(f"  wrote Live2D bundle -> {bundle.model3_path}  ({len(bundle.files)} files, moc3={moc})")

    print(
        "  open in nijigenerate and drive: ParamEyeLOpen / ParamMouthOpenY / ParamAngleX "
        "to check the quality gate"
    )
    return 0


def _report_landmarks(stack, *, overlay_path: str | None) -> None:
    """Extract + summarize silhouette landmarks; optionally write a debug overlay PNG."""
    from .core import landmark

    lm = landmark.extract_landmarks(stack)
    found = []
    if lm.face_oval:
        found.append("face_oval")
    found += [n for n, v in (("eye_l", lm.eye_l), ("eye_r", lm.eye_r), ("mouth", lm.mouth),
                             ("brow_l", lm.brow_l), ("brow_r", lm.brow_r)) if v]
    found += sorted(lm.joints)
    print(f"  landmarks: {', '.join(found) if found else '(none)'}")
    for code in landmark.landmark_warnings(lm):
        print(f"  [landmark-warn] {code}")
    if overlay_path:
        written = landmark.render_overlay(stack, lm, overlay_path)
        print(f"  wrote landmark overlay -> {written}")


def _run_batch(root: str, out: str | None, *, live2d: bool) -> int:
    """Batch mode: convert every PSD / layer dir under ROOT and print an aggregate QA report."""
    from .batch import convert_batch, discover_inputs

    inputs = discover_inputs(root)
    if not inputs:
        print(f"error: no PSDs or layer dirs found under {root}", file=sys.stderr)
        return 2
    out_dir = Path(out) if out else Path.cwd() / "batch_out"
    print(f"converting {len(inputs)} input(s) -> {out_dir}")
    try:
        outcome = convert_batch(inputs, out_dir, live2d=live2d)
    except ImportError as exc:
        print(f"error: PSD inputs need the decompose extra (pip install psd-tools pillow): {exc}",
              file=sys.stderr)
        return 2
    print(outcome.format())
    ok = not outcome.errors and outcome.qa().pass_rate >= 1.0
    return 0 if ok else 1


def _run_qa(layer_dirs: list[str]) -> int:
    """QA mode: build rigs from the given layer dirs (or the built-in samples) and print a
    pass-rate report (lint + param-sweep + landmark sanity). Exit non-zero if below 100%."""
    from .core import decompose, landmark
    from .core.qa import BatchReport, evaluate
    from .pipeline import rig_from_stack

    cases: list[tuple[str, Path]] = []
    if layer_dirs:
        cases = [(Path(d).name, Path(d)) for d in layer_dirs]
    else:
        try:
            from .samples import make_sample_fullbody, make_sample_layers
        except ImportError:  # pragma: no cover - Pillow missing
            print("error: built-in QA set needs Pillow (pip install pillow)", file=sys.stderr)
            return 2
        import tempfile

        tmp = Path(tempfile.mkdtemp())
        cases = [("portrait", make_sample_layers(tmp / "portrait")),
                 ("fullbody", make_sample_fullbody(tmp / "fullbody"))]

    reports = []
    for label, path in cases:
        try:
            stack = decompose.from_layer_dir(path)
        except (FileNotFoundError, ValueError) as exc:
            print(f"error: {path}: {exc}", file=sys.stderr)
            return 2
        rig = rig_from_stack(stack, name=label)
        lw = landmark.landmark_warnings(landmark.extract_landmarks(stack))  # per-character checks
        reports.append(evaluate(rig, label, landmark_warnings=lw))

    report = BatchReport(items=reports)
    print(report.format())
    return 0 if report.pass_rate >= 1.0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
