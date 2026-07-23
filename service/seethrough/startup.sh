#!/usr/bin/env bash
# VM bootstrap for the See-through decompose service.
# Designed for a GCP Deep Learning VM image (NVIDIA driver + Python already present). PyTorch is
# installed with its own bundled CUDA 12.8 runtime, so the host only needs a recent driver (550+),
# which the DLVM images have — no system CUDA toolkit needed.
#
# Idempotent-ish: safe to re-run. Logs to /var/log/seethrough-startup.log.
set -euxo pipefail
exec > >(tee -a /var/log/seethrough-startup.log) 2>&1

SEE_THROUGH_DIR=/opt/see-through
VENV=/opt/seethrough-venv
PORT="${SEE_THROUGH_PORT:-8000}"

# Optional config from instance metadata (set via `gcloud ... --metadata=resolution=1024,seethrough-token=secret`)
meta() { curl -fs -H "Metadata-Flavor: Google" \
  "http://metadata.google.internal/computeMetadata/v1/instance/attributes/$1" 2>/dev/null || true; }
# NOTE: -f is essential — without it, a missing metadata key returns the 404 HTML page (not empty),
# which would get injected into the systemd unit and crash the service (int('<!DOCTYPE') etc.).
RESOLUTION="$(meta resolution)"
SEE_THROUGH_TOKEN="$(meta seethrough-token)"

apt-get update
# libgl1 + libglib2.0-0: OpenCV (cv2, a See-through dep) needs them or it fails with
# "ImportError: libGL.so.1: cannot open shared object file" on headless servers.
apt-get install -y git python3.12 python3.12-venv libgl1 libglib2.0-0

# 1. See-through source, PINNED to a known commit. See-through's `main` is a moving target (its
# requirements + HF model weights float), so an unpinned clone made our decompose non-reproducible —
# the same input gave different layer sets across runs. Pin the code so our vendored patches apply
# against a known shape and behaviour is stable. Override with `--metadata=seethrough-commit=<sha>`.
SEE_THROUGH_COMMIT="$(meta seethrough-commit)"; SEE_THROUGH_COMMIT="${SEE_THROUGH_COMMIT:-58a1cb11d13f85acec9bbddb8cd4b6487843d4cf}"
if [ ! -d "$SEE_THROUGH_DIR" ]; then
  git clone https://github.com/shitagaki-lab/see-through.git "$SEE_THROUGH_DIR"
fi
cd "$SEE_THROUGH_DIR"
git fetch --quiet origin "$SEE_THROUGH_COMMIT" 2>/dev/null || git fetch --quiet origin || true
git checkout --quiet "$SEE_THROUGH_COMMIT" || echo "WARNING: could not pin See-through to $SEE_THROUGH_COMMIT — using default HEAD"
ln -sf common/assets assets || true

# 1b. Vendored robustness patches (image2live2d). See-through crashes the WHOLE decompose on inputs it
# doesn't fully segment:
#   - guard_empty_head_crop: the 'head' body segment comes out empty on some non-human faces (dragon
#     mascots) -> degenerate head crop -> cv2.resize on a zero-size image. Skips head sub-part refinement.
#   - guard_missing_part_pngs: load_parts does a bare Image.open on every tag from info.json; a tag whose
#     PNG is absent (head parts skipped above, OR a tag that came out empty this run) -> FileNotFoundError.
#     Skips absent tags. Together they turn a hard crash into graceful degradation (a coarser rig).
# Fetched from our public repo; each is non-fatal (|| echo) so an upstream shape change just leaves us
# unpatched rather than aborting boot. PATCH_REF selects the branch (default main).
PATCH_REF="$(meta patch-ref)"; PATCH_REF="${PATCH_REF:-main}"
PATCH_BASE="https://raw.githubusercontent.com/Wzhang3912/image2live2d/${PATCH_REF}/service/seethrough/patches"
apply_patch() {  # <patcher-file> <target-relpath> <label>
  if curl -fsSL "$PATCH_BASE/$1" -o "/tmp/$1"; then
    python3 "/tmp/$1" "$SEE_THROUGH_DIR/$2" || echo "WARNING: $3 not applied (anchor not found?) — serving unpatched"
  else
    echo "WARNING: could not fetch $1 from $PATCH_BASE — serving unpatched"
  fi
}
apply_patch guard_empty_head_crop.py  common/utils/inference_utils.py  "empty-head-crop guard"
apply_patch guard_missing_part_pngs.py common/utils/io_utils.py         "missing-tag-PNG guard"

# 2. Python env + deps (base PSD path only; skip detectron2/mmdet/SAM2 install-hell tiers)
python3.12 -m venv "$VENV"
# shellcheck disable=SC1091
source "$VENV/bin/activate"
pip install --upgrade pip
pip install torch==2.8.0+cu128 torchvision==0.23.0+cu128 torchaudio==2.8.0+cu128 \
  --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
pip install fastapi "uvicorn[standard]"

# 3. Drop the service app (uploaded separately via `gcloud compute scp`, or fetched here)
#    Expecting /opt/app.py to exist (scp it before/after boot). If absent, warn and exit.
if [ ! -f /opt/app.py ]; then
  echo "WARNING: /opt/app.py not found — scp service/seethrough/app.py to the VM at /opt/app.py"
fi

# 4. systemd unit so the service survives reboots
cat >/etc/systemd/system/seethrough.service <<UNIT
[Unit]
Description=see-through decompose service
After=network-online.target

[Service]
WorkingDirectory=/opt
Environment=SEE_THROUGH_DIR=$SEE_THROUGH_DIR
Environment=SEE_THROUGH_PYTHON=$VENV/bin/python
Environment=SEE_THROUGH_TIMEOUT=1500
${RESOLUTION:+Environment=RESOLUTION=$RESOLUTION}
${SEE_THROUGH_TOKEN:+Environment=SEE_THROUGH_TOKEN=$SEE_THROUGH_TOKEN}
ExecStart=$VENV/bin/uvicorn app:app --host 0.0.0.0 --port $PORT
Restart=on-failure

[Install]
WantedBy=multi-user.target
UNIT

systemctl daemon-reload
systemctl enable seethrough.service
# Start only if the app is present; otherwise scp it then `systemctl start seethrough`.
[ -f /opt/app.py ] && systemctl restart seethrough.service || true
echo "startup complete. First /decompose call downloads weights (~10-15GB) and is slow."
