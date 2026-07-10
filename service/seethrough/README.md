# See-through decompose service

A standalone GPU HTTP service that wraps [See-through](https://github.com/shitagaki-lab/see-through):
`POST /decompose` an image → get a layered `.psd`. The image2live2d client
(`decompose.from_service`) turns that into a rig. Full GCP setup (provisioning, GPU choice, cost vs the
$300 free credit, security) is in **[../../docs/DECOMPOSE_SERVICE.md](../../docs/DECOMPOSE_SERVICE.md)**.

## Files
- `app.py` — the FastAPI service (shells out to See-through's `inference_psd.py`)
- `startup.sh` — GCP VM bootstrap (installs See-through + venv, registers a systemd unit)
- `Dockerfile` — reproducible alternative (CUDA 12.8 base)
- `requirements-service.txt` — service-only deps (fastapi + uvicorn)

## Run locally on a GPU box (quick)
```bash
git clone https://github.com/shitagaki-lab/see-through /opt/see-through
cd /opt/see-through && ln -sf common/assets assets
python3.12 -m venv .venv && . .venv/bin/activate
pip install torch==2.8.0+cu128 --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt fastapi 'uvicorn[standard]'
SEE_THROUGH_DIR=/opt/see-through uvicorn app:app --host 0.0.0.0 --port 8000
```

## Endpoints
- `GET /health` → `{"ok": true, ...}`
- `POST /decompose` (raw image bytes; optional `X-Auth-Token`) → `.psd` bytes

Notes: single GPU ⇒ requests are serialized; first call downloads weights (~10–15 GB) and is slow.
Set `SEE_THROUGH_TOKEN` to require auth. Don't expose the port to `0.0.0.0/0` — restrict to your IP.
