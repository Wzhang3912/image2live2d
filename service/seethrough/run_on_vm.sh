#!/usr/bin/env bash
# Run the image2live2d orchestration client ON the GPU VM, talking to the See-through service at
# localhost:8000. This is the correct topology: the (large) image never traverses your home NAT to a
# raw port-8000 endpoint — the client and service are co-located, so the broken-pipe-on-upload issue
# can't happen. The client is light (pydantic + Pillow + psd-tools); the heavy GPU work is the service.
#
# Usage:  service/seethrough/run_on_vm.sh <image> [name] [zone] [instance] [project]
# Result: ./out/<name>.inp  (the VM is started before and stopped after).
set -euo pipefail

IMG="${1:?usage: run_on_vm.sh <image> [name] [zone] [instance] [project]}"
NAME="${2:-$(basename "${IMG%.*}")}"
ZONE="${3:-us-west4-a}"
INST="${4:-seethrough-svc}"
PROJ="${5:-image2live2d-77562}"
ROOT="$(git -C "$(dirname "$0")" rev-parse --show-toplevel)"
ext="${IMG##*.}"

g() { gcloud compute "$@" --project="$PROJ" --zone="$ZONE" --quiet; }

echo ">> starting $INST ($ZONE) ..."
gcloud compute instances start "$INST" --project="$PROJ" --zone="$ZONE" --quiet

echo ">> waiting for SSH ..."
for _ in $(seq 1 20); do g ssh "$INST" --command='true' >/dev/null 2>&1 && break; sleep 6; done

echo ">> uploading image + client source to the VM ..."
g scp "$IMG" "$INST:/tmp/i2l_input.$ext"
tar czf /tmp/i2l_src.tgz -C "$ROOT" src/image2live2d
g scp /tmp/i2l_src.tgz "$INST:/tmp/i2l_src.tgz"

echo ">> running the pipeline ON the VM (decompose via localhost) ..."
g ssh "$INST" --command="
set -e
rm -rf ~/i2l && mkdir -p ~/i2l && tar xzf /tmp/i2l_src.tgz -C ~/i2l
[ -d ~/i2l/venv ] || python3 -m venv ~/i2l/venv
~/i2l/venv/bin/pip -q install --upgrade pip >/dev/null
~/i2l/venv/bin/pip -q install pydantic pillow psd-tools >/dev/null
PYTHONPATH=~/i2l/src ~/i2l/venv/bin/python -m image2live2d \
  --image /tmp/i2l_input.$ext -n '$NAME' \
  --decompose-url http://localhost:8000 \
  --work-dir /tmp/${NAME}_layers -o /tmp/$NAME.inp
"

echo ">> fetching the result ..."
mkdir -p "$ROOT/out"
g scp "$INST:/tmp/$NAME.inp" "$ROOT/out/$NAME.inp"

echo ">> stopping $INST ..."
gcloud compute instances stop "$INST" --project="$PROJ" --zone="$ZONE" --quiet
echo ">> done: out/$NAME.inp"
