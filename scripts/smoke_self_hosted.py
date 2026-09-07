"""Exercise a running self-hosted instance using synthetic art, without a GPU."""
import io
import json
import sys
import tempfile
import time
import urllib.request
import zipfile
from pathlib import Path

from image2live2d.samples import make_sample_layers

base = sys.argv[1] if len(sys.argv) > 1 else 'http://127.0.0.1:8000'


def request(path, data=None):
    with urllib.request.urlopen(base + path, data=data, timeout=10) as response:
        return response.read()


assert json.loads(request('/healthz'))['ok']
assert json.loads(request('/api/setup'))['layers_ready']
with tempfile.TemporaryDirectory() as directory:
    layers = make_sample_layers(Path(directory))
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, 'w') as archive:
        for png in layers.glob('*.png'):
            archive.write(png, png.name)
job = json.loads(request('/api/jobs?name=smoke.zip', buf.getvalue()))['job_id']
deadline = time.monotonic() + 90
while True:
    result = json.loads(request('/api/jobs/' + job))
    if result['status'] != 'running':
        break
    if time.monotonic() > deadline:
        raise RuntimeError('Conversion timed out')
    time.sleep(0.5)
assert result['status'] == 'done', result
assert request('/api/jobs/' + job + '/download').startswith(b'TRNSRTS')
print('PASS: health, readiness, ZIP conversion and .inp download')
