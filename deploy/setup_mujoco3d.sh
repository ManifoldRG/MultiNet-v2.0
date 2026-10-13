#!/usr/bin/env bash
# Make this VM able to draw 3D frames for a fleet run (next-run-fixes item 20).
#
# Run ON the VM from the repo root:  bash deploy/setup_mujoco3d.sh <venv-dir>
# launch_distributed.sh does this on every VM when the run-config's render
# block names mujoco3d. Installs the OSMesa GL library and the pyproject
# `mujoco3d` extra into <venv-dir>, then draws one frame and prints
# "mujoco3d-ok <mujoco version>". Every VM renders with OSMesa: EGL rasterises
# slightly differently, and the frames are what the model sees.
set -euo pipefail

venv="${1:?usage: setup_mujoco3d.sh <venv-dir>}"
# shellcheck disable=SC1090
source "$venv/bin/activate"
export MUJOCO_GL=osmesa

if ! ldconfig -p 2>/dev/null | grep -q libOSMesa; then
  sudo DEBIAN_FRONTEND=noninteractive apt-get update -qq
  sudo DEBIAN_FRONTEND=noninteractive apt-get install -y -qq libosmesa6
fi

# The extra's requirements, read from the synced pyproject (tomllib is 3.11+).
mapfile -t reqs < <(python - <<'PY'
import pathlib
import re

text = pathlib.Path("pyproject.toml").read_text()
try:
    import tomllib
    reqs = tomllib.loads(text)["project"]["optional-dependencies"]["mujoco3d"]
except ModuleNotFoundError:
    block = re.search(r"^mujoco3d\s*=\s*\[(.*?)\]", text, re.M | re.S).group(1)
    reqs = re.findall(r'"([^"]+)"', block)
print("\n".join(reqs))
PY
)
python -m pip install -q "${reqs[@]}"

python - <<'PY'
import mujoco

from gridworld.backends import get_backend
from gridworld.task_spec import TaskSpecification

spec = TaskSpecification.from_json("mazes/validation_10/V01_empty_room.json")
backend = get_backend("mujoco3d", camera="top_down", resolution=64)
backend.configure(spec)
frame, _state, _info = backend.reset(seed=0)
assert frame.shape == (64, 64, 3), frame.shape
backend.close()
print(f"mujoco3d-ok {mujoco.__version__}")
PY
