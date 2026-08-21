# MultiNet MiniGrid play API - the backend behind the playable maze on
# multinet.ai.
#
# Local (init the ogbench submodule first — it carries the R1 maze corpus):
#   git submodule update --init
#   docker build -t multinet-maze .
#   docker run -p 8080:8080 -e MULTINET_CORS_ORIGINS=http://127.0.0.1:8899 multinet-maze
#
# Cloud Run: see deploy/CLOUDRUN.md.

FROM python:3.11-slim

# minigrid pulls in pygame transitively (minigrid.core.world_object), so pygame
# is a hard dependency of the API even though pyproject lists it only under the
# optional "visual" extra. Its manylinux wheel bundles SDL, but SDL still tries
# to open a display and an audio device at import unless told there are none.
# Without these two the image builds clean and dies on first boot.
ENV SDL_VIDEODRIVER=dummy \
    SDL_AUDIODRIVER=dummy \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# Dependencies in their own layer so source edits don't rebuild the wheels.
# Pinned to the runtime set rather than `pip install .[web]`: the API needs
# pygame, which that extra does not carry.
RUN pip install --upgrade pip && \
    pip install \
      gymnasium \
      minigrid \
      numpy \
      pillow \
      pygame \
      fastapi \
      "uvicorn[standard]"

# The first-party modules the API actually loads. This list is not guesswork -
# it is what `sys.modules` contains after importing demo.api.app and calling
# list_r1_tasks(). The chain reaches further than it looks: demo -> interface
# -> interface.coords -> prompting_experiments, and interface pulls pipeline
# and scorer behind that.
#
# demo/data/ carries the vendored R1 results table; there is no sibling
# Multinet-v2-results checkout inside an image and R1ResultCatalog raises
# FileNotFoundError without it. The R1 manifest resolves all 42 playable tasks
# under ogbench/ogbench/procgen/maze_jsons/ (same corpus as local dev).
COPY demo/ ./demo/
COPY gridworld/ ./gridworld/
COPY interface/ ./interface/
COPY pipeline/ ./pipeline/
COPY prompting_experiments/ ./prompting_experiments/
COPY scorer/ ./scorer/
COPY scripts/ ./scripts/
COPY ogbench/ogbench/procgen/maze_jsons/ ./ogbench/ogbench/procgen/maze_jsons/
COPY model_interface.py ./

# Fail the build rather than the deploy. The task count is asserted, not just
# printed: load_manifest_tasks skips manifest rows whose `source` file is
# missing with a warning rather than an error, so a missing maze corpus would
# otherwise produce a container that boots happily and serves zero mazes.
RUN python -c "\
import demo.api.app; \
from demo.r1_tasks import list_r1_tasks; \
n = len(list_r1_tasks()); \
assert n == 42, 'expected 42 R1 tasks, got %d - check ogbench maze_jsons and the manifest' % n; \
print('import ok,', n, 'tasks')"

# Cloud Run injects PORT and ignores EXPOSE; 8080 is its default and a fine
# local one. One worker on purpose: GameRegistry holds live games in process
# memory, so a second worker would not recognise a game the first one started.
ENV PORT=8080
EXPOSE 8080
CMD exec uvicorn demo.api.app:app --host 0.0.0.0 --port ${PORT} --workers 1
