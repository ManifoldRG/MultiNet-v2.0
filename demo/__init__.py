"""Interactive human-play demo for MiniGrid tasks.

Split into layers so UI redesigns never have to touch game logic, and vice
versa:

- ``demo.session``: ``MiniGridPlaySession`` -- task loading, stepping,
  transcript recording, and the model-parity text views. No pygame
  dependency, so it's independently testable/reusable.
- ``demo.ui``: ``MiniGridPlayerUI`` -- the pygame window, layout chrome,
  input handling, and wiring. Holds a session and reads/drives it; owns
  no game logic.
- ``demo.theme``: layout constants, palette, fonts, and shared color helpers.
- ``demo.icons``: keycap / glyph drawing primitives used by the rails.
- ``demo.overlays``: start splash and end-of-episode result cards (basic + R1).
- ``demo.sounds``: synthesized UI interaction feedback (soft ticks / latches /
  chimes), best-effort and silent when audio isn't available.
- ``demo.fx``: display-only motion feedback (wall bounce, cell flash/fade,
  goal pulse) composited on the rendered frame -- never mutates the env.
- ``demo.compare``: R1 results lookup (sibling checkout or nested
  Multinet-v2-results directory; degrades gracefully when neither is
  present). Any maze plays; the table only gates the comparison card.
- ``demo.r1_config`` / ``demo.r1_tasks``: shared R1 experiment config and
  allowlist helpers used by both the desktop UI and the web API.
- ``demo.api``: FastAPI HTTP surface for the web player
  (``uvicorn demo.api.app:app --reload --app-dir .``).

``play_task.py`` at the repo root is the thin CLI entry point that wires the
session and UI together.
"""
