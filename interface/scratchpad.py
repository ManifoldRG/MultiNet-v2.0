"""Model-written scratchpad memory: the ``ExperimentConfig.memory`` arms.

The harness is stateless chat: each query resends the task and the history,
and the model's reasoning from earlier steps is gone. The scratchpad arms give
the model one piece of text it carries forward itself.

Arms (``memory``):

* ``full_context`` (default): no scratchpad. The requests are byte-identical to
  the harness before the arms existed, and the run hash leaves the memory
  fields out (scripts/run_pipeline.py).
* ``scratchpad_window``: the scratchpad plus only the last K steps of history.
  K is ``context_n`` and the arm requires ``context_window="last_n"``: the
  per-step history lines are the full history's, under the header "Recent
  history (last K steps, oldest first):".
* ``full_context_scratchpad``: the scratchpad plus the full history
  (``context_window="full"``).

Both scratchpad arms require ``chat_history="stateless"``.

Prompt: one sentence after the Valid actions line of the system prompt
(``SCRATCHPAD_SYSTEM_LINE``), the current notes as a section of the current
turn right before the question, and the reply format opening with the
scratchpad block. Nothing else changes; no maze rules or hints.

Reply format::

    <scratchpad>
    ...notes...
    </scratchpad>
    FINAL_OUTPUT: <action>

The last complete ``<scratchpad>...</scratchpad>`` block (tags case-insensitive)
is the new scratchpad, stripped of surrounding whitespace. Every complete block
is removed before the action parser runs, so notes that mention an action (a
``FINAL_OUTPUT:`` line, "turn left") are never read as the answer, and an
action written before a long block still sits in the parser's last lines. The
text is the visible reply only: Claude's thinking blocks, OpenAI reasoning and
Kimi reasoning content are never searched, so the format is the same for all
three providers.

Carry-over (``scratchpad_status`` on the query record):

* ``updated``: a block and a valid action; the block replaces the notes.
* ``truncated``: as ``updated`` but over ``scratchpad_max_chars``; the first
  ``scratchpad_max_chars`` characters are kept and the next request says so.
* ``missing``: a valid action but no complete block (none, or an unclosed
  one); the previous notes stay. An empty block clears them.
* ``kept_parse_failure``: no valid action (including an infrastructure stub):
  the previous notes stay, whatever the reply wrote.

The cap is in characters, not tokens: it is the same for every provider's
tokenizer and the model can count it.

Caching: the notes change every step, so they sit in the current turn after
the last cache breakpoint, where the current state already is. The long prefix
(system prompt with the one-line addition, then every history step) is the same
from one request to the next, so the full_context_scratchpad arm reads it from
cache exactly as full_context does. The window arm's history slides every step,
so only its system prompt is cached.

Recording: each query record carries ``scratchpad_before`` (the notes the
request showed), ``scratchpad`` (the notes carried to the next request),
``scratchpad_status`` and ``scratchpad_written_chars`` (length of the block the
reply wrote, before any cut; None without one). They land in episode.json's
transcript; join steps to queries on ``query_index``. The notes in effect are
always read back from the transcript, so a checkpoint resume (which restores
the transcript) restores them.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Optional

from prompting_experiments.prompt_templates import memory as memory_templates
from prompting_experiments.prompt_templates import system as system_templates
from prompting_experiments.prompt_templates import user as user_templates

_OPEN_RE = re.compile(r"(?i)<scratchpad\s*>")
_CLOSE_RE = re.compile(r"(?i)</scratchpad\s*>")


@dataclass(frozen=True)
class ScratchpadReply:
    """A reply split into its scratchpad and the text the action parser reads."""

    notes: Optional[str]  # the last complete block, stripped; None without one
    action_text: str  # the reply with every complete block removed


def split_reply(text: str) -> ScratchpadReply:
    """Pair each closing tag with the nearest opening tag before it (and after
    the previous block); keep the last block's text, drop every block."""
    spans: list[tuple[int, int, int, int]] = []
    start = 0
    for close in _CLOSE_RE.finditer(text):
        opens = list(_OPEN_RE.finditer(text, start, close.start()))
        if not opens:
            continue
        spans.append((opens[-1].start(), opens[-1].end(), close.start(), close.end()))
        start = close.end()
    if not spans:
        return ScratchpadReply(notes=None, action_text=text)
    pieces: list[str] = []
    cursor = 0
    for block_start, _, _, block_end in spans:
        pieces.append(text[cursor:block_start])
        cursor = block_end
    pieces.append(text[cursor:])
    _, body_start, body_end, _ = spans[-1]
    return ScratchpadReply(
        notes=text[body_start:body_end].strip(), action_text="".join(pieces)
    )


def record_fields(
    previous: str, parsed: ScratchpadReply, *, parse_ok: bool, max_chars: int
) -> dict[str, Any]:
    """The scratchpad fields of one query record."""
    written = None if parsed.notes is None else len(parsed.notes)
    if not parse_ok:
        status, notes = "kept_parse_failure", previous
    elif parsed.notes is None:
        status, notes = "missing", previous
    elif len(parsed.notes) > max_chars:
        status, notes = "truncated", parsed.notes[:max_chars]
    else:
        status, notes = "updated", parsed.notes
    return {
        "scratchpad_before": previous,
        "scratchpad": notes,
        "scratchpad_status": status,
        "scratchpad_written_chars": written,
    }


def current(transcript: list[dict]) -> tuple[str, Optional[int]]:
    """(notes in effect, characters the model wrote when the latest query cut
    them to the cap, else None), read from the newest query record."""
    for rec in reversed(transcript):
        if rec.get("kind") != "query":
            continue
        if "scratchpad" not in rec:
            return "", None
        cut = rec.get("scratchpad_status") == "truncated"
        return str(rec["scratchpad"]), (rec.get("scratchpad_written_chars") if cut else None)
    return "", None


def section(notes: str, truncated_from: Optional[int], max_chars: int) -> str:
    """The current-turn text showing the notes (and the cut, if any)."""
    # The runner later expands this placeholder into the current image; notes
    # must never trigger that.
    notes = notes.replace(user_templates.CURRENT_IMAGE_PLACEHOLDER, "(current_image)")
    text = memory_templates.SCRATCHPAD_SECTION.format(notes=f"{notes}\n" if notes else "")
    if truncated_from is not None:
        text += "\n" + memory_templates.SCRATCHPAD_TRUNCATED_NOTE.format(
            chars=truncated_from, max_chars=max_chars
        )
    return text


def with_system_line(system_prompt: str, max_chars: int) -> str:
    """``system_prompt`` with the scratchpad sentence after the strategy's
    lines and before the initial maze, when the prompt carries one."""
    line = memory_templates.SCRATCHPAD_SYSTEM_LINE.format(max_chars=max_chars)
    head, maze, tail = system_prompt.partition("\n\n" + system_templates.INITIAL_MAZE_SECTION)
    return f"{head}\n{line}{maze}{tail}"


def with_output_format(instruction: str) -> str:
    """The querying mode's "Output exactly:" instruction, opening with the block."""
    lead = memory_templates.OUTPUT_FORMAT_LEAD
    if instruction.startswith(lead):
        return lead + memory_templates.SCRATCHPAD_OUTPUT_FORMAT + instruction[len(lead):]
    return memory_templates.SCRATCHPAD_OUTPUT_FORMAT + instruction


def insert_before_question(prompt_text: str, text: str, question: str) -> str:
    """``text`` as its own paragraph right before the current question."""
    before, match, after = prompt_text.rpartition(question)
    if not match:
        return f"{prompt_text}\n\n{text}"
    if before and not before.endswith("\n\n"):
        before += "\n" if before.endswith("\n") else "\n\n"
    return f"{before}{text}\n\n{match}{after}"
