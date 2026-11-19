"""The scratchpad on the wire, for each provider that caches.

The notes are treated like the inventory: they sit in the current turn, after
the cached prefix, so notes that change every step never re-write the cache,
and the full-history arm still reads back everything the previous request
wrote. The cap bounds the notes every request resends.

The agents are the R2 NL panel's, built the way a run builds them
(``_agent_model_config`` + ``_build_agent_from_spec`` on
run_config.mechanics_15_nl.json); only ``urllib.request.urlopen`` is stubbed,
so every assertion is on the JSON body the agent would POST.
"""

from __future__ import annotations

import json
import urllib.request
from pathlib import Path

import pytest

from gridworld.backends.minigrid_backend import MiniGridBackend
from interface import scratchpad as scratchpad_mod
from interface.agents.openai_agent import SPEND
from interface.config import ExperimentConfig
from interface.episode_log import flush_episode_log
from interface.runner import build_runner
from prompting_experiments.prompt_templates import memory as memory_templates
from scratchpad_test_utils import ARMS, IMAGE_CELL, NL_CELL, OVERFLOW_CAP, REPLIES, SCRATCHPAD_ARMS, notes
from scripts.run_pipeline import _agent_model_config, _build_agent_from_spec
from start_map_test_utils import r1_spec

_R2_NL = json.loads(
    (Path(__file__).resolve().parents[1] / "gridworld/fixtures/run_config.mechanics_15_nl.json").read_text()
)
# provider -> (model key in the R2 NL config, API key variable, explicit cache marker)
PROVIDERS = {
    "claude": ("claude_opus55", "ANTHROPIC_API_KEY", "cache_control"),
    "openai": ("gpt6_astra", "OPENAI_API_KEY", "prompt_cache_breakpoint"),
    "kimi": ("kimi_k3", "MOONSHOT_API_KEY", None),  # implicit prefix caching
}
CELLS = {"nl": NL_CELL, "2d": IMAGE_CELL}
_MARKERS = ("cache_control", "prompt_cache_breakpoint")


@pytest.fixture(autouse=True)
def _reset_openai_spend():
    SPEND.reset()
    yield
    SPEND.reset()


class _Response:
    def __init__(self, payload: dict):
        self._raw = json.dumps(payload).encode("utf-8")

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def read(self):
        return self._raw


def _completion(provider: str, text: str) -> dict:
    if provider == "claude":
        return {
            "content": [{"type": "text", "text": text}],
            "usage": {"input_tokens": 100, "output_tokens": 10},
            "stop_reason": "end_turn",
        }
    return {
        "choices": [{"message": {"content": text}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 100, "completion_tokens": 10, "total_tokens": 110},
    }


def _run(provider: str, arm: str, cell: dict, monkeypatch, tmp_path: Path):
    """One scripted episode through the provider's real agent; returns the
    request bodies it POSTed and the episode.json it wrote."""
    key, env, _ = PROVIDERS[provider]
    monkeypatch.setenv(env, "test-key")
    experiment = {**cell, **ARMS[arm]}
    agent, _ = _build_agent_from_spec(
        key, _agent_model_config(_R2_NL["models"][key], experiment)
    )
    replies = list(REPLIES)
    bodies: list[dict] = []

    def fake_urlopen(req, timeout=None):
        bodies.append(json.loads(req.data.decode("utf-8")))
        text = replies.pop(0) if replies else notes("z") + "FINAL_OUTPUT: MOVE_FORWARD"
        return _Response(_completion(provider, text))

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    spec = r1_spec(12)
    backend = MiniGridBackend(render_mode="rgb_array")
    backend.configure(spec)
    result = build_runner(ExperimentConfig(**experiment), backend, spec).run(agent, verbose=False)
    episode = json.loads(flush_episode_log(result, tmp_path / "ep").read_text())
    queries = [r for r in episode["transcript"] if r["kind"] == "query"]
    assert len(bodies) == len(queries) > len(REPLIES)
    return bodies, queries, episode


def _strip(block: dict) -> dict:
    return {k: v for k, v in block.items() if k not in _MARKERS}


def _segments(provider: str, body: dict) -> list[dict]:
    """The request as one list of blocks in the order the provider reads it."""
    out: list[dict] = []
    if provider == "claude":
        system = body["system"]
        out += system if isinstance(system, list) else [{"type": "text", "text": system}]
        contents = [turn["content"] for turn in body["messages"]]
    else:
        contents = [message["content"] for message in body["messages"]]
    for content in contents:
        out += content if isinstance(content, list) else [{"type": "text", "text": content}]
    return out


def _section(record: dict) -> str:
    """The scratchpad section the request for ``record`` showed."""
    return scratchpad_mod.section(record["scratchpad_before"], None, OVERFLOW_CAP)


def _common_prefix(a: str, b: str) -> int:
    n = 0
    for x, y in zip(a, b):
        if x != y:
            break
        n += 1
    return n


@pytest.mark.parametrize("cell", sorted(CELLS))
@pytest.mark.parametrize("arm", SCRATCHPAD_ARMS)
@pytest.mark.parametrize("provider", sorted(PROVIDERS))
def test_the_notes_sit_after_the_cached_prefix_like_the_inventory(
    provider, arm, cell, monkeypatch, tmp_path
):
    bodies, queries, _ = _run(provider, arm, CELLS[cell], monkeypatch, tmp_path)
    marker = PROVIDERS[provider][2]
    previous: dict | None = None
    for index, (body, record) in enumerate(zip(bodies, queries)):
        section = _section(record)
        segments = _segments(provider, body)
        texts = [s.get("text", "") for s in segments]
        # Once, in the request's last block, after the current inventory line:
        # the place of the current state, which changes every step anyway.
        assert [i for i, t in enumerate(texts) if section in t] == [len(segments) - 1], index
        last = texts[-1]
        assert last.rindex("Your inventory:") < last.index(section)
        if marker is not None:
            cached = [i for i, s in enumerate(segments) if marker in s]
            assert cached and len(cached) <= 4
            # Every breakpoint precedes the notes: no cached span holds them.
            assert cached[-1] < len(segments) - 1
            assert "scratchpad" not in json.dumps(segments[: cached[-1] + 1]).lower().replace(
                memory_templates.SCRATCHPAD_SYSTEM_LINE.format(max_chars=OVERFLOW_CAP).lower(), ""
            )
            if arm == "full_context_scratchpad" and previous is not None:
                # This request reads back the whole prefix the previous one wrote.
                before = _segments(provider, previous)
                end = max(i for i, s in enumerate(before) if marker in s)
                assert [_strip(s) for s in segments[: end + 1]] == [
                    _strip(s) for s in before[: end + 1]
                ], index
        elif previous is not None:
            # Implicit prefix caching reuses what this request shares with the
            # previous one; the whole system prompt is in it, and when the
            # notes changed they start after it (they never cut it short).
            now, then = json.dumps(body["messages"]), json.dumps(previous["messages"])
            shared = _common_prefix(now, then)
            assert shared > len(json.dumps(body["messages"][0]))
            if record["scratchpad_before"] != queries[index - 1]["scratchpad_before"]:
                assert now.index(json.dumps(section)[1:-1]) > shared, index
        previous = body


@pytest.mark.parametrize("cell", sorted(CELLS))
@pytest.mark.parametrize("provider", ["claude", "openai"])
def test_full_context_scratchpad_caches_what_full_context_caches(
    provider, cell, monkeypatch, tmp_path
):
    """The same breakpoints on the same history blocks, so the notes cost no
    cache: only the system prompt differs (by the one scratchpad sentence)."""
    plain, _, _ = _run(provider, "full_context", CELLS[cell], monkeypatch, tmp_path / "plain")
    arm, _, _ = _run(provider, "full_context_scratchpad", CELLS[cell], monkeypatch, tmp_path / "arm")
    marker = PROVIDERS[provider][2]
    line = memory_templates.SCRATCHPAD_SYSTEM_LINE.format(max_chars=OVERFLOW_CAP)
    # Reply 8 parses only in the scratchpad arm; the episodes match before it.
    for index in range(1, 8):
        got, want = _segments(provider, arm[index]), _segments(provider, plain[index])
        cached = [i for i, s in enumerate(got) if marker in s]
        assert cached == [i for i, s in enumerate(want) if marker in s], index
        assert len(cached) >= 2  # the system prompt and the newest history step
        assert got[1 : cached[-1] + 1] == want[1 : cached[-1] + 1], index
        assert got[0]["text"].replace(f"\n{line}", "", 1) == want[0]["text"]


@pytest.mark.parametrize("provider", sorted(PROVIDERS))
def test_the_cap_cuts_the_notes_on_the_wire_and_the_episode_reports_it(
    provider, monkeypatch, tmp_path
):
    """Reply 5 writes 125 characters over an 80-character cap."""
    bodies, queries, episode = _run(
        provider, "full_context_scratchpad", NL_CELL, monkeypatch, tmp_path
    )
    written = "Gate should be open now. " + "x" * 100
    record = queries[4]
    # Reported in episode.json: status, what the reply wrote, what is carried.
    assert record["scratchpad_status"] == "truncated"
    assert record["scratchpad_written_chars"] == len(written) == 125
    assert record["scratchpad"] == written[:OVERFLOW_CAP]
    assert episode["config"]["scratchpad_max_chars"] == OVERFLOW_CAP
    # On the wire: the next request carries the first 80 characters and says
    # so; the system prompt states the cap on every request.
    sent = json.dumps(bodies[5])
    assert json.dumps(written[:OVERFLOW_CAP])[1:-1] in sent
    assert written[: OVERFLOW_CAP + 1] not in sent
    told = memory_templates.SCRATCHPAD_TRUNCATED_NOTE.format(chars=125, max_chars=OVERFLOW_CAP)
    assert [told in json.dumps(b) for b in bodies] == [i == 5 for i in range(len(bodies))]
    line = memory_templates.SCRATCHPAD_SYSTEM_LINE.format(max_chars=OVERFLOW_CAP)
    assert all(line in json.dumps(_segments(provider, b)[0]) for b in bodies)
    # No request ever resends more notes than the cap.
    assert all(len(r["scratchpad_before"]) <= OVERFLOW_CAP for r in queries)
