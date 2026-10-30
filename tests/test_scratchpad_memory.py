"""The scratchpad memory arms with a scripted agent: what each request shows,
how a reply's notes carry over, and that the episode machinery around them
(recording, resume, caching, stall watchdog, step cap, usage, lockstep) is
unchanged."""

from __future__ import annotations

import json

import pytest

from gridworld.backends.minigrid_backend import MiniGridBackend
from interface.agents.claude import _resolve_cache_markers, _to_anthropic_turns
from interface.agents.kimi_k26 import _to_openai_messages as kimi_messages
from interface.agents.openai_agent import _to_openai_messages as openai_messages
from interface.agents.reply import Reply
from interface.batch_runner import LockstepBatchRunner, LockstepUnit
from interface.config import ExperimentConfig
from interface.episode_checkpoint import resume_stepper, save_checkpoint
from interface.episode_log import flush_episode_log
from interface.episode_step import EpisodeStepper
from interface.runner import build_runner
from interface.scratchpad import (
    ScratchpadReply,
    current,
    record_fields,
    split_reply,
    with_system_line,
)
from prompting_experiments.prompt_templates import memory as memory_templates
from prompting_experiments.prompt_templates import system as system_templates
from scratchpad_test_utils import (
    ARMS,
    IMAGE_CELL,
    NL_CELL,
    OVERFLOW_CAP,
    REPLIES,
    SCRATCHPAD_ARMS,
    drive,
    notes,
)
from start_map_test_utils import r1_spec

QUESTION = "What is your next action?"
SYSTEM_LINE = memory_templates.SCRATCHPAD_SYSTEM_LINE.format(max_chars=OVERFLOW_CAP)
OUTPUT_FORMAT = (
    "Output exactly:\n<scratchpad>\n<your notes>\n</scratchpad>\nFINAL_OUTPUT: <action>"
)



def _runner(arm: str = "full_context_scratchpad", cell: dict = NL_CELL, max_steps: int = 12, **extra):
    spec = r1_spec(max_steps)
    backend = MiniGridBackend(render_mode="rgb_array")
    backend.configure(spec)
    return build_runner(ExperimentConfig(**{**cell, **ARMS[arm], **extra}), backend, spec)


def _system(messages: list[dict]) -> str:
    return next(m["content"] for m in messages if m["role"] == "system")


def _user_text(messages: list[dict]) -> str:
    content = next(m["content"] for m in messages if m["role"] == "user")
    if isinstance(content, str):
        return content
    return "".join(b.get("text", "") for b in content if b["type"] == "text")


def _queries(stepper) -> list[dict]:
    return [rec for rec in stepper.transcript if rec["kind"] == "query"]


def _shown(text: str) -> str:
    """The notes a request shows."""
    head = "Your scratchpad:\n<scratchpad>\n"
    start = text.index(head) + len(head)
    return text[start : text.index("</scratchpad>", start)].removesuffix("\n")


# --------------------------------------------------------------------------- #
# Reply parsing
# --------------------------------------------------------------------------- #
def test_split_reply_takes_the_last_block_and_hides_every_block():
    reply = (
        "<scratchpad>old</scratchpad>\n"
        "I will update my <scratchpad> now.\n"
        "<SCRATCHPAD >\n  new notes\n  line 2\n</Scratchpad>\n"
        "FINAL_OUTPUT: MOVE_FORWARD"
    )
    parsed = split_reply(reply)
    # The prose tag opens nothing: each closing tag pairs with the nearest
    # opening tag before it.
    assert parsed.notes == "new notes\n  line 2"
    assert parsed.action_text == (
        "\nI will update my <scratchpad> now.\n\nFINAL_OUTPUT: MOVE_FORWARD"
    )


@pytest.mark.parametrize(
    "reply, expected",
    [
        ("FINAL_OUTPUT: TOGGLE", None),
        ("<scratchpad>\nunclosed\nFINAL_OUTPUT: TOGGLE", None),
        ("stray </scratchpad>\nFINAL_OUTPUT: TOGGLE", None),
        ("<scratchpad></scratchpad>\nFINAL_OUTPUT: TOGGLE", ""),
        ("<scratchpad>\n\n  a  \n\n</scratchpad>FINAL_OUTPUT: TOGGLE", "a"),
    ],
)
def test_split_reply_edge_cases(reply, expected):
    parsed = split_reply(reply)
    assert parsed.notes == expected
    if expected is None:
        assert parsed.action_text == reply


@pytest.mark.parametrize(
    "parsed, parse_ok, status, carried",
    [
        (ScratchpadReply("new", ""), True, "updated", "new"),
        (ScratchpadReply("", ""), True, "updated", ""),
        (ScratchpadReply("x" * 12, ""), True, "truncated", "x" * 10),
        (ScratchpadReply(None, ""), True, "missing", "prev"),
        (ScratchpadReply("new", ""), False, "kept_parse_failure", "prev"),
        (ScratchpadReply(None, ""), False, "kept_parse_failure", "prev"),
    ],
)
def test_record_fields(parsed, parse_ok, status, carried):
    fields = record_fields("prev", parsed, parse_ok=parse_ok, max_chars=10)
    assert fields == {
        "scratchpad_before": "prev",
        "scratchpad": carried,
        "scratchpad_status": status,
        "scratchpad_written_chars": None if parsed.notes is None else len(parsed.notes),
    }


def test_current_reads_the_newest_query_record():
    assert current([]) == ("", None)
    assert current([{"kind": "reset"}, {"kind": "query"}]) == ("", None)
    transcript = [
        {"kind": "query", "scratchpad": "a", "scratchpad_status": "updated"},
        {"kind": "step"},
        {"kind": "query", "scratchpad": "b" * 10, "scratchpad_status": "truncated",
         "scratchpad_written_chars": 12},
        {"kind": "step"},
    ]
    assert current(transcript) == ("b" * 10, 12)
    transcript.append(
        {"kind": "query", "scratchpad": "b" * 10, "scratchpad_status": "missing",
         "scratchpad_written_chars": None}
    )
    assert current(transcript) == ("b" * 10, None)


# --------------------------------------------------------------------------- #
# Prompts per arm
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("arm", SCRATCHPAD_ARMS)
@pytest.mark.parametrize("cell", [NL_CELL, IMAGE_CELL], ids=["nl", "2d"])
def test_scratchpad_arms_add_one_system_line_a_section_and_the_reply_format(arm, cell):
    _, base = drive(_runner("full_context", cell))
    _, requests = drive(_runner(arm, cell))
    for plain, messages in zip(base, requests):
        system = _system(messages)
        # One sentence after the Valid actions line, before any initial maze.
        assert system == _system(plain).replace(
            "DROP, TOGGLE, DONE.", "DROP, TOGGLE, DONE.\n" + SYSTEM_LINE, 1
        )
        text = _user_text(messages)
        assert text.count("Your scratchpad:") == 1
        assert text.endswith(f"{QUESTION}\n\n{OUTPUT_FORMAT}")
        # The notes sit right before the question (after any parse-failure line).
        section_end = text.index("</scratchpad>")
        assert section_end < text.rindex(QUESTION)
    assert _shown(_user_text(requests[0])) == ""


def test_full_context_prompts_carry_none_of_it():
    _, requests = drive(_runner("full_context"))
    for messages in requests:
        assert "scratchpad" not in json.dumps(messages).lower()


def test_the_system_line_precedes_an_inlined_initial_maze():
    prompt = "Task.\nValid actions: A.\n\n" + system_templates.INITIAL_MAZE_SECTION
    out = with_system_line(prompt, 50)
    line = memory_templates.SCRATCHPAD_SYSTEM_LINE.format(max_chars=50)
    assert out == f"Task.\nValid actions: A.\n{line}\n\n{system_templates.INITIAL_MAZE_SECTION}"
    runner = _runner(prompting="text_initial_maze")
    system, _ = runner.build_prompt_message(runner.backend.reset(seed=0)[1], "", [])
    assert system.index(SYSTEM_LINE) < system.index("Initial maze (fixed for this episode):")


def test_window_arm_shows_only_the_last_k_steps_in_the_full_history_format():
    _, full = drive(_runner("full_context_scratchpad"))
    _, window = drive(_runner("scratchpad_window"))
    last = _user_text(window[-1])
    assert last.startswith("Recent history (last 3 steps, oldest first):\n")
    assert last.count("Position before:") == 3
    full_text = _user_text(full[-1])
    assert full_text.count("Position before:") > 3
    # The same per-step lines as the full history, minus the older ones.
    steps = full_text.split("\n\nYou are at")[0].split("\nPosition before:")[1:]
    window_steps = last.split("\n\nYou are at")[0].split("\nPosition before:")[1:]
    assert window_steps == steps[-3:]
    # Before K steps exist, the window shows them all.
    assert _user_text(window[2]).count("Position before:") == 2


# --------------------------------------------------------------------------- #
# Carry-over, overflow, parse failure
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("arm", SCRATCHPAD_ARMS)
def test_notes_carry_over_by_the_rules(arm):
    stepper, requests = drive(_runner(arm))
    records = _queries(stepper)
    assert [r["scratchpad_status"] for r in records] == [
        "updated",             # 1
        "updated",             # 2
        "kept_parse_failure",  # 3: notes but no action
        "missing",             # 4: action but no notes
        "truncated",           # 5: over the cap
        "updated",             # 6: empty block clears
        "updated",             # 7
        "updated",             # 8: action before the block
        "updated",             # 9
        "updated",             # 10
    ]
    for record, request in zip(records, requests):
        # Every request shows the notes the previous record carried.
        assert _shown(_user_text(request)) == record["scratchpad_before"]
    for previous, record in zip(records, records[1:]):
        assert record["scratchpad_before"] == previous["scratchpad"]
    kept = records[1]["scratchpad"]
    assert kept == "FINAL_OUTPUT: TURN_LEFT was a bad idea; do not turn left yet."
    assert records[2]["scratchpad"] == kept and records[2]["scratchpad_written_chars"] == 9
    assert records[3]["scratchpad"] == kept and records[3]["scratchpad_written_chars"] is None
    assert records[5]["scratchpad"] == ""
    assert _shown(_user_text(requests[6])) == ""


@pytest.mark.parametrize("arm", SCRATCHPAD_ARMS)
def test_overflow_is_cut_and_the_model_is_told_once(arm):
    stepper, requests = drive(_runner(arm))
    record = _queries(stepper)[4]
    written = "Gate should be open now. " + "x" * 100
    assert record["scratchpad_written_chars"] == len(written)
    assert record["scratchpad"] == written[:OVERFLOW_CAP]
    note = memory_templates.SCRATCHPAD_TRUNCATED_NOTE.format(
        chars=len(written), max_chars=OVERFLOW_CAP
    )
    texts = [_user_text(r) for r in requests]
    assert [note in t for t in texts] == [i == 5 for i in range(len(texts))]
    assert f"{written[:OVERFLOW_CAP]}\n</scratchpad>\n{note}\n\n{QUESTION}" in texts[5]


@pytest.mark.parametrize("arm", SCRATCHPAD_ARMS)
def test_parse_failure_keeps_the_notes_and_feedback_follows_them(arm):
    stepper, requests = drive(_runner(arm))
    records = _queries(stepper)
    assert records[2]["parse_ok"] is False
    retry = _user_text(requests[3])
    assert retry.index("</scratchpad>") < retry.index("Could not parse FINAL_OUTPUT") < retry.index(QUESTION)
    # The parse failure costs a query, not a step: the retry sees the same state.
    assert stepper.transcript.index(records[3]) - stepper.transcript.index(records[2]) == 1


def test_notes_never_reach_the_action_parser():
    stepper, _ = drive(_runner("full_context_scratchpad"))
    parsed = [r["parsed_actions"] for r in _queries(stepper)]
    assert parsed[1] == ["MOVE_FORWARD"]  # notes held FINAL_OUTPUT: TURN_LEFT
    assert parsed[7] == ["TURN_RIGHT"]  # action before a six-line block
    # Without the scratchpad arm the same reply 8 is not parsed (today's parser
    # looks for FINAL_OUTPUT in the last five lines), so the arms differ there.
    base, _ = drive(_runner("full_context"))
    assert [r["parsed_actions"] for r in _queries(base)][7] == []


def test_subgoal_querying_reads_sub_goal_outside_the_notes():
    replies = [
        notes("SUB_GOAL: wrong") + "SUB_GOAL: reach the switch\nFINAL_OUTPUT: MOVE_FORWARD, MOVE_FORWARD",
        notes("x") + "SUB_GOAL: toggle\nFINAL_OUTPUT: TOGGLE",
        notes("y") + "SUB_GOAL: on\nFINAL_OUTPUT: MOVE_FORWARD",
    ]
    stepper, requests = drive(_runner(querying="subgoal"), replies)
    assert _user_text(requests[0]).endswith(
        "Output exactly:\n<scratchpad>\n<your notes>\n</scratchpad>\nSUB_GOAL:"
        " <short description of your next sub_goal>\nFINAL_OUTPUT:  <a>, <b>, ...  "
        "(comma-separated; one or more valid actions to reach it)"
    )
    steps = [r for r in stepper.transcript if r["kind"] == "step"]
    assert [s["subgoal"] for s in steps] == ["reach the switch"] * 2 + ["toggle"]


# --------------------------------------------------------------------------- #
# Recording, resume, caching
# --------------------------------------------------------------------------- #
class _Scripted:
    """Legacy ``__call__`` agent with usage, like the existing test doubles."""

    def __init__(self, replies):
        self._replies = iter(replies)
        self.last_usage = None

    def __call__(self, messages):
        self.last_usage = {"input_tokens": 100, "output_tokens": 7, "total_tokens": 107}
        return next(self._replies)


@pytest.mark.parametrize("arm", SCRATCHPAD_ARMS)
def test_episode_json_records_the_scratchpad_per_query(arm, tmp_path):
    result = _runner(arm).run(_Scripted(REPLIES + [notes("z") + "FINAL_OUTPUT: MOVE_FORWARD"] * 5), verbose=False)
    path = flush_episode_log(result, tmp_path / "ep")
    episode = json.loads(path.read_text())
    assert episode["config"]["memory"] == arm
    assert episode["config"]["scratchpad_max_chars"] == OVERFLOW_CAP
    queries = [r for r in episode["transcript"] if r["kind"] == "query"]
    assert len(queries) == episode["query_count"]
    for record in queries:
        assert set(record) >= {
            "scratchpad_before", "scratchpad", "scratchpad_status", "scratchpad_written_chars"
        }
        # Cost accounting reads the same usage block as before.
        assert record["usage"]["output_tokens"] == 7
        query_json = json.loads(
            (tmp_path / "ep" / "queries" / f"query_{record['query_index']:03d}" / "query.json").read_text()
        )
        assert query_json["scratchpad"] == record["scratchpad"]
    assert queries[0]["scratchpad"] == "Start. Plan: follow the corridor to the switch."
    # Steps join their query on query_index.
    steps = [r for r in episode["transcript"] if r["kind"] == "step"]
    assert {s["query_index"] for s in steps} <= {q["query_index"] for q in queries}


def test_full_context_records_carry_no_scratchpad_fields():
    stepper, _ = drive(_runner("full_context"))
    assert not any(key.startswith("scratchpad") for r in _queries(stepper) for key in r)


@pytest.mark.parametrize("arm", SCRATCHPAD_ARMS)
@pytest.mark.parametrize("at", [4, 5, 6])
def test_checkpoint_resume_restores_the_notes(arm, at, tmp_path):
    _, uninterrupted = drive(_runner(arm))
    stepper = EpisodeStepper(_runner(arm), verbose=False)
    stepper.start()
    for index, reply in enumerate(REPLIES):
        messages = stepper.next_query()
        if index == at:
            save_checkpoint(tmp_path / "ckpt.json", stepper)
            break
        stepper.apply_reply(Reply(text=reply))
    resumed = resume_stepper(tmp_path / "ckpt.json", runner=_runner(arm))
    assert resumed.next_query() == messages == uninterrupted[at]
    for index, reply in enumerate(REPLIES[at:], start=at):
        if index > at:
            assert resumed.next_query() == uninterrupted[index]
        resumed.apply_reply(Reply(text=reply))


def _anthropic_blocks(messages):
    system, turns = _to_anthropic_turns(messages)
    system, turns = _resolve_cache_markers(system, turns, ttl="5m", enabled=True)
    return system, turns[-1]["content"]


@pytest.mark.parametrize("cell", [NL_CELL, IMAGE_CELL], ids=["nl", "2d"])
def test_full_context_scratchpad_keeps_the_cached_prefix_of_full_context(cell):
    _, base = drive(_runner("full_context", cell))
    _, arm = drive(_runner("full_context_scratchpad", cell))
    # Reply 8 parses only in the scratchpad arm, so the two episodes walk the
    # same steps up to request 8.
    for plain, messages in zip(base[1:8], arm[1:8]):
        _, blocks = _anthropic_blocks(messages)
        _, plain_blocks = _anthropic_blocks(plain)
        marked = [i for i, b in enumerate(blocks) if "cache_control" in b]
        # The same breakpoints on the same history blocks as full_context.
        assert marked == [i for i, b in enumerate(plain_blocks) if "cache_control" in b]
        assert blocks[: marked[-1] + 1] == plain_blocks[: marked[-1] + 1]
    previous = None
    for messages in arm[1:]:
        system, blocks = _anthropic_blocks(messages)
        assert "cache_control" in system[0]
        marked = [i for i, b in enumerate(blocks) if "cache_control" in b]
        assert 1 + len(marked) <= 4
        # The notes only in the uncached current turn.
        assert all("Your scratchpad:" not in b.get("text", "") for b in blocks[: marked[-1] + 1])
        assert "Your scratchpad:" in blocks[-1]["text"]
        if previous is not None:
            end = max(i for i, b in enumerate(previous) if "cache_control" in b)
            strip = lambda b: {k: v for k, v in b.items() if k != "cache_control"}  # noqa: E731
            assert [strip(b) for b in blocks[: end + 1]] == [strip(b) for b in previous[: end + 1]]
        previous = blocks


@pytest.mark.parametrize("arm", SCRATCHPAD_ARMS)
def test_openai_and_kimi_requests(arm):
    _, requests = drive(_runner(arm))
    for messages in requests:
        out = openai_messages(messages, "high", True)
        blocks = next(m["content"] for m in out if m["role"] == "user")
        if isinstance(blocks, list):
            marked = [i for i, b in enumerate(blocks) if "prompt_cache_breakpoint" in b]
            assert len(marked) + 1 <= 4  # plus the system message
            assert "Your scratchpad:" in blocks[-1]["text"] and len(blocks) - 1 not in marked
        kimi_user = next(m["content"] for m in kimi_messages(messages) if m["role"] == "user")
        assert kimi_user == _user_text(messages)  # one string, the text as sent


# --------------------------------------------------------------------------- #
# Episode machinery
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("arm", SCRATCHPAD_ARMS)
def test_stall_watchdog_still_ends_a_looping_episode(arm):
    runner = _runner(arm, max_steps=40, progress_stall_k=4)
    result = runner.run(_Scripted([notes("spin") + "FINAL_OUTPUT: TURN_LEFT"] * 40), verbose=False)
    assert result["end_reason"] == "stalled"
    assert result["steps_used"] == 4


@pytest.mark.parametrize("arm", SCRATCHPAD_ARMS)
def test_step_cap_still_ends_the_episode(arm):
    base = _runner("full_context", max_steps=5).run(_Scripted(REPLIES), verbose=False)
    result = _runner(arm, max_steps=5).run(_Scripted(REPLIES), verbose=False)
    # MiniGrid truncates at the cap; the episode ends as full_context's does.
    assert result["end_reason"] == base["end_reason"] == "truncated"
    assert result["steps_used"] == base["steps_used"] == 5


@pytest.mark.parametrize("arm", SCRATCHPAD_ARMS)
def test_parse_retry_cap_still_ends_the_episode(arm):
    runner = _runner(arm, max_parse_retries=2)
    result = runner.run(_Scripted([notes("thinking") + "no action"] * 5), verbose=False)
    assert result["end_reason"] == "parse_failed"
    assert result["query_count"] == 2


def test_infrastructure_stub_keeps_the_notes_and_costs_no_retry():
    stepper = EpisodeStepper(_runner(), verbose=False)
    stepper.start()
    stepper.next_query()
    stepper.apply_reply(Reply(text=notes("keep me") + "FINAL_OUTPUT: MOVE_FORWARD"))
    stepper.next_query()
    stepper.apply_reply(Reply(text="", stop_reason="batch_expired"))
    record = _queries(stepper)[-1]
    assert record["scratchpad_status"] == "kept_parse_failure"
    assert record["scratchpad"] == "keep me"
    assert stepper.parse_failures == 0
    assert _shown(_user_text(stepper.next_query())) == "keep me"


class _BatchAgent:
    """Batch double: the scripted replies in order, then plain moves."""

    def __init__(self, replies):
        self._replies = list(replies)

    def generate_batch(self, batch):
        out = []
        for _ in batch:
            text = self._replies.pop(0) if self._replies else notes("z") + "FINAL_OUTPUT: MOVE_FORWARD"
            out.append(Reply(text=text))
        return out


@pytest.mark.parametrize("arm", SCRATCHPAD_ARMS)
def test_lockstep_runner_matches_the_serial_runner(arm, tmp_path):
    serial, _ = drive(_runner(arm))
    batch = LockstepBatchRunner(_BatchAgent(REPLIES), max_batches=1)
    batch.add(LockstepUnit("u", EpisodeStepper(_runner(arm)), checkpoint_path=tmp_path / "u.ckpt.json"))
    result = batch.run()["u"]
    keys = ("scratchpad_before", "scratchpad", "scratchpad_status", "scratchpad_written_chars")
    lockstep_records = [r for r in result["transcript"] if r["kind"] == "query"]
    assert len(lockstep_records) > len(REPLIES)
    assert [{k: r[k] for k in keys} for r in lockstep_records[: len(REPLIES)]] == [
        {k: r[k] for k in keys} for r in _queries(serial)
    ]


@pytest.mark.parametrize("arm", SCRATCHPAD_ARMS)
def test_3d_first_person_with_start_map(arm):
    """The R2 3D cell (first_person, start map): the map still opens the first
    user message, the notes sit after the current view, the prefix caches."""
    pytest.importorskip("mujoco")
    from gridworld.backends import get_backend

    spec = r1_spec(12)
    backend = get_backend("mujoco3d", camera="first_person", resolution=64, start_map=True)
    try:
        backend.configure(spec)
        runner = build_runner(ExperimentConfig(**{**IMAGE_CELL, **ARMS[arm]}), backend, spec)
        stepper, requests = drive(runner)
    finally:
        backend.close()
    assert [r["scratchpad_status"] for r in _queries(stepper)][:5] == [
        "updated", "updated", "kept_parse_failure", "missing", "truncated",
    ]
    for messages in requests:
        assert SYSTEM_LINE in _system(messages)
        content = next(m["content"] for m in messages if m["role"] == "user")
        assert content[0]["text"].startswith("Map: a top-down view")
        assert content[1]["type"] == "image_url"
        # The current view, then the text with the notes, the question, the format.
        assert content[-2]["type"] == "image_url"
        assert "Your scratchpad:" in content[-1]["text"]
        assert content[-1]["text"].endswith(f"{QUESTION}\n\n{OUTPUT_FORMAT}")
    if arm == "full_context_scratchpad":
        _, blocks = _anthropic_blocks(requests[-1])
        last_mark = max(i for i, b in enumerate(blocks) if "cache_control" in b)
        assert all("Your scratchpad:" not in b.get("text", "") for b in blocks[: last_mark + 1])
