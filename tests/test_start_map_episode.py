"""Where the start map goes: the FIRST user message of every request opens
with [map text, map image], in every chat mode, exactly once; stored chat
turns never carry it; the logged agent_messages show it."""

from __future__ import annotations

import json

import numpy as np
import pytest

pytest.importorskip("mujoco")
pytest.importorskip("minigrid")

from gridworld.backends import get_backend  # noqa: E402
from interface.agents.reply import Reply  # noqa: E402
from interface.config import ExperimentConfig  # noqa: E402
from interface.episode_checkpoint import resume_stepper, save_checkpoint  # noqa: E402
from interface.episode_log import flush_episode_log  # noqa: E402
from interface.episode_step import EpisodeStepper  # noqa: E402
from interface.renderer import rgb_to_image_block  # noqa: E402
from interface.runner import build_runner  # noqa: E402
from prompting_experiments.prompt_templates import user as user_templates  # noqa: E402
from start_map_test_utils import R1_CELL, r1_spec, scripted_replies  # noqa: E402

RES = 64
QUERIES = 10  # 9 steps + one unparseable reply
_OPEN: list = []


@pytest.fixture(autouse=True)
def _close_backends():
    yield
    while _OPEN:
        _OPEN.pop().close()


def _build(camera="first_person", start_map=True, **overrides):
    spec = r1_spec()
    backend = get_backend("mujoco3d", camera=camera, resolution=RES, start_map=start_map)
    _OPEN.append(backend)
    backend.configure(spec)
    return build_runner(ExperimentConfig(**{**R1_CELL, **overrides}), backend, spec), spec


def _drive(runner, spec):
    """Drive the stepper by hand; return (stepper, requests, stored-chat snapshots)."""
    stepper = EpisodeStepper(runner)
    stepper.start()
    replies = iter(scripted_replies(spec))
    requests, stored = [], []
    while (messages := stepper.next_query()) is not None:
        requests.append(json.loads(json.dumps(messages)))
        stepper.apply_reply(Reply(text=next(replies), usage={"input_tokens": 1, "output_tokens": 1}))
        stored.append(json.loads(json.dumps(stepper.messages)))
    return stepper, requests, stored


def _fresh_map_url(spec):
    top = get_backend("mujoco3d", camera="top_down", resolution=RES)
    top.configure(spec)
    try:
        frame, _, _ = top.reset(seed=spec.seed)
    finally:
        top.close()
    return rgb_to_image_block(frame)["image_url"]["url"]


def _urls(message):
    content = message["content"]
    if not isinstance(content, list):
        return []
    return [b["image_url"]["url"] for b in content if b["type"] == "image_url"]


def _first_user(request):
    return next(m for m in request if m["role"] == "user")


def _assert_map_opens_every_request(requests, map_text, map_url):
    assert len(requests) == QUERIES
    for request in requests:
        first = _first_user(request)
        assert first["content"][0] == {"type": "text", "text": map_text}
        assert first["content"][1]["image_url"]["url"] == map_url
        urls = [u for m in request for u in _urls(m)]
        assert urls.count(map_url) == 1  # exactly one map per request
        texts = [b["text"] for m in request if isinstance(m["content"], list) for b in m["content"] if b["type"] == "text"]
        assert sum(map_text in t for t in texts) == 1


def _assert_stored_chat_is_lean(stored, map_text, map_url):
    for messages in stored:
        for message in messages:
            assert map_url not in _urls(message)
            if isinstance(message["content"], list):
                assert all(map_text not in b.get("text", "") for b in message["content"])
            else:
                assert map_text not in message["content"]


def _logged(stepper):
    return [rec["agent_messages"] for rec in stepper.transcript if rec["kind"] == "query"]


@pytest.mark.parametrize("context_window", ["current", "last_n", "text_summary", "text_summary_and_last_n"])
def test_stateless_every_query_opens_with_the_map(context_window):
    runner, spec = _build(context_window=context_window)
    stepper, requests, _ = _drive(runner, spec)
    map_text, map_url = user_templates.START_MAP_MINIMAL, _fresh_map_url(spec)
    _assert_map_opens_every_request(requests, map_text, map_url)
    reset_url = rgb_to_image_block(stepper.transcript[0]["_reset_frame_rgb"])["image_url"]["url"]
    # turn 1: map + the current first-person view, nothing else
    assert _urls(requests[0][1]) == [map_url, reset_url]
    for request in requests:
        assert [m["role"] for m in request] == ["system", "user"]
        urls = _urls(request[1])
        assert urls[-1] != map_url  # the current view still closes the message
        if context_window in ("last_n", "text_summary_and_last_n"):
            assert len(urls) <= 1 + 3 + 1  # map + at most n history frames + current
    if context_window in ("last_n", "text_summary_and_last_n"):
        assert len(_urls(requests[-1][1])) == 1 + 3 + 1  # the map never rotates out
    assert _logged(stepper) == requests


@pytest.mark.parametrize("turns", [1, 3])
def test_rolling_puts_the_map_on_the_oldest_user_turn_in_the_window(turns):
    runner, spec = _build(chat_history="rolling", chat_turns_max=turns)
    stepper, requests, stored = _drive(runner, spec)
    map_text, map_url = user_templates.START_MAP_MINIMAL, _fresh_map_url(spec)
    _assert_map_opens_every_request(requests, map_text, map_url)
    _assert_stored_chat_is_lean(stored, map_text, map_url)
    reset_url = rgb_to_image_block(stepper.transcript[0]["_reset_frame_rgb"])["image_url"]["url"]
    for q, request in enumerate(requests, start=1):
        users = [m for m in request if m["role"] == "user"]
        assert len(users) == min(q, turns + 1)  # the trimmed window
        # the map rides on whichever user turn is oldest, then that turn's own content
        assert len(_urls(users[0])) == 2 and _urls(users[0])[0] == map_url
        if q > turns + 1:
            # turn 1 and its first-person view have rotated out; the map has not
            assert reset_url not in [u for m in request for u in _urls(m)]
    assert _logged(stepper) == requests


def test_full_chat_carries_the_map_once_on_turn_one():
    runner, spec = _build(chat_history="full")
    stepper, requests, stored = _drive(runner, spec)
    map_text, map_url = user_templates.START_MAP_MINIMAL, _fresh_map_url(spec)
    _assert_map_opens_every_request(requests, map_text, map_url)
    _assert_stored_chat_is_lean(stored, map_text, map_url)
    reset_url = rgb_to_image_block(stepper.transcript[0]["_reset_frame_rgb"])["image_url"]["url"]
    for request in requests:
        assert _urls(_first_user(request)) == [map_url, reset_url]  # turn 1, kept whole
    assert _logged(stepper) == requests


def test_map_comes_before_the_one_shot_example_too():
    runner, spec = _build(chat_history="rolling", chat_turns_max=1, in_context_learning="one_shot")
    _, requests, stored = _drive(runner, spec)
    map_text, map_url = user_templates.START_MAP_MINIMAL, _fresh_map_url(spec)
    _assert_map_opens_every_request(requests, map_text, map_url)
    _assert_stored_chat_is_lean(stored, map_text, map_url)
    first = requests[0][1]["content"]  # turn 1 is the current turn: map, then one-shot
    assert first[2]["text"].startswith(user_templates.ONE_SHOT_EXAMPLE_INTRO)


def test_standard_labels_the_current_view_and_uses_the_standard_map_text():
    runner, spec = _build(camera="chase", prompting="standard")
    _, requests, _ = _drive(runner, spec)
    _assert_map_opens_every_request(requests, user_templates.START_MAP_STANDARD, _fresh_map_url(spec))
    content = requests[0][1]["content"]
    label = "Your view now (from behind and above you):"
    i = next(i for i, b in enumerate(content) if b["type"] == "text" and label in b["text"])
    assert content[i]["text"].rstrip("\n").endswith(label)
    assert content[i + 1]["type"] == "image_url"  # right before the current image
    assert i + 1 == max(j for j, b in enumerate(content) if b["type"] == "image_url")


def test_the_map_is_kept_on_the_runner_and_logged_once(tmp_path):
    runner, spec = _build()
    result = runner.run(lambda messages: "FINAL_OUTPUT: TURN_LEFT", verbose=False)
    assert runner.start_map_rgb is not None
    assert np.array_equal(result["transcript"][0]["_start_map_rgb"], runner.start_map_rgb)
    flush_episode_log(result, tmp_path / "ep")
    episode = json.loads((tmp_path / "ep" / "episode.json").read_text())
    assert episode["transcript"][0]["start_map_frame"] == "frames/start_map.png"
    assert (tmp_path / "ep" / "frames" / "start_map.png").exists()
    # the logged request shows what the model saw: the map image file is there
    query = json.loads((tmp_path / "ep" / "queries" / "query_001" / "query.json").read_text())
    user = query["agent_messages"][1]["content"]
    assert user[0] == {"type": "text", "text": user_templates.START_MAP_MINIMAL}
    assert user[1]["type"] == "image"
    png = (tmp_path / "ep" / "queries" / "query_001" / user[1]["file"]).read_bytes()
    assert png == (tmp_path / "ep" / "frames" / "start_map.png").read_bytes()


def test_no_map_no_map_artifacts(tmp_path):
    runner, spec = _build(start_map=False)
    result = runner.run(lambda messages: "FINAL_OUTPUT: TURN_LEFT", verbose=False)
    assert runner.start_map_rgb is None
    assert "_start_map_rgb" not in result["transcript"][0]
    flush_episode_log(result, tmp_path / "ep")
    assert not (tmp_path / "ep" / "frames" / "start_map.png").exists()
    assert "start_map_frame" not in json.loads((tmp_path / "ep" / "episode.json").read_text())["transcript"][0]


def test_a_text_only_run_refuses_a_start_map():
    runner, _ = _build(observation="text_only")
    with pytest.raises(ValueError, match="text_only"):
        EpisodeStepper(runner).start()


def test_a_resumed_episode_still_opens_with_the_map(tmp_path):
    runner, spec = _build()
    stepper = EpisodeStepper(runner)
    stepper.start()
    replies = iter(scripted_replies(spec, parse_failure_at=None))
    for _ in range(3):
        stepper.next_query()
        stepper.apply_reply(Reply(text=next(replies)))
    expected = stepper.next_query()  # the round in flight at the crash
    stepper.action_queue, stepper.primitive_buffer = [], []
    save_checkpoint(tmp_path / "ckpt.json", stepper)

    fresh, _ = _build()
    resumed = resume_stepper(tmp_path / "ckpt.json", runner=fresh)
    assert resumed.next_query() == expected
    assert np.array_equal(resumed.transcript[0]["_start_map_rgb"], fresh.start_map_rgb)
