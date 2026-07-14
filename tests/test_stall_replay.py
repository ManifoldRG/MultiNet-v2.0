"""Unit tests for analysis.stall_replay (frozen-replay stall calibration).

`replay_stall` never re-runs an episode; it walks each episode's already-
recorded `state_after` snapshots and reproduces the runner's watchdog rule
(interface/runner.py: seed `seen_signatures` with the initial state's
signature, increment a streak on a signature already seen, reset to 0 on a
novel one) to answer "would the watchdog have fired, and what would it have
cost/saved?" This is a decision aid for calibrating K against already-
collected data, not a live safety check — the tests below lock the COUNTING
LOGIC, not any claim that a particular K is "safe".
"""

from __future__ import annotations

from analysis.stall_replay import replay_stall


def _state(pos, *, direction=0):
    # GridState.from_dict only requires agent_position/agent_direction;
    # everything else (agent_carrying, collected_keys, open_doors, ...)
    # defaults per gridworld/backends/base.py, so distinct agent_position
    # values alone are enough to make _progress_signature novel/non-novel.
    return {"agent_position": list(pos), "agent_direction": direction}


def _step(pos):
    return {"kind": "step", "state_after": _state(pos)}


def _eventual_win_episode():
    # Novel start -> 1 novel move -> 5 repeats of that same cell (streak
    # maxes at 5) -> breaks out to two more novel cells, the second being
    # the goal. success=True.
    transcript = (
        [_step((0, 1))]  # novel: streak resets to 0
        + [_step((0, 1))] * 5  # non-novel repeats: streak climbs 1..5
        + [_step((0, 2))]  # novel: streak resets to 0
        + [_step((0, 3))]  # novel goal cell
    )
    return {
        "success": True,
        "initial_state": _state((0, 0)),
        "transcript": transcript,
    }


def _stalling_failure_episode():
    # Novel start -> 1 novel move -> 14 repeats of that same cell (streak
    # climbs 1..14), episode ends without success. 15 step records total.
    transcript = [_step((5, 6))] + [_step((5, 6))] * 14
    return {
        "success": False,
        "initial_state": _state((5, 5)),
        "transcript": transcript,
    }


def _corpus():
    return [_eventual_win_episode(), _stalling_failure_episode()]


def test_replay_stall_k3_counts():
    # K=3: win's max streak (5) >= 3 -> killed. Failure's streak first hits
    # 3 at step index 4 (1 novel + 3 repeats); 15 - 4 = 11 primitives saved.
    result = replay_stall(_corpus(), K=3)
    assert result == {
        "eventual_wins_killed": 1,
        "failures_caught": 1,
        "failed_primitive_steps_saved": 11,
    }


def test_replay_stall_k10_counts():
    # K=10: win's max streak (5) < 10 -> not killed. Failure's streak first
    # hits 10 at step index 11 (1 novel + 10 repeats); 15 - 11 = 4 saved.
    result = replay_stall(_corpus(), K=10)
    assert result == {
        "eventual_wins_killed": 0,
        "failures_caught": 1,
        "failed_primitive_steps_saved": 4,
    }


def test_replay_stall_empty_corpus():
    assert replay_stall([], K=5) == {
        "eventual_wins_killed": 0,
        "failures_caught": 0,
        "failed_primitive_steps_saved": 0,
    }


def test_replay_stall_no_stall_when_streak_never_reached():
    # A short episode whose signatures never repeat can't be killed at any K.
    transcript = [_step((0, 1)), _step((0, 2)), _step((0, 3))]
    episode = {"success": True, "initial_state": _state((0, 0)), "transcript": transcript}
    result = replay_stall([episode], K=3)
    assert result == {
        "eventual_wins_killed": 0,
        "failures_caught": 0,
        "failed_primitive_steps_saved": 0,
    }
