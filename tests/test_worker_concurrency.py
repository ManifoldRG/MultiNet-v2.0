"""run_worker_loop concurrency: with concurrency>1 it runs up to N units in parallel
(so a served-vLLM worker's prefills batch), while concurrency=1 stays serial. Uses an
injected client + process_fn so it's testable without a coordinator or GPU."""
import threading
import time

from scripts.distributed_run_pipeline import run_worker_loop


class _FakeClient:
    def __init__(self, n_units: int):
        self.units = [{"unit_id": f"u{i}"} for i in range(n_units)]
        self._lock = threading.Lock()

    def register(self, payload):
        return {"worker_id": "w1"}

    def assign(self, worker_id, capabilities):
        with self._lock:
            return {"unit": self.units.pop(0)} if self.units else {"unit": None}


def _run(concurrency, n_units, tmp_path):
    client = _FakeClient(n_units)
    state = {"n": 0, "peak": 0}
    lock = threading.Lock()
    processed = []

    def fake_process(_c, _w, unit):
        with lock:
            state["n"] += 1
            state["peak"] = max(state["peak"], state["n"])
        time.sleep(0.05)
        with lock:
            state["n"] -= 1
            processed.append(unit["unit_id"])
        return True

    run_worker_loop(
        coordinator_url="x", artifacts_root=tmp_path, capabilities={},
        worker_state_path=tmp_path / "ws.json", poll_interval_seconds=0.01,
        concurrency=concurrency, drain=True, client=client, process_fn=fake_process,
    )
    return processed, state["peak"]


def test_concurrent_runs_in_parallel_and_processes_all(tmp_path):
    processed, peak = _run(concurrency=4, n_units=6, tmp_path=tmp_path)
    assert set(processed) == {f"u{i}" for i in range(6)}
    assert peak >= 2          # genuinely concurrent
    assert peak <= 4          # respected the cap


def test_serial_processes_all_one_at_a_time(tmp_path):
    processed, peak = _run(concurrency=1, n_units=5, tmp_path=tmp_path)
    assert set(processed) == {f"u{i}" for i in range(5)}
    assert peak == 1          # strictly serial
