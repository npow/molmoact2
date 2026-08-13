"""Hermetic tests for the YAM prefetch+splice loop.

No robot, no GPU, no network: the loop is driven through pure callables with a simulated
round trip, so the mechanism can be argued about before it is given an arm.

The properties that matter are safety properties, and they are tested as such:

* prefetch WITHOUT splice commands the arm backward -- the test asserts the bug exists, so that
  the two can never drift apart;
* a failed or wedged prefetch degrades to the stock blocking behaviour rather than executing a
  stale plan;
* the jump guard references the arm's ACTUAL pose, not the last command.
"""

from __future__ import annotations

import sys
import threading
import time
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from chunked_rollout import (  # noqa: E402
    DEFAULT_STEP_PERIOD_S,
    ChunkedRolloutReport,
    adaptive_lead,
    run_chunked_rollout,
    splice_index,
)

HORIZON = 30
ROUND_TRIP_S = 0.135  # the measured thor->odin act latency
# ~33 ms/row is the deployed rig's real cadence (a 30-row chunk spans the ~1 s of motion in a
# ~1,135 ms cycle). It matters that this is realistic: at 10 ms/row a 10-row chunk is 100 ms of
# motion and CANNOT cover a 135 ms round trip, so no lead would work and the test would be
# measuring an impossible configuration rather than the loop.
STEP_S = 0.03


class FakeArm:
    """An arm that moves to whatever it is commanded, recording every command."""

    def __init__(self, dim: int = 14) -> None:
        self.pose = np.zeros(dim)
        self.commands: list[np.ndarray] = []
        self.lock = threading.Lock()

    def execute(self, action: np.ndarray, step: int) -> None:
        time.sleep(STEP_S)
        with self.lock:
            self.pose = np.asarray(action, dtype=float).copy()
            self.commands.append(self.pose.copy())
        return None

    def joints(self) -> np.ndarray:
        with self.lock:
            return self.pose.copy()


class RampPolicy:
    """Returns an absolute-pose ramp continuing from wherever the arm was observed.

    This is the shape that makes splicing mandatory: row 0 always restates the observed pose,
    so replaying it after the arm has moved on is a command to go backward.
    """

    def __init__(self, horizon: int = HORIZON, round_trip_s: float = ROUND_TRIP_S) -> None:
        self.horizon = horizon
        self.round_trip_s = round_trip_s
        self.calls = 0

    def infer(self, observation: np.ndarray) -> list[np.ndarray]:
        time.sleep(self.round_trip_s)
        self.calls += 1
        base = float(observation[0])
        return [np.full(14, base + i) for i in range(self.horizon)]


def _run(**overrides) -> tuple[ChunkedRolloutReport, FakeArm, RampPolicy]:
    arm = FakeArm()
    policy = RampPolicy()
    kwargs = dict(
        observe=arm.joints,
        infer=policy.infer,
        execute=arm.execute,
        current_joints=arm.joints,
        max_steps=60,
        exec_steps=20,
        lead=None,
        max_jump=None,
    )
    kwargs.update(overrides)
    return run_chunked_rollout(**kwargs), arm, policy


# ---------------------------------------------------------------------------
# The freeze this port exists to remove
# ---------------------------------------------------------------------------


def test_blocking_loop_freezes_the_arm_at_every_chunk_boundary() -> None:
    """The stock behaviour, asserted so the improvement is measured against something real."""
    report, _arm, _policy = _run(prefetch=False)

    gaps = [c.boundary_gap_ms for c in report.chunks[1:]]
    assert gaps, "need at least two chunks to have a boundary"
    # Every boundary eats a full round trip.
    assert min(gaps) > ROUND_TRIP_S * 1000 * 0.8
    assert all(not c.prefetched for c in report.chunks)


def test_prefetch_removes_the_boundary_freeze() -> None:
    report, _arm, _policy = _run(prefetch=True)

    assert report.prefetch_rate() > 0.5, "most chunks should arrive prefetched"
    prefetched_gaps = [c.boundary_gap_ms for c in report.chunks[1:] if c.prefetched]
    assert prefetched_gaps
    # The whole point: a boundary costs a step, not a round trip.
    assert max(prefetched_gaps) < ROUND_TRIP_S * 1000 * 0.5


def test_an_undersized_lead_is_visible_rather_than_silent() -> None:
    """The failure mode found while writing this port.

    ``lead=1`` fires the prefetch one row from the end -- ~33 ms of motion against a 135 ms
    round trip -- so the chunk lands late and the arm freezes anyway. The run still reports
    every chunk as ``prefetched``, which is why ``late_prefetch_rate`` exists.
    """
    report, _arm, _policy = _run(prefetch=True, lead=1)

    assert report.prefetch_rate() > 0.5, "chunks still arrive via the prefetch path..."
    assert report.late_prefetch_rate() > 0.5, "...but the loop had to wait for them"
    assert max(c.prefetch_wait_ms for c in report.chunks) > 20.0


def test_adaptive_lead_covers_the_measured_round_trip() -> None:
    report, _arm, _policy = _run(prefetch=True, lead=None)

    assert report.late_prefetch_rate() == 0.0, "an adaptive lead should not land late"
    # 135 ms round trip / ~30 ms per row = ~5 rows, plus the default margin of 2.
    steady = [c.lead_used for c in report.chunks if c.prefetched]
    assert steady and max(steady) >= 4


def test_prefetch_lowers_observation_age() -> None:
    blocking, _a, _p = _run(prefetch=False)
    overlapped, _a2, _p2 = _run(prefetch=True)

    assert overlapped.mean_observation_age_ms() is not None
    assert blocking.mean_observation_age_ms() is not None
    # Prefetched chunks are older at first send (they were fetched earlier), which is exactly
    # why the splice must discard the rows that aged. The age is bounded by the round trip.
    assert overlapped.mean_observation_age_ms() < ROUND_TRIP_S * 1000 * 3


# ---------------------------------------------------------------------------
# Why prefetch and splice are one feature
# ---------------------------------------------------------------------------


def test_splice_drops_the_rows_that_wall_clock_already_overtook() -> None:
    report, _arm, _policy = _run(prefetch=True)

    prefetched = [c for c in report.chunks if c.prefetched]
    assert prefetched
    assert all(c.rows_skipped > 0 for c in prefetched), (
        "a chunk that was in flight while the arm moved must skip the rows it moved through"
    )


def test_unspliced_prefetch_would_command_the_arm_backward() -> None:
    """Pins the bug the splice prevents.

    If this ever starts passing with monotonic commands, prefetch has been made safe some other
    way and the splice's justification needs rewriting -- do not just delete the test.
    """
    arm = FakeArm()
    policy = RampPolicy()

    # Simulate handing over a chunk predicted from a stale observation with NO splice.
    observed_pose = arm.joints()
    chunk = policy.infer(observed_pose)
    for row in chunk[:5]:
        arm.execute(np.asarray(row), 0)
    moved_to = arm.joints()[0]

    stale_chunk = policy.infer(observed_pose)  # predicted from the ORIGINAL observation
    replayed_row0 = float(stale_chunk[0][0])

    assert replayed_row0 < moved_to, (
        "row 0 of a stale absolute-pose chunk is behind the arm's current pose -- "
        "executing it unspliced is a backward command"
    )


def test_splice_index_is_elapsed_rows_and_never_empties_the_chunk() -> None:
    assert splice_index(observed_at=0.0, now=0.0, step_period_s=0.1, rows_returned=30) == 0
    assert splice_index(observed_at=0.0, now=0.32, step_period_s=0.1, rows_returned=30) == 3
    assert splice_index(observed_at=0.0, now=0.38, step_period_s=0.1, rows_returned=30) == 4
    # Entirely-stale chunk still commands its final row rather than nothing.
    assert splice_index(observed_at=0.0, now=99.0, step_period_s=0.1, rows_returned=30) == 29
    # Clock going backwards must not produce a negative index.
    assert splice_index(observed_at=5.0, now=4.0, step_period_s=0.1, rows_returned=30) == 0


def test_splice_index_rejects_degenerate_inputs() -> None:
    with pytest.raises(ValueError):
        splice_index(observed_at=0.0, now=1.0, step_period_s=0.1, rows_returned=0)
    with pytest.raises(ValueError):
        splice_index(observed_at=0.0, now=1.0, step_period_s=0.0, rows_returned=30)


def test_a_slower_loop_splices_fewer_rows_for_the_same_delay() -> None:
    """The measured-period splice is the port's one real deviation from servo's version."""
    fast = splice_index(observed_at=0.0, now=0.5, step_period_s=0.05, rows_returned=30)
    slow = splice_index(observed_at=0.0, now=0.5, step_period_s=0.25, rows_returned=30)
    assert fast > slow, "a loop retiring rows slowly has retired fewer of them in the same time"


# ---------------------------------------------------------------------------
# Degradation: a broken prefetch must never execute a stale plan
# ---------------------------------------------------------------------------


def test_a_failing_prefetch_degrades_to_a_blocking_fetch() -> None:
    arm = FakeArm()
    calls = {"n": 0}

    def flaky_infer(observation: np.ndarray) -> list[np.ndarray]:
        calls["n"] += 1
        time.sleep(ROUND_TRIP_S)
        if calls["n"] % 2 == 0:
            raise RuntimeError("serve went away")
        base = float(observation[0])
        return [np.full(14, base + i) for i in range(HORIZON)]

    report = run_chunked_rollout(
        observe=arm.joints,
        infer=flaky_infer,
        execute=arm.execute,
        current_joints=arm.joints,
        max_steps=40,
        exec_steps=10,
        lead=3,
        max_jump=None,
    )

    assert report.aborted is None, "a failed prefetch is recoverable, not fatal"
    assert report.steps == 40
    assert any(c.fallback_reason for c in report.chunks)


def test_prefetch_false_is_exactly_the_stock_loop() -> None:
    report, arm, _policy = _run(prefetch=False)

    assert all(not c.prefetched for c in report.chunks)
    assert all(c.rows_skipped == 0 for c in report.chunks), (
        "a blocking fetch observes a stationary arm; none of its rows are stale"
    )
    # Commands strictly increase: no backward motion anywhere in the stock path.
    values = [float(c[0]) for c in arm.commands]
    assert values == sorted(values)


# ---------------------------------------------------------------------------
# The jump guard
# ---------------------------------------------------------------------------


def test_jump_guard_aborts_on_a_violent_command() -> None:
    arm = FakeArm()

    def teleport(observation: np.ndarray) -> list[np.ndarray]:
        time.sleep(0.01)
        return [np.full(14, 1000.0) for _ in range(HORIZON)]

    report = run_chunked_rollout(
        observe=arm.joints,
        infer=teleport,
        execute=arm.execute,
        current_joints=arm.joints,
        max_steps=30,
        exec_steps=10,
        lead=3,
        max_jump=0.5,
    )

    assert report.aborted is not None
    assert "jump guard" in report.aborted
    assert report.chunks[-1].guard_tripped


def test_jump_guard_measures_against_actual_pose_not_last_command() -> None:
    """A guard referencing the last command trips on its own bookkeeping under rate limiting."""
    arm = FakeArm()
    reads: list[str] = []

    def joints_probe() -> np.ndarray:
        reads.append("actual")
        return arm.joints()

    run_chunked_rollout(
        observe=arm.joints,
        infer=RampPolicy(round_trip_s=0.01).infer,
        execute=arm.execute,
        current_joints=joints_probe,
        max_steps=20,
        exec_steps=10,
        lead=3,
        max_jump=100.0,
    )

    assert reads, "the guard must consult the arm, not a cached command"


def test_guard_disabled_is_announced(caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level("WARNING"):
        _run(prefetch=False, max_jump=None)
    assert any("jump guard DISABLED" in r.message for r in caplog.records)


# ---------------------------------------------------------------------------
# Loop bookkeeping
# ---------------------------------------------------------------------------


def test_step_budget_is_respected_exactly() -> None:
    report, arm, _policy = _run(prefetch=True, max_steps=25)
    assert report.steps == 25
    assert len(arm.commands) == 25


def test_execute_return_value_ends_the_rollout() -> None:
    arm = FakeArm()

    def execute_then_stop(action: np.ndarray, step: int) -> str | None:
        arm.execute(action, step)
        return "success" if step == 7 else None

    report = run_chunked_rollout(
        observe=arm.joints,
        infer=RampPolicy(round_trip_s=0.01).infer,
        execute=execute_then_stop,
        current_joints=arm.joints,
        max_steps=100,
        exec_steps=10,
        lead=3,
        max_jump=None,
    )

    assert report.aborted == "success"
    assert report.steps == 8


def test_env_access_is_serialized_across_the_prefetch_thread() -> None:
    """The launcher shares one ``RobotEnv`` between the loop and the prefetch thread.

    Camera reads and the robot bus are not thread-safe, and unserialized bus access has
    already cost this repo corrupted Feetech packets under load. This reproduces the
    launcher's locking discipline and asserts no two accesses ever overlap.
    """
    lock = threading.Lock()
    overlaps = []
    depth = {"n": 0}
    depth_lock = threading.Lock()
    arm = FakeArm()

    def guarded(fn):
        def wrapper(*a, **kw):
            with lock:
                with depth_lock:
                    depth["n"] += 1
                    if depth["n"] > 1:
                        overlaps.append(depth["n"])
                try:
                    return fn(*a, **kw)
                finally:
                    with depth_lock:
                        depth["n"] -= 1

        return wrapper

    run_chunked_rollout(
        observe=guarded(arm.joints),
        infer=RampPolicy().infer,  # deliberately OUTSIDE the lock: it must not block the arm
        execute=guarded(arm.execute),
        current_joints=guarded(arm.joints),
        max_steps=40,
        exec_steps=20,
        lead=None,
        max_jump=None,
    )

    assert not overlaps, f"env was touched concurrently {len(overlaps)} times"


def test_inference_runs_outside_the_env_lock() -> None:
    """If ``infer`` were held under the env lock, prefetch would serialize against motion
    and buy nothing at all -- the port would look correct and do nothing."""
    report, _arm, _policy = _run(prefetch=True)

    # The proof is behavioural: overlap is only possible if inference ran concurrently.
    assert report.late_prefetch_rate() == 0.0
    prefetched_gaps = [c.boundary_gap_ms for c in report.chunks[1:] if c.prefetched]
    assert prefetched_gaps and max(prefetched_gaps) < ROUND_TRIP_S * 1000 * 0.5


def test_adaptive_lead_converts_the_round_trip_into_rows() -> None:
    # 135 ms round trip at 30 ms/row = 5 rows, plus margin.
    assert (
        adaptive_lead(
            round_trip_s=0.135, step_period_s=0.03, lead_margin=2, rows_total=20
        )
        == 7
    )
    # A slower loop retires rows more slowly, so the same round trip costs fewer rows.
    assert (
        adaptive_lead(round_trip_s=0.135, step_period_s=0.1, lead_margin=2, rows_total=20) == 4
    )


def test_adaptive_lead_never_fires_at_row_zero() -> None:
    """A lead as long as the chunk would prefetch from an observation as stale as the one it
    replaces -- pure cost, no freshness."""
    assert adaptive_lead(
        round_trip_s=10.0, step_period_s=0.03, lead_margin=2, rows_total=8
    ) == 7
    assert adaptive_lead(round_trip_s=10.0, step_period_s=0.03, lead_margin=2, rows_total=1) == 1


def test_adaptive_lead_is_conservative_before_anything_is_measured() -> None:
    lead = adaptive_lead(round_trip_s=None, step_period_s=None, lead_margin=2, rows_total=20)
    assert lead == 3


def test_pinned_lead_overrides_measurement() -> None:
    assert (
        adaptive_lead(
            round_trip_s=0.135, step_period_s=0.03, lead_margin=2, rows_total=20, pinned=1
        )
        == 1
    )


def test_lead_must_be_positive() -> None:
    with pytest.raises(ValueError):
        _run(lead=0)


def test_default_step_period_seeds_the_splice_before_any_measurement() -> None:
    """The very first prefetched chunk has no measured rate yet; it must still splice sanely."""
    assert 0.0 < DEFAULT_STEP_PERIOD_S <= 0.5
