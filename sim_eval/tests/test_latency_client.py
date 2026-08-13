"""Hermetic tests for the sim latency emulator and the A/B run's validity checks.

No simulator, no GPU, no serve: the client is driven against a fake inner client so the
boundary policies can be checked exactly, and the validity checks are exercised against the
numbers the first real run actually produced.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from sim_eval.inference.latency_client import LatencyEmulatingClient  # noqa: E402

HORIZON = 10


class FakeInner:
    """Returns an absolute-pose ramp counting up from the query index."""

    schema = None
    state_adapter = None
    action_adapter = None

    def __init__(self) -> None:
        self.queries = 0

    def reset(self) -> None:
        self.queries = 0

    def _query_server(self, obs, instruction):
        base = 100 * self.queries
        self.queries += 1
        return [np.full(4, base + i, dtype=np.float32) for i in range(HORIZON)]


def _drive(mode: str, steps: int = 40, **kwargs) -> tuple[LatencyEmulatingClient, list[float]]:
    client = LatencyEmulatingClient(FakeInner(), mode=mode, latency_steps=4, **kwargs)
    commands = [float(client.infer({}, "task")[0]) for _ in range(steps)]
    return client, commands


def test_blocking_freezes_for_the_round_trip_at_every_boundary() -> None:
    client, commands = _drive("blocking")

    assert client.stats["frozen_steps"] > 0
    # A freeze repeats the previous command verbatim.
    repeats = sum(1 for a, b in zip(commands, commands[1:]) if a == b)
    assert repeats >= client.stats["frozen_steps"] - 1


def test_prefetch_splice_never_freezes_when_the_lead_covers_the_round_trip() -> None:
    client, _commands = _drive("prefetch_splice")

    assert client.stats["frozen_steps"] == 0
    assert client.stats["late_prefetches"] == 0
    assert client.stats["rows_skipped"] > 0


def test_an_undersized_lead_reintroduces_the_freeze() -> None:
    client, _commands = _drive("prefetch_splice", lead=1)

    assert client.stats["late_prefetches"] > 0
    assert client.stats["frozen_steps"] > 0


def test_nosplice_replays_stale_rows_and_splice_does_not() -> None:
    spliced, _ = _drive("prefetch_splice")
    stale, _ = _drive("prefetch_nosplice")

    assert spliced.stats["rows_skipped"] > 0
    assert stale.stats["rows_skipped"] == 0


def test_first_fetch_is_exempt_from_the_freeze_in_every_mode() -> None:
    """At reset the arm is stationary; a freeze there costs no progress and would otherwise be
    charged to some modes and not others."""
    for mode in ("blocking", "prefetch_splice", "prefetch_nosplice"):
        client = LatencyEmulatingClient(FakeInner(), mode=mode, latency_steps=4)
        first = client.infer({}, "task")
        assert first is not None
        # The very first call must return a real chunk row, not a hold.
        assert float(first[0]) == 0.0


def test_exec_steps_equalizes_the_requery_rate_across_modes() -> None:
    """The confound the first real run walked into: unpinned exec_steps makes the splice arm
    replan more often, so the comparison also contains the exec_steps lever."""
    unpinned = {m: _drive(m, steps=60)[0].stats["queries"] for m in ("blocking", "prefetch_splice")}
    assert unpinned["prefetch_splice"] > unpinned["blocking"]

    pinned = {
        m: _drive(m, steps=60, exec_steps=5)[0].stats["queries"]
        for m in ("blocking", "prefetch_splice")
    }
    # Pinned, the splice arm no longer replans more often *because of* the splice. Blocking
    # still queries less because its freeze consumes steps without executing rows -- that is
    # the effect under test, not a confound.
    assert pinned["prefetch_splice"] >= pinned["blocking"]
    assert (
        pinned["prefetch_splice"] - pinned["blocking"]
        <= unpinned["prefetch_splice"] - unpinned["blocking"]
    )


def test_rejects_unknown_mode_and_bad_lead() -> None:
    with pytest.raises(ValueError):
        LatencyEmulatingClient(FakeInner(), mode="nope", latency_steps=4)
    with pytest.raises(ValueError):
        LatencyEmulatingClient(FakeInner(), mode="blocking", latency_steps=-1)
    with pytest.raises(ValueError):
        LatencyEmulatingClient(FakeInner(), mode="blocking", latency_steps=4, lead=0)


# ---------------------------------------------------------------------------
# Validity checks, against the numbers the first real run produced
# ---------------------------------------------------------------------------


def test_power_calculation_matches_the_observed_shortfall() -> None:
    from sim_eval.run_boundary_ab import required_episodes

    # The real run at n=60. The requirement depends entirely on WHICH pair is being resolved,
    # which is why a single "episodes needed" number for this experiment is meaningless.
    #
    # blocking 1.7% vs splice 11.7% -- a large relative effect, so relatively cheap:
    assert required_episodes(0.017, 0.117) == 95
    # zero-latency ceiling 6.7% vs splice 11.7% -- a 5 pp gap at a higher base rate, and the
    # comparison that actually decides whether the port recovers anything:
    assert required_episodes(0.067, 0.117) > 500
    # Either way n=60 was short.
    assert required_episodes(0.017, 0.117) > 60


def test_validity_flags_the_requery_confound_from_the_real_run() -> None:
    from sim_eval.run_boundary_ab import ABConfig, _validity_warnings

    summary = {
        "blocking": {"episodes": 60, "success_rate": 0.017, "queries_per_episode": 24.0},
        "prefetch_splice": {"episodes": 60, "success_rate": 0.117, "queries_per_episode": 31.6},
    }
    warnings = _validity_warnings(summary, ABConfig(remote_url="x", exec_steps=None))

    assert any("re-query at different rates" in w for w in warnings)
    assert any("underpowered" in w for w in warnings)


def test_validity_is_quiet_on_a_clean_run() -> None:
    from sim_eval.run_boundary_ab import ABConfig, _validity_warnings

    summary = {
        "blocking": {"episodes": 600, "success_rate": 0.30, "queries_per_episode": 30.0},
        "prefetch_splice": {"episodes": 600, "success_rate": 0.45, "queries_per_episode": 31.0},
    }
    warnings = _validity_warnings(summary, ABConfig(remote_url="x", exec_steps=20))

    assert warnings == []
