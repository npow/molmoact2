"""Emulate act-path latency inside ManiSkill, to A/B the chunk-boundary policy.

``examples/yam/chunked_rollout.py`` removes the round-trip freeze from the real YAM loop. Its
hermetic tests prove the *mechanism* (boundary gap 135 ms -> 0 ms, no backward commands), but
they cannot answer the question that gates turning it on: **does executing spliced chunks change
what the robot accomplishes?**

**Simulation has no wall clock.** ``env.step()`` advances sim time deterministically; a policy
that costs the operator 135 ms costs the simulated arm exactly nothing. Run naively, every mode
scores identically and the experiment proves nothing.

So latency is injected explicitly, denominated in **sim steps**: at the default 30 Hz control
rate one step is 33.3 ms, so the measured 135 ms thor->odin round trip is ~4 steps. That makes
three boundary policies physically distinct inside the simulator:

``blocking``
    The stock loop. When the chunk runs out the arm holds its last command for
    ``latency_steps`` while the policy thinks, then executes the fresh chunk from row 0.

``prefetch_splice``
    The port. The query is issued ``lead`` rows before the chunk runs out, so the answer is in
    hand at the boundary; rows whose moment passed in flight are dropped.

``prefetch_nosplice``
    Prefetch without the splice, executing a stale absolute-pose chunk from row 0. This is what
    a partial port ships, and it is the positive control: if it does not score worse, the
    experiment is not sensitive to boundary behaviour at all.

READ THIS BEFORE TRUSTING A RESULT
----------------------------------
**Pin ``exec_steps`` across every arm.** With ``exec_steps=None`` each mode consumes whatever
its chunk leaves it, and splicing discards leading rows -- so the splice arm executes fewer rows
per chunk and therefore **re-queries more often**. Measured on the first run of this harness:
31.6 queries/episode for ``prefetch_splice`` against 26.6 for a zero-latency baseline, a 30 %
difference. That is a *different* lever (the ``exec_steps`` freshness frontier) silently mixed
into the comparison, and it makes any apparent splice benefit uninterpretable.

``run_boundary_ab`` reports queries-per-episode per arm and flags a spread, because this
confound is invisible in the success numbers alone.

**Check the power.** The stock ManiSkill YAM task succeeds ~7 % of the time zero-shot. Resolving
a 5 pp difference at that base rate needs roughly 500 episodes per arm; at n=60 the noise floor
exceeded the effect and the spliced arm scored *above* the zero-latency ceiling, which is
impossible and is the tell that nothing was being measured. See
``sim_eval/BOUNDARY_AB_RESULTS.md``.
"""

from __future__ import annotations

import logging
from typing import Any, Optional

import numpy as np

from .client import MolmoActClientBase

logger = logging.getLogger(__name__)

MODES = ("blocking", "prefetch_splice", "prefetch_nosplice")


class LatencyEmulatingClient(MolmoActClientBase):
    """Wraps a real client, reshaping WHEN it is queried and WHICH rows are executed.

    Deliberately does not touch the wire format, the observation, or the policy: the inner
    client's ``_query_server`` is called exactly as the stock evaluation calls it. The only
    variables are the boundary policy and the injected latency.
    """

    def __init__(
        self,
        inner: MolmoActClientBase,
        *,
        mode: str,
        latency_steps: int,
        lead: Optional[int] = None,
        exec_steps: Optional[int] = None,
    ) -> None:
        if mode not in MODES:
            raise ValueError(f"unknown mode {mode!r}; expected one of {MODES}")
        if latency_steps < 0:
            raise ValueError("latency_steps cannot be negative")
        self._inner = inner
        self.mode = mode
        self.latency_steps = latency_steps
        # The lead must cover the round trip, or the prefetch lands late and the freeze returns.
        # One row of margin, matching chunked_rollout's posture.
        self.lead = int(lead) if lead is not None else latency_steps + 1
        if self.lead < 1:
            raise ValueError("lead must be at least 1 row")
        self.exec_steps = exec_steps
        self.reset()

    # -- MolmoActClientBase delegation -----------------------------------------------------

    @property
    def schema(self):  # pragma: no cover - delegation
        return self._inner.schema

    @property
    def state_adapter(self):  # pragma: no cover - delegation
        return self._inner.state_adapter

    @property
    def action_adapter(self):  # pragma: no cover - delegation
        return self._inner.action_adapter

    def reset(self) -> None:
        self._inner.reset()
        self._queue: list[np.ndarray] = []
        self._step = 0
        self._last_action: Optional[np.ndarray] = None
        self._pending: Optional[dict[str, Any]] = None
        # Per-episode counters, read by the runner.
        #
        # `queries` is not bookkeeping -- it is the confound detector. Arms that re-query at
        # different rates are not comparable, and the success numbers alone will not say so.
        self.stats = {
            "queries": 0,
            "frozen_steps": 0,
            "rows_skipped": 0,
            "late_prefetches": 0,
        }

    def infer(self, obs: dict, instruction: str) -> np.ndarray:
        action = self._next_action(obs, instruction)
        self._last_action = np.asarray(action, dtype=np.float32).copy()
        self._step += 1
        return action

    # -- boundary policy ------------------------------------------------------------------

    def _next_action(self, obs: dict, instruction: str) -> np.ndarray:
        if self.mode == "blocking":
            return self._blocking(obs, instruction)
        return self._prefetching(obs, instruction)

    def _blocking(self, obs: dict, instruction: str) -> np.ndarray:
        if self._queue:
            return self._pop()

        # The boundary: observe, then hold position for the whole round trip.
        if self._pending is None:
            self._pending = {
                "chunk": self._query(obs, instruction),
                "ready_at": self._step + self.latency_steps,
            }

        if self._freezing(self._pending["ready_at"]):
            self.stats["frozen_steps"] += 1
            return self._hold()

        chunk = self._pending["chunk"]
        self._pending = None
        # A blocking fetch observed a stationary arm, so none of its rows are stale.
        self._queue = list(chunk[: self._budget(len(chunk))])
        return self._pop()

    def _prefetching(self, obs: dict, instruction: str) -> np.ndarray:
        # Fire early enough that the answer is in hand before the chunk runs out.
        if self._pending is None and 0 < len(self._queue) <= self.lead:
            self._issue(obs, instruction)

        if self._queue:
            return self._pop()

        # Cold start, or a prefetch that has not landed yet.
        if self._pending is None:
            self._issue(obs, instruction)

        if self._freezing(self._pending["ready_at"]):
            self.stats["frozen_steps"] += 1
            self.stats["late_prefetches"] += 1
            return self._hold()

        pending = self._pending
        self._pending = None
        chunk = pending["chunk"]

        if self.mode == "prefetch_splice":
            # Drop the rows whose moment passed while the chunk was in flight. Never empty the
            # chunk: its last row is the closest thing to a current target it contains.
            skip = int(min(max(0, self._step - pending["observed_at"]), len(chunk) - 1))
        else:
            skip = 0
        self.stats["rows_skipped"] += skip

        rows = chunk[skip:]
        self._queue = list(rows[: self._budget(len(rows))])
        return self._pop()

    # -- helpers --------------------------------------------------------------------------

    def _issue(self, obs: dict, instruction: str) -> None:
        self._pending = {
            "chunk": self._query(obs, instruction),
            "observed_at": self._step,
            "ready_at": self._step + self.latency_steps,
        }

    def _budget(self, available: int) -> int:
        """Rows to execute from this chunk.

        ``exec_steps=None`` consumes whatever the chunk leaves, which is what makes the arms
        re-query at different rates -- see the module docstring. Pin it for a valid A/B.
        """
        if self.exec_steps is None:
            return available
        return max(1, min(self.exec_steps, available))

    def _query(self, obs: dict, instruction: str) -> list[np.ndarray]:
        self.stats["queries"] += 1
        return self._inner._query_server(obs, instruction)

    def _pop(self) -> np.ndarray:
        raw = np.asarray(self._queue.pop(0), dtype=np.float32)
        adapter = self._inner.action_adapter
        if adapter is not None:
            return np.asarray(adapter(raw), dtype=np.float32)
        return raw

    def _freezing(self, ready_at: int) -> bool:
        """Whether this step is spent waiting on the policy rather than moving.

        The episode's first fetch is exempt in every mode: at reset the arm is stationary and
        nothing has been commanded, so there is no pose to hold and no motion to interrupt.
        Every mode pays that cold start identically, so exempting it removes a constant from all
        arms rather than tilting the comparison.
        """
        return self._last_action is not None and self._step < ready_at

    def _hold(self) -> np.ndarray:
        """Command the arm to stay where it was told to go. This IS the freeze."""
        assert self._last_action is not None  # guaranteed by _freezing
        return self._last_action.copy()
