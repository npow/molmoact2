"""Prefetch + wall-clock splice for the YAM rollout loop.

The stock loop in ``launch_yaml_eval_molmoact.py`` asks the policy for a chunk and then
**blocks** on the round trip while the arm holds its last commanded target:

    if action_chunk is None or (step % chunk_size) == 0:
        obs_for_policy = env.get_obs()
        action_chunk = policy.inference(...)["actions"]   # arm frozen for the whole round trip

Measured on the thor->odin path that is ~135 ms of every ~1,135 ms cycle standing still --
11.9 % of wall-clock time -- and because the loop walks each chunk to its end before asking
again, the final row of every chunk acts on evidence up to ~1,102 ms old (mean action age
~618 ms, of which only 135 ms is the round trip; the other 483 ms is walking a one-second plan
to its end).

This module overlaps the round trip with motion, exactly as servo's
``servo/execution/chunked_loop.py`` does for SO-101, and it is a port of that mechanism rather
than a new idea.

WHY PREFETCH AND SPLICE ARE ONE FEATURE, NOT TWO
------------------------------------------------
MolmoAct2 emits **absolute joint poses**, not deltas. A chunk prefetched at row ``t`` is a plan
whose row 0 describes where the arm was at ``t``, not where it will be when the chunk starts
executing. Handing that chunk to the arm unmodified commands it *backward* to a pose it has
already left, at the boundary, every time. The splice -- dropping the leading rows whose moment
has already passed -- is what makes the prefetch legal. **Never ship one without the other.**

WHAT THIS PORT CHANGES FROM SERVO'S VERSION
-------------------------------------------
Servo's loop is wall-clock paced: it holds a fixed ``fps`` and splices with
``skip = (now - t_obs) * fps``. The YAM loop has **no fixed cadence** -- ``dynamic_smoothing``
interpolates to each target in 0.01 rad sub-steps, so a step lasts as long as the motion takes.
Using a nominal rate here would mis-splice whenever the arm moved fast or slow.

So the splice divides by a **measured** inter-row period (an EMA of observed step durations)
instead of an assumed one. That is self-consistent: the plan is executed at the loop's natural
rate, so rows are retired at that same rate, so that is the rate at which they go stale.

``dynamic_smoothing`` is also, incidentally, the safety layer this loop already had: a large
commanded delta becomes more interpolation sub-steps rather than a snap. This module preserves
it untouched and adds a jump guard *above* it as a second, explicit line of defence.
"""

from __future__ import annotations

from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
import logging
import threading
import time
from typing import Any, Callable, Optional, Sequence

import numpy as np

log = logging.getLogger(__name__)

# A prefetch that has not landed within this long is treated as wedged rather than late: the
# loop stops waiting on it and falls back to a blocking fetch. Generous because a cold serve
# can legitimately take tens of seconds; this only needs to be shorter than "forever".
PREFETCH_TIMEOUT_S = 90.0

# Until a run has measured its own step rate, the splice has nothing to divide by. Seeded from
# the YAM manifest's native control rate (10 Hz) and replaced by measurement after the first
# few steps.
DEFAULT_STEP_PERIOD_S = 0.1

# How much of the new sample each step-period update takes. Slow enough that one unusually long
# step (a big move, a camera hiccup) cannot swing the splice.
_STEP_PERIOD_ALPHA = 0.2


@dataclass
class ChunkRecord:
    """What one chunk actually did, for post-run attribution."""

    index: int
    prefetched: bool
    rows_returned: int
    rows_skipped: int
    rows_executed: int
    inference_ms: float
    # Wall-clock gap between the last command of the previous chunk and the first of this one.
    # This is the number the whole port exists to move: ~135 ms blocking, ~0 ms overlapped.
    boundary_gap_ms: float
    # Age of the observation this chunk was predicted from, at the moment its first row is sent.
    observation_age_ms: float
    # Rows before the end of the chunk at which the next prefetch was fired, and how long the
    # loop still had to block for it. A non-zero wait means the lead did not cover the round
    # trip -- the freeze is back, and this is the only place that says so.
    lead_used: int = 0
    prefetch_wait_ms: float = 0.0
    guard_tripped: bool = False
    fallback_reason: Optional[str] = None


@dataclass
class ChunkedRolloutReport:
    chunks: list[ChunkRecord] = field(default_factory=list)
    steps: int = 0
    aborted: Optional[str] = None

    def boundary_gap_p95_ms(self) -> Optional[float]:
        gaps = [c.boundary_gap_ms for c in self.chunks[1:]]
        if not gaps:
            return None
        return float(np.percentile(gaps, 95))

    def mean_observation_age_ms(self) -> Optional[float]:
        ages = [c.observation_age_ms for c in self.chunks]
        return float(np.mean(ages)) if ages else None

    def prefetch_rate(self) -> Optional[float]:
        if not self.chunks:
            return None
        return sum(c.prefetched for c in self.chunks) / len(self.chunks)

    def late_prefetch_rate(self) -> Optional[float]:
        """Fraction of prefetched chunks the loop still had to wait on.

        A prefetch that lands after its chunk has run out is a freeze wearing a prefetch's
        name. Watch this, not ``prefetch_rate``, when judging whether the port is working.
        """
        prefetched = [c for c in self.chunks if c.prefetched]
        if not prefetched:
            return None
        return sum(c.prefetch_wait_ms > 1.0 for c in prefetched) / len(prefetched)


class _Prefetcher:
    """Runs observe+infer on one background thread.

    Single worker on purpose. A second in-flight request would be a speculative act whose
    result is usually discarded, and on a single-GPU serve it competes with the one the loop is
    actually waiting for.
    """

    def __init__(self, fetch: Callable[[], dict[str, Any]]) -> None:
        self._fetch = fetch
        self._pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="yam-prefetch")
        self._inflight: Optional[Future[dict[str, Any]]] = None
        self._lock = threading.Lock()

    def submit(self) -> None:
        """Start a prefetch unless one is already pending OR already finished and unclaimed.

        The ``is not None`` test carries real weight. ``submit`` is called on every row from
        the trigger onward, so a guard of "not currently running" re-fires the moment the
        first prefetch *completes* -- discarding a finished chunk and starting a fresh round
        trip a row or two before ``take``, which restores the exact freeze this module removes
        (measured: 75 ms of a 135 ms round trip, on a lead that was correctly sized).

        A completed-but-unclaimed prefetch does get staler while it waits. That is the splice's
        job, and the splice is exact; re-fetching to chase freshness costs a guaranteed stall to
        buy a few rows of staleness the splice would have discarded anyway.
        """
        with self._lock:
            if self._inflight is not None:
                return
            self._inflight = self._pool.submit(self._fetch)

    def take(self, timeout_s: float) -> Optional[dict[str, Any]]:
        """Return the prefetched result, or None if there is none / it failed.

        A failed prefetch is logged and swallowed: the caller's contract is that a None here
        means "fetch it yourself, blocking", which is always safe.

        The returned dict carries ``waited_ms`` -- how long the loop blocked here. That is the
        diagnostic for an undersized lead: a prefetch that lands *after* the chunk runs out
        re-introduces exactly the freeze this module exists to remove, and it does so silently
        unless the wait is measured.
        """
        with self._lock:
            future, self._inflight = self._inflight, None
        if future is None:
            return None
        waiting_from = time.monotonic()
        try:
            result = future.result(timeout=timeout_s)
        except TimeoutError:
            log.warning("prefetch wedged past %.0fs; falling back to a blocking fetch", timeout_s)
            return None
        except Exception:
            log.exception("prefetch failed; falling back to a blocking fetch")
            return None
        result["waited_ms"] = (time.monotonic() - waiting_from) * 1000.0
        return result

    def shutdown(self) -> None:
        self._pool.shutdown(wait=False, cancel_futures=True)


def adaptive_lead(
    *,
    round_trip_s: Optional[float],
    step_period_s: Optional[float],
    lead_margin: int,
    rows_total: int,
    pinned: Optional[int] = None,
) -> int:
    """Rows before the end of a chunk at which the next prefetch should fire.

    The lead must cover one round trip measured in rows, which needs both the round trip and
    the loop's own step period -- and neither is knowable before the run. Until they are
    measured this returns a small conservative lead; ``pinned`` overrides everything, for A/B.

    Bounded strictly below ``rows_total``: a lead as long as the chunk fires at row 0, which
    would prefetch from an observation exactly as stale as the one it is replacing.
    """
    if pinned is not None:
        chosen = pinned
    elif round_trip_s is None:
        chosen = lead_margin + 1
    else:
        period = step_period_s or DEFAULT_STEP_PERIOD_S
        chosen = int(np.ceil(round_trip_s / max(period, 1e-6))) + lead_margin
    return int(max(1, min(chosen, max(1, rows_total - 1))))


def splice_index(
    *,
    observed_at: float,
    now: float,
    step_period_s: float,
    rows_returned: int,
) -> int:
    """How many leading rows of a prefetched chunk have already been overtaken by wall clock.

    Row ``k`` of a chunk predicted from an observation at ``observed_at`` describes the world at
    ``observed_at + k * step_period_s``. Any row whose moment is already in the past would
    command the arm backward, so it is dropped.

    Always leaves at least one row: an entirely-stale chunk still has to command *something*,
    and its last row is the closest thing to a current target that the plan contains. The caller
    sees this as a large ``rows_skipped`` and the jump guard remains the safety net.
    """
    if rows_returned <= 0:
        raise ValueError("cannot splice an empty chunk")
    if step_period_s <= 0:
        raise ValueError("step period must be positive")
    elapsed = max(0.0, now - observed_at)
    return int(min(max(0, round(elapsed / step_period_s)), rows_returned - 1))


def run_chunked_rollout(
    *,
    observe: Callable[[], Any],
    infer: Callable[[Any], Sequence[Any]],
    execute: Callable[[np.ndarray, int], Optional[str]],
    current_joints: Callable[[], np.ndarray],
    max_steps: int,
    exec_steps: Optional[int] = None,
    lead: Optional[int] = None,
    lead_margin: int = 2,
    max_jump: Optional[float] = None,
    prefetch: bool = True,
) -> ChunkedRolloutReport:
    """Drive a rollout, overlapping each round trip with the arm's motion.

    Pure callables, no ``RobotEnv`` and no policy type, so the whole mechanism is testable on
    recorded chunks with no hardware and no GPU. ``launch_yaml_eval_molmoact.py`` supplies the
    real ones.

    Args:
        observe: capture the observation to predict from. Called on the prefetch thread.
        infer: run the policy; returns the action chunk. Called on the prefetch thread.
        execute: send one row. Receives ``(action, step_index)`` and returns a non-None string
            to end the rollout (the eval loop's ``y``/``n``/``q`` keypresses).
        current_joints: the arm's actual pose right now, for the jump guard. Read on the main
            thread only.
        exec_steps: rows to execute per chunk before re-querying. ``None`` runs the whole chunk,
            which is the stock behaviour.
        lead: rows before the end of a chunk at which to fire the prefetch. ``None`` (default)
            sizes it from the measured round trip, which is the only way to get it right: the
            lead must cover one round trip, and a lead that is too short silently restores the
            boundary freeze while still reporting itself as "prefetched". Pass an int to pin it
            for an A/B.
        lead_margin: extra rows added to the adaptive lead, absorbing round-trip jitter.
        max_jump: abort if a commanded row differs from the arm's actual pose by more than this
            (same units as the action space -- radians for YAM). ``None`` disables the guard and
            logs that it is disabled.
        prefetch: ``False`` reproduces the stock blocking loop exactly, for A/B.
    """
    if lead is not None and lead < 1:
        raise ValueError("lead must be at least 1 row")
    if lead_margin < 0:
        raise ValueError("lead margin cannot be negative")

    if max_jump is None:
        # Per the repo's own rule: a bracket that has never been characterised for this
        # model/arm pair must not be silently inherited from another one. Say so, loudly.
        log.warning(
            "jump guard DISABLED (max_jump=None): no characterised bracket for this "
            "model/arm pair. dynamic_smoothing's interpolation remains the only limiter."
        )

    report = ChunkedRolloutReport()
    fetcher: Optional[_Prefetcher] = None

    def fetch() -> dict[str, Any]:
        t_obs = time.monotonic()
        observation = observe()
        t0 = time.monotonic()
        actions = infer(observation)
        finished = time.monotonic()
        return {
            "actions": list(actions),
            "t_obs": t_obs,
            "inference_ms": (finished - t0) * 1000.0,
            # Observe + infer. This, not inference alone, is what the lead has to cover.
            "round_trip_s": finished - t_obs,
            "waited_ms": 0.0,
        }

    if prefetch:
        fetcher = _Prefetcher(fetch)

    # Both of these stay None until measured, and the first measurement REPLACES the seed
    # rather than blending with it. Blending a guess into an EMA means the lead is sized
    # against a fictional cadence for the first several chunks -- which is where an
    # undersized lead does its damage, since that is when the loop has no history to correct
    # with. Only later samples are smoothed.
    step_period_s: Optional[float] = None
    round_trip_ema_s: Optional[float] = None
    step = 0
    chunk_index = 0
    last_send_t: Optional[float] = None

    try:
        while step < max_steps:
            # ---- acquire a chunk -------------------------------------------------------
            fallback_reason: Optional[str] = None
            result = fetcher.take(PREFETCH_TIMEOUT_S) if fetcher is not None else None
            was_prefetched = result is not None
            if result is None:
                if fetcher is not None and chunk_index > 0:
                    fallback_reason = "prefetch unavailable"
                result = fetch()

            actions = result["actions"]
            if not actions:
                report.aborted = "policy returned an empty chunk"
                break

            observed_rt = result.get("round_trip_s")
            if observed_rt is not None and observed_rt > 0:
                round_trip_ema_s = (
                    observed_rt
                    if round_trip_ema_s is None
                    else round_trip_ema_s + _STEP_PERIOD_ALPHA * (observed_rt - round_trip_ema_s)
                )

            # ---- splice ----------------------------------------------------------------
            now = time.monotonic()
            # A chunk fetched blocking was predicted from a fresh observation of a stationary
            # arm, so none of its rows are stale and it executes from row 0. Only a chunk that
            # was in flight *while the arm moved* needs the splice.
            effective_period = step_period_s or DEFAULT_STEP_PERIOD_S
            skip = (
                splice_index(
                    observed_at=result["t_obs"],
                    now=now,
                    step_period_s=effective_period,
                    rows_returned=len(actions),
                )
                if was_prefetched
                else 0
            )
            budget = len(actions) - skip if exec_steps is None else exec_steps
            rows = actions[skip : skip + max(1, budget)]

            boundary_gap_ms = (
                0.0 if last_send_t is None else ((now - last_send_t) - effective_period) * 1000.0
            )
            record = ChunkRecord(
                index=chunk_index,
                prefetched=was_prefetched,
                rows_returned=len(actions),
                rows_skipped=skip,
                rows_executed=0,
                inference_ms=result["inference_ms"],
                boundary_gap_ms=boundary_gap_ms,
                observation_age_ms=(now - result["t_obs"]) * 1000.0,
                prefetch_wait_ms=float(result.get("waited_ms", 0.0)),
                fallback_reason=fallback_reason,
            )

            # ---- execute, firing the next prefetch mid-chunk ----------------------------
            # Fire before the last `lead` rows so the round trip overlaps real motion. A chunk
            # spliced down to a single row would otherwise never reach its trigger index and
            # overlap would die silently in exactly the regime that needs it most.
            # The lead is re-derived on every row rather than fixed at chunk start. The step
            # period is unmeasurable until the loop has actually stepped, so a lead computed
            # once at chunk start is sized against the seeded cadence -- and the first chunk is
            # exactly where that seed is most wrong (measured: the first prefetch landed 15 ms
            # late at a 3x-wrong seed). Re-deriving lets it use a rate measured rows ago.
            ended: Optional[str] = None

            for i, row in enumerate(rows):
                if fetcher is not None:
                    lead_i = adaptive_lead(
                        round_trip_s=round_trip_ema_s,
                        step_period_s=step_period_s,
                        lead_margin=lead_margin,
                        rows_total=len(rows),
                        pinned=lead,
                    )
                    # A chunk spliced down to a single row must fire on that row, or its
                    # trigger index is unreachable and overlap dies silently in exactly the
                    # regime (a long round trip) that needs it most.
                    if len(rows) == 1 or i >= max(1, len(rows) - lead_i):
                        record.lead_used = lead_i
                        fetcher.submit()

                action = np.asarray(row, dtype=float)
                if max_jump is not None:
                    # Guard against the ACTUAL pose, not the last commanded one. A guard that
                    # references the last command diverges from reality whenever the arm is
                    # rate-limited, and then trips on its own bookkeeping instead of on danger.
                    delta = float(np.abs(current_joints() - action).max())
                    if delta > max_jump:
                        record.guard_tripped = True
                        report.aborted = (
                            f"jump guard: chunk {chunk_index} row {i} commands {delta:.3f} "
                            f"> max_jump {max_jump:.3f}"
                        )
                        break

                t_send = time.monotonic()
                ended = execute(action, step)
                if last_send_t is not None:
                    # Measured, not assumed -- this is what the splice divides by.
                    observed_period = t_send - last_send_t
                    if 0.0 < observed_period < 5.0:
                        step_period_s = (
                            observed_period
                            if step_period_s is None
                            else step_period_s
                            + _STEP_PERIOD_ALPHA * (observed_period - step_period_s)
                        )
                last_send_t = t_send

                record.rows_executed += 1
                step += 1
                if ended is not None or step >= max_steps:
                    break

            report.chunks.append(record)
            report.steps = step
            chunk_index += 1

            if report.aborted is not None:
                break
            if ended is not None:
                report.aborted = None if ended == "done" else ended
                break
    finally:
        if fetcher is not None:
            fetcher.shutdown()

    return report
