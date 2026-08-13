"""Paired A/B of the chunk-boundary policy in ManiSkill.

Answers the one question the hermetic tests in ``examples/yam/tests/`` cannot: executing
*spliced* chunks changes which rows the arm runs, so does it change what the arm accomplishes?

Runs the same seeds through the same task with the same policy under each boundary mode, so
the only variable is the boundary policy:

    blocking            stock: freeze for the round trip, then execute from row 0
    prefetch_splice     the port: query early, drop the rows that went stale in flight
    prefetch_nosplice   prefetch without the splice (what a partial port ships)

Usage::

    python -m sim_eval.run_boundary_ab \
        --remote-url http://127.0.0.1:8202/act \
        --n-episodes 10 --latency-steps 4 --max-episode-steps 600

Latency is denominated in sim steps because simulation has no wall clock -- see
``sim_eval/inference/latency_client.py``. At the default 30 Hz control rate, the measured
135 ms thor->odin round trip is ~4 steps.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import logging
import math
from pathlib import Path
import statistics
import time
from typing import Any, Optional

import gymnasium as gym
import torch
import tyro
from tqdm import tqdm

import mani_skill.envs  # noqa: F401

from .inference.client import YAMClient
from .inference.latency_client import MODES, LatencyEmulatingClient
from .tasks import *  # noqa: F401, F403

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

DEFAULT_ENV = "BimanualYAMPutEverythingInBox-v1"
DEFAULT_INSTRUCTION = "put everything into the box"


@dataclass
class ABConfig:
    remote_url: str
    """URL of a running MolmoAct2 YAM serve (``host_server_yam.py``)."""

    env_id: str = DEFAULT_ENV
    instruction: str = DEFAULT_INSTRUCTION
    n_episodes: int = 10
    seed: int = 20260813

    latency_steps: int = 4
    """Round trip in sim steps. 4 steps at 30 Hz = 133 ms, the measured thor->odin act."""

    lead: Optional[int] = None
    """Rows before the chunk ends to fire the prefetch. Default: latency_steps + 1."""

    exec_steps: Optional[int] = None
    """Rows to execute per chunk. None consumes the whole chunk."""

    modes: tuple[str, ...] = MODES

    max_episode_steps: int = 600
    control_freq: int = 30
    sim_freq: int = 150
    shader_pack: str = "minimal"
    """`minimal` is much faster than `rt-fast` and this experiment scores task success, not
    image quality. The policy still sees rendered RGB."""

    output: str = "sim_eval/outputs/boundary_ab.json"


def _scalar(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.item() if value.numel() == 1 else value.any().item()
    return value


def _run_episode(env, client, instruction: str, seed: int, max_steps: int) -> dict[str, Any]:
    obs, _ = env.reset(seed=seed)
    torch.manual_seed(seed)
    client.reset()

    success = False
    steps = 0
    for steps in range(max_steps):
        action = client.infer(obs, instruction)
        obs, _reward, terminated, truncated, info = env.step(action)

        if bool(_scalar(info.get("success", False))):
            success = True
            break
        if bool(_scalar(terminated)) or bool(_scalar(truncated)):
            break

    return {
        "success": success,
        "steps": steps + 1,
        "seed": seed,
        **{f"stat_{k}": v for k, v in client.stats.items()},
    }


def _wilson(successes: int, total: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson 95% interval -- the same instrument the pHash rejection was judged with."""
    if total == 0:
        return (0.0, 0.0)
    p = successes / total
    denom = 1 + z**2 / total
    centre = (p + z**2 / (2 * total)) / denom
    margin = z * ((p * (1 - p) / total + z**2 / (4 * total**2)) ** 0.5) / denom
    return (max(0.0, centre - margin), min(1.0, centre + margin))


def required_episodes(p1: float, p2: float, power: float = 0.80) -> int:
    """Episodes per arm needed to resolve ``p1`` vs ``p2`` at alpha 0.05.

    Printed with every run because this harness was first used at n=60 against a ~7 % base
    success rate, where the honest requirement is closer to 500 -- and an underpowered run does
    not announce itself, it just returns a plausible-looking ordering.
    """
    z_alpha, z_beta = 1.96, 0.84 if power <= 0.80 else 1.28
    gap = abs(p1 - p2)
    if gap < 1e-9:
        return 0
    return int(math.ceil((z_alpha + z_beta) ** 2 * (p1 * (1 - p1) + p2 * (1 - p2)) / gap**2))


def _validity_warnings(summary: dict[str, dict[str, Any]], config: ABConfig) -> list[str]:
    """Reasons not to believe this run, computed from the run itself.

    Every one of these was a real defect in the first use of this harness, found only by
    noticing an impossible number by hand. They are checks now so the next run says so itself.
    """
    issues: list[str] = []

    # 1. Unequal re-query rates mean the arms differ by more than their boundary policy.
    rates = {m: v["queries_per_episode"] for m, v in summary.items() if v["episodes"]}
    if len(rates) > 1:
        lo, hi = min(rates.values()), max(rates.values())
        if lo > 0 and (hi - lo) / lo > 0.10:
            issues.append(
                f"arms re-query at different rates ({', '.join(f'{m}={r:.1f}' for m, r in rates.items())}"
                f"); spread {(hi - lo) / lo:.0%} > 10%. Splicing shortens each chunk, so with "
                "--exec-steps unset the splice arm replans more often and the comparison also "
                "contains the exec_steps lever. Pin --exec-steps across arms."
            )
    if config.exec_steps is None and len(rates) > 1:
        issues.append("--exec-steps is unset; arms consume different row counts per chunk.")

    # 2. Underpowered: the sample cannot resolve the differences it is being asked about.
    rated = {m: v["success_rate"] for m, v in summary.items() if v["episodes"]}
    if len(rated) > 1:
        best, worst = max(rated.values()), min(rated.values())
        need = required_episodes(worst, best)
        have = min(v["episodes"] for v in summary.values() if v["episodes"])
        if need > have:
            issues.append(
                f"underpowered: resolving {worst:.1%} vs {best:.1%} at 80% power needs ~{need} "
                f"episodes/arm; this run had {have}."
            )

    return issues


def main(config: ABConfig) -> None:
    env = gym.make(
        config.env_id,
        obs_mode="rgb",
        control_mode="pd_joint_pos",
        render_mode="rgb_array",
        max_episode_steps=config.max_episode_steps,
        reward_mode="none",
        sensor_configs=dict(shader_pack=config.shader_pack),
        sim_config=dict(sim_freq=config.sim_freq, control_freq=config.control_freq),
    )

    results: dict[str, list[dict[str, Any]]] = {}
    started = time.time()
    try:
        for mode in config.modes:
            inner = YAMClient(url=config.remote_url)
            client = LatencyEmulatingClient(
                inner,
                mode=mode,
                latency_steps=config.latency_steps,
                lead=config.lead,
                exec_steps=config.exec_steps,
            )
            episodes = []
            bar = tqdm(range(config.n_episodes), desc=mode, leave=False)
            for ep in bar:
                # Identical seeds across modes: this is a PAIRED comparison, so the arms differ
                # only in boundary policy, not in the scene they were handed.
                episode = _run_episode(
                    env,
                    client,
                    config.instruction,
                    seed=config.seed + ep,
                    max_steps=config.max_episode_steps,
                )
                episodes.append(episode)
                bar.set_postfix(success=sum(e["success"] for e in episodes))
            results[mode] = episodes
            wins = sum(e["success"] for e in episodes)
            lo, hi = _wilson(wins, len(episodes))
            logger.info(
                "%-18s success %d/%d (%.0f%%, CI %.0f-%.0f%%)  queries/ep=%.1f  "
                "frozen=%d skipped=%d late=%d",
                mode,
                wins,
                len(episodes),
                100 * wins / len(episodes),
                100 * lo,
                100 * hi,
                sum(e["stat_queries"] for e in episodes) / len(episodes),
                sum(e["stat_frozen_steps"] for e in episodes),
                sum(e["stat_rows_skipped"] for e in episodes),
                sum(e["stat_late_prefetches"] for e in episodes),
            )
    finally:
        env.close()

    summary = {}
    for mode, episodes in results.items():
        wins = sum(e["success"] for e in episodes)
        lo, hi = _wilson(wins, len(episodes))
        summary[mode] = {
            "episodes": len(episodes),
            "successes": wins,
            "success_rate": wins / len(episodes) if episodes else 0.0,
            "wilson95": [lo, hi],
            "median_steps": statistics.median([e["steps"] for e in episodes]) if episodes else 0,
            "frozen_steps_total": sum(e["stat_frozen_steps"] for e in episodes),
            "rows_skipped_total": sum(e["stat_rows_skipped"] for e in episodes),
            "queries_per_episode": sum(e["stat_queries"] for e in episodes) / len(episodes),
            "late_prefetches_total": sum(e["stat_late_prefetches"] for e in episodes),
            "queries_total": sum(e["stat_queries"] for e in episodes),
        }

    warnings = _validity_warnings(summary, config)
    for warning in warnings:
        logger.warning("VALIDITY: %s", warning)

    payload = {
        "schema": "molmoact2.yam.boundary-ab.v1",
        "validity_warnings": warnings,
        "config": {
            "env_id": config.env_id,
            "instruction": config.instruction,
            "n_episodes": config.n_episodes,
            "seed": config.seed,
            "latency_steps": config.latency_steps,
            "latency_ms_equivalent": 1000.0 * config.latency_steps / config.control_freq,
            "lead": config.lead,
            "exec_steps": config.exec_steps,
            "max_episode_steps": config.max_episode_steps,
            "control_freq": config.control_freq,
            "sim_freq": config.sim_freq,
            "remote_url": config.remote_url,
        },
        "elapsed_s": round(time.time() - started, 1),
        "summary": summary,
        "episodes": results,
    }
    out = Path(config.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    logger.info("wrote %s", out)


if __name__ == "__main__":
    main(tyro.cli(ABConfig))
