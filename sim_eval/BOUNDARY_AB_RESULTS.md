# Chunk-boundary A/B in simulation — inconclusive, and why

**Date**: 2026-08-13 | **Host**: odin (RTX PRO 6000) | **Serve**: `host_server_yam.py`,
`allenai/MolmoAct2-BimanualYAM` | **Task**: `BimanualYAMPutEverythingInBox-v1`, 800 steps,
30 Hz control | **Verdict**: **does not clear `--prefetch` for default-on**

## What this was for

`examples/yam/chunked_rollout.py` removes the round-trip freeze from the YAM eval loop. Its
hermetic tests prove the mechanism (boundary gap 135 ms → 0 ms, no backward commands) but say
nothing about whether executing *spliced* chunks changes task success. This harness was built to
answer that in sim, since thor is offline.

It did not answer it. The run is recorded here so the next attempt starts from the defects
rather than rediscovering them.

## Results (n=60 per arm, 133 ms injected latency)

| arm | success | Wilson 95% CI | queries/ep |
|---|---:|---|---:|
| no latency (ceiling) | 4/60 — 6.7% | 3–16% | 26.6 |
| `blocking` (stock) | 1/60 — 1.7% | 0–9% | 24.0 |
| `prefetch_nosplice` | 3/60 — 5.0% | 2–14% | 26.9 |
| `prefetch_splice` (the port) | 7/60 — 11.7% | 6–22% | **31.6** |

Fisher exact, every pair: **no comparison reaches p < 0.05.** Closest is blocking vs splice at
**p = 0.061**.

## Why the result cannot be believed

**1. The spliced arm beat the zero-latency ceiling.** 11.7% against 6.7%. A latency-injected arm
cannot outperform no-latency; the ordering is sampling noise, and the noise floor exceeds the
effect being chased.

**2. The design was confounded, and the query counts prove it.** Splicing discards leading rows,
so each chunk yields fewer executable rows, so the splice arm **re-queried 31.6 times per episode
against the ceiling's 26.6** — 19% more replanning. That is the `exec_steps` freshness lever
(lever 12 in the act-latency report) mixed into a comparison that claimed to isolate the splice.
Any apparent benefit is unattributable.

*Fix*: pin `--exec-steps` across all arms. `run_boundary_ab` now emits a validity warning when
re-query rates differ by more than 10%, and when `--exec-steps` is unset.

**3. Underpowered, and the requirement depends on which pair you care about.** There is no single
"episodes needed" number here:

| comparison | base rates | episodes/arm at 80% power |
|---|---|---:|
| blocking vs splice | 1.7% vs 11.7% | **95** |
| ceiling vs splice | 6.7% vs 11.7% | **>500** |

The second is the one that decides whether the port *recovers* anything, and it is the expensive
one. `run_boundary_ab` now computes and reports this per run.

**4. The discontinuity diagnostics were removed because they measured nothing.** Three
generations, all failures, recorded so they are not rebuilt:

- a "backward commands" sign-flip counter fired 43/48/36 across all three modes — it was counting
  ordinary per-joint oscillation;
- a commanded-delta direction metric scored the *blocking* arm perfectly, because during a freeze
  the command is held constant, its delta is exactly zero, and no samples are recorded at all —
  a metric that flatters the mode standing still;
- a qpos-referenced version gave 61–71% reversals and 300–427× reach in **every** mode, including
  the zero-latency ceiling, discriminating nothing.

Task success carried the entire result, and it was not enough.

## The one number that held up

`blocking` spent **11.5% of all steps frozen**, reproduced independently in both batches
(1,840/16,000 and 3,678/32,000). The act-latency report derives **11.9%** of wall clock frozen
from the deployed code by a completely different route. The injection here was calibrated only
from the round trip, so this is genuine corroboration that the freeze is real and the sim model
of it is faithful.

## Is this task a usable gate at all?

Probably not, and that is the most useful finding. Zero-shot MolmoAct2 succeeds ~7% of the time
on this task; a gate whose baseline is 4/60 needs hundreds of episodes per arm to resolve
anything, at ~17 s/episode and with odin's GPU occupied throughout. **60 paired episodes on real
hardware carry far more signal per episode than 500 here.**

## If you run it again

```bash
python examples/yam/host_server_yam.py --host 127.0.0.1 --port 8202 --device cuda:0
python -m sim_eval.run_boundary_ab \
    --remote-url http://127.0.0.1:8202/act \
    --n-episodes 500 --exec-steps 20 --latency-steps 4 --max-episode-steps 800
```

`--exec-steps` is not optional for a valid comparison. Run the zero-latency ceiling
(`--latency-steps 0 --modes blocking`) at the same n, or the other arms have nothing to be good
or bad relative to — omitting it was the original error here.
