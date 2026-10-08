# Bimanual YAM — MolmoAct2 closed-loop eval

This directory holds the **robot/client side** of MolmoAct2 on a bimanual YAM
setup. Inference itself runs either on Servo (a managed hosted deployment of
the [`allenai/MolmoAct2-BimanualYAM`](https://huggingface.co/allenai/MolmoAct2-BimanualYAM)
checkpoint, reached over one official Servo action session), in a self-hosted
`host_server_yam.py` process on this LAN, or in-process from the checkpoint.
The eval launcher here drives the two YAM arms, captures the 3-camera
observation, queries a policy, executes the returned action chunk, records each
rollout, and (optionally) converts a labeled session into a LeRobot v3.0
dataset.

It is vendored and trimmed from the reference YAM implementation at
<https://github.com/williamtsai726/YAM> — only the eval-relevant pieces are
kept (teleop, data collection, and the Gello leader-arm code are omitted).

> Hardware-coupled example: it talks to real YAM arms over CAN (via `i2rt`) and
> Intel RealSense cameras. It is meant to run on the workstation wired to the
> robot, not in the dependency-light server environment.

## Layout

```
examples/yam/
├── host_server_yam.py            # self-hosted inference server (separate; see top-level README §5)
├── launch_yaml_eval_molmoact.py  # eval launcher — main entry point
├── molmoact_client.py            # MolmoActServo (Servo session) + MolmoActHTTP + MolmoActLocal policies
├── servo_session_bridge.py       # official Servo SDK session host + out-of-process bridge
├── camera_server.py              # long-lived ZMQ server owning the 3 RealSense cams
├── camera_client.py              # ZMQ client + standalone live viewer
├── eval_utils.py                 # per-rollout saver, cv2 viewer, labeling, conversion
├── rerun_rollout.py               # saved rollout -> offline Rerun .rrd playback
├── rerun_export_watchdog.py       # detached crash-resilient .rrd exporter
├── view_rollout.sh                # open an .rrd on a network-accessible web viewer
├── lerobot_convert.py            # raw rollouts -> LeRobot v3.0 dataset
├── start_camera_server.sh        # convenience launcher for camera_server.py
├── requirements.txt
├── configs/
│   ├── yam_left.yaml             # cameras, storage, eval, lerobot + left arm
│   └── yam_right.yaml            # right arm only
└── gello_min/                    # trimmed YAM runtime (robot/env/camera drivers)
```

## Install

```bash
pip install -r examples/yam/requirements.txt
# Plus the two non-PyPI deps (see requirements.txt):
#   i2rt    — YAM CAN/motor driver (required)
#   lerobot — only for the optional dataset conversion
```

`server` mode additionally needs the official `servo-client` SDK, which
requires Python >= 3.12 — see [Servo action sessions](#servo-action-sessions)
below.

Run every command below **from the molmoact2 repo root** (the scripts add
`examples/yam/` to `sys.path`, so `gello_min` and the sibling modules resolve).

## Inference modes

Set `eval.mode` in the rollout YAML, or override it per run with
`--policy-mode`:

- **`server`** — hosted policy over **one official Servo action session**. The
  SDK resolves a managed Servo deployment, opens a single action session
  (signed offer + lease) for the whole run, and every action chunk rides that
  session's protobuf transport. Requires a managed deployment id
  (`--servo-deployment dep_...` or `eval.server.deployment`). This is the
  launcher's default when `eval.mode` is absent from the config.
- **`http`** — POST observations to a running `host_server_yam.py` on this LAN
  using the legacy `json_numpy` protocol. Point `eval.molmoact_server` (or
  `--molmoact-server`) at it (`host:port` or full URL; `/act` is appended).
  Start the server per the top-level README §5, e.g.
  `uv run python examples/yam/host_server_yam.py --port 8202`. Development path
  only: no authentication, no generation fencing, no control-plane record.
- **`local`** — load the checkpoint in-process via `transformers` (no server).
  Configure under `eval.local`. bf16 needs ~10–14 GB VRAM, fp32 ~26 GB.

The three policies are interchangeable behind the same
`prepare_input` -> `inference` interface; the launcher picks one from the
config/CLI. `configs/yam_left.yaml` ships `mode: http`; the physical-arm
configs ship `mode: local`. Pass `--policy-mode server` to take the hosted path
without editing a rollout YAML.

### Servo action sessions

`server` mode is pure official Servo SDK: `Servo()` ->
`deployments.get(...)` -> `deployment.policy(...)` ->
`sv.session(policy, observation_encoding="h264")`.
There is no custom endpoint, no bespoke wire format, and no client-side token
signing — the control plane owns authentication, the signed offer/lease, and
generation fencing.

**Servo identity.** On the robot computer, enroll the machine once with
`servo agent install --name yam-cell-01`. Approve the printed code from an
authenticated administrator computer with
`servo robot approve CODE --name yam-cell-01`. Keep the agent running on the
robot. The rollout uses `servo.Servo()` with no credential arguments: Servo
gets the robot identity through its local agent. On your own computer, use
`servo login` for account access and deployment management. See Servo's
[identity guide](https://servo.mintlify.app/guides/identity) and
[Python API](https://servo.mintlify.app/reference/python-api).

**Interpreter requirement.** The official SDK requires Python >= 3.12.
`servo_session_bridge.py` holds the SDK session code (`ServoSessionHost`) and
can also run as a small stdio helper process under a separate interpreter. If
`servo` is importable in the running interpreter, the session runs in-process;
otherwise you must name an interpreter that has it with `--servo-python`,
`eval.server.servo_python`, or the `SERVO_PYTHON` environment variable. An
explicitly named interpreter is always used, even when `servo` is importable
here. There is no silent fallback: if none of the three is set and `servo` is
not importable, `server`
mode fails immediately instead of choosing for you. The parent/child hop is a
local pipe only; everything that leaves the machine is SDK transport.

**One session per run.** The launcher opens the session right after policy
construction and *before* any motor can be enabled, so a bad credential,
deployment, or lease fails with the arms still cold. That same session serves
every action chunk of every rollout in the session, and is completed on exit —
`_shutdown_runtime()` releases the robot first and closes the policy second, so
the session's closing network round trip never delays motor release.

**What rides the wire.** Each request carries the three camera frames as raw
JPEG bytes through the SDK's `EncodedObservation` (Servo accepts JPEG or PNG;
there is **no base64 step**) plus a 14-float state vector
`[left arm (7), right arm (7)]`. Camera roles, where "left"/"right" mean
standing behind the arms and facing into the workspace:

| Device alias | Servo camera key | View |
| --- | --- | --- |
| `/dev/yam-cameras/middle` | `top` | middle/static workspace camera |
| `/dev/yam-cameras/left` | `left` | left wrist |
| `/dev/yam-cameras/right` | `right` | right wrist |

The instruction you type at the stdin prompt rides with every action request.
Responses are validated client-side: a 30x14 chunk, action space
`joint_position`, decoded, finite, and served by the same binding generation the
session opened against.

The full pi0.5 red-cap checkpoint is instead a native **15x7** single-arm YAM
policy. Reuse the standard pi0.5 hardware configs and select its physical
placement with `--active-arm-side`: when Servo advertises a 7-D contract, the
client sends that arm's seven live values and reinserts the returned actions
into the same half of the 14-D executor command. The opposite half is filled
from live feedback. All three cameras remain required. Gripper values stay in
the native YAM convention (`0 = closed`, `1 = open`); the client performs no
second inversion.

Managed and direct sessions default to **H.264**. The client passes raw RGB
pixels to the SDK, whose session-owned encoder maintains the video stream.
Use `--observation-encoding h264` to select it explicitly. Managed sessions
use the SDK's encoder settings; `--h264-crf` applies only to direct sessions.
`eval.server.image_size` defaults to `null` so the SDK/runtime owns preprocessing.
The example defaults to unseeded sampling. Explicit `--seed` values are
forwarded to the current Servo SDK.

**Launch a hosted run:**

```bash
PYTHONPATH=examples/yam \
/home/npow/molmoact2-venv/bin/python \
examples/yam/launch_yaml_eval_molmoact.py \
  --config-path examples/yam/configs/yam_left_physical.yaml \
  --right-config-path examples/yam/configs/yam_right_primary.yaml \
  --policy-mode server \
  --servo-deployment dep_xxxxxxxxxxxx \
  --observation-encoding h264 \
  --active-arm-side left \
  --execution-mode active_arm_hold \
  --num-rollouts 1
```

If the rollout interpreter already has Servo installed, no `--servo-python`
option is needed. Otherwise point `--servo-python` (or `SERVO_PYTHON`) at a
Python >= 3.12 interpreter with Servo installed.

The [pi0.5 red-cap checkpoint](https://huggingface.co/npow/pi05-yam-red-cap-full-7500)
is a single-active-arm policy: `middle`, `left`, and `right` RGB images at
224 × 224, a 7-value state, and 15 × 7 action chunks at **15 Hz**. The adapter
maps the overhead camera to `middle`, fits images to the declared geometry,
and holds the inactive arm. Select `--active-arm-side left` or `right`.
The pi0.5 hardware configs use 15 Hz; overriding them with `--control-hz 30`
executes the trajectory twice as fast as its training cadence.

**Launch the self-hosted full pi0.5 red-cap endpoint:**

```bash
PYTHONPATH=examples/yam \
/home/npow/molmoact2-venv/bin/python \
examples/yam/launch_yaml_eval_molmoact.py \
  --config-path examples/yam/configs/pi05_bimanual_physical.yaml \
  --right-config-path examples/yam/configs/pi05_right_primary.yaml \
  --policy-mode direct \
  --servo-grant ~/.config/servo/pi05-yam-full-7500-grant.json \
  --active-arm-side left \
  --control-hz 15 \
  --max-steps 600 \
  --num-rollouts 1
```

Selecting `--active-arm-side left` executes that arm by default. Keep the other
arm connected: its real feedback supplies the held half of the bimanual
executor rather than fabricating state. Shadow execution remains available
only when requested explicitly with `--execution-mode shadow`.

**State of play (2026-08-02).** The client side is complete, but a live run
also needs the *remote* Servo control plane to expose a managed MolmoAct2
deployment for the YAM embodiment with action sessions enabled on its runtime
artifact. That deployment does not exist yet: the catalog reachable with this
machine key still advertises the single-arm/SO-101 view, and runtime
`action_sessions_enabled` defaults to false. Until it is created, `server` mode
fails at session open (before any motor is enabled, by design). Use `http` or
`local` mode in the meantime.

### Metrics

Metrics for hosted (`server` mode) deployments are in the official Grafana at
https://grafana.orchestrallabs.ai (Clerk SSO). This repo collects no metrics itself.

## Hardware setup

1. Both YAM arms powered, e-stop released.
2. The 3 RealSense cameras plugged into USB 3, listed in the rollout config
   under `sensors.cameras`. **Order matters** — the model was trained on
   `[top, left, right]`; here `front_camera` plays the `top` role. The
   physical-arm configs address the cameras through the stable
   `/dev/yam-cameras/{middle,left,right}` aliases (udev rule in
   `examples/yam/udev/`, installed to `/etc/udev/rules.d/`) rather than
   `/dev/video*`; replug or hub changes are then harmless, but replacing a
   camera means updating that rule's serial.
3. Bring up CAN and set the camera/CAN interface names in the configs
   (`channel:` — find them with `ip link show`). Disable the motor watchdog so
   the arms don't collapse during long sessions (see the `i2rt` docs / your
   YAM bring-up scripts).
4. Some right-arm motors (4 and the gripper) boot with a latched DaMiao
   status `0x3` ("output-shaft calibration") even though their encoders are
   fine (verified 2026-08-13: positions stable through clear/enable,
   registers identical to healthy motors). The local `i2rt` driver now clears
   this automatically at startup — at most once per motor, requiring a clean
   enable reply and a position that is continuous across the clear, so a
   genuinely lost calibration still fails closed. If an arm refuses to start
   with a calibration error despite this, the fault is real: inspect the
   joint before retrying. `~/code/i2rt/scripts/yam_motor_preflight.py` checks
   all motors without starting a rollout.

## Run a session

Two terminals when the camera server is enabled (the default).

**Terminal A — camera server (long-lived):**

```bash
bash examples/yam/start_camera_server.sh
# or: python examples/yam/camera_server.py --config examples/yam/configs/yam_left.yaml
```

Wait for `REP bound on tcp://127.0.0.1:5555` / `PUB bound on tcp://127.0.0.1:5556`.
It holds the cameras warm across sessions and feeds the live viewer's PUB
stream so the cv2 window keeps repainting during inference.

**Terminal B — eval:**

```bash
python examples/yam/launch_yaml_eval_molmoact.py \
    --config_path       examples/yam/configs/yam_left.yaml \
    --right-config-path examples/yam/configs/yam_right.yaml \
    -n 10
```

`-n 10` runs 10 rollouts. Set `eval.camera_server.enabled: false` to open the
cameras in-process instead (one fewer terminal, but the viewer freezes during
inference).

## What happens per rollout

1. Arms interpolate to `agent.start_joints` — your cue to reset the workspace.
2. Stdin prompts for the task instruction (Enter reuses the previous one).
3. The rollout runs; a 3-pane cv2 window (`YAM Eval`) shows LEFT / FRONT / RIGHT.
4. End it by pressing a key **in the cv2 window**:
   - `y` → success, `n` → failure, `q` → quit (kept unlabeled under `eval/`)
   - or let it hit `max_steps` → you're prompted on stdin afterwards.

`Ctrl-C` is handled: the in-progress rollout is flushed with an `err.md`
marker and any rollouts already labeled this session are still converted.

## Where files land

Under `{storage.base_dir}/data/{storage.task_directory}/`:

```
eval/<ts>/                       # quit / unlabeled rollouts
success/<YYYY-MM-DD>/<ts>/
failure/<YYYY-MM-DD>/<ts>/
eval_lerobot_v30/<session_ts>/   # LeRobot v3.0 dataset (labeled rollouts, end of session)
```

Each rollout has `episode.h5` (joint trajectory + instruction) and one PNG per
camera per frame under `left_rgb/`, `front_rgb/`, `right_rgb/`.

## Post-rollout playback in Rerun

Rerun is an **offline** diagnostic artifact, not a live control dependency.
With `eval.rerun.enabled: true` (enabled in both physical-arm configs), the
launcher starts and verifies a detached exporter *before* it creates any robot
or camera resources. After the launcher exits—normally or due to an
exception—the exporter scans raw rollout directories and atomically publishes
`rollout.rrd` alongside each one. It never opens a camera, CAN interface, or
robot, so export cannot affect motion timing.

Normally a replay includes camera frames, encoder feedback, policy targets,
and action chunks from `episode.h5`. If the launcher dies before HDF5 was
durably written, the exporter still creates an explicitly marked camera-only
recovery RRD from synchronized PNGs; it does not invent lost telemetry. Its
per-rollout state is recorded in `rerun_export.status.json`.

Open the recording after the rollout finishes:

```bash
/home/npow/molmoact2-venv/bin/rerun \
  yam_eval_runs/data/red_lid_left_arm/eval/<timestamp>/rollout.rrd
```

Or use the helper to bind the browser viewer and recording server to every
network interface. It prints a complete LAN URL that opens the saved recording
directly (opening the bare port 9090 URL only shows Rerun's file picker):

```bash
examples/yam/view_rollout.sh \
  yam_eval_runs/data/red_lid_left_arm/eval/<timestamp>/rollout.rrd
```

The helper uses ports 9090 (viewer) and 9876 (recording) by default. Override
them with `RERUN_WEB_PORT` and `RERUN_GRPC_PORT`, or override the executable
with `RERUN_BIN`. Set `RERUN_PUBLIC_IP` when the browser reaches the machine
through a different hostname or address, such as Tailscale.

The recording starts with a three-camera timeline and includes:

- the instruction;
- encoder feedback before and after every command;
- the exact absolute-joint target sent to the arm and target-minus-feedback;
- the full 30-action plan at each policy replan, plus inference timing.

For a historical rollout made before Rerun was enabled, convert it after the
fact (it will show camera and encoder feedback; policy targets are shown only
when that older `episode.h5` recorded them):

```bash
PYTHONPATH=examples/yam /home/npow/molmoact2-venv/bin/python \
  examples/yam/rerun_rollout.py \
  yam_eval_runs/data/red_lid_left_arm/eval/<timestamp>
```

## Key config knobs (`configs/yam_left.yaml`)

| Key | Meaning |
|---|---|
| `eval.mode` | `server` (one Servo action session), `http` (self-hosted `host_server_yam.py`), or `local` (in-process). Defaults to `server` when unset. CLI: `--policy-mode`. |
| `eval.server.deployment` | Managed Servo deployment id (`dep_...`) for `mode: server`; required. CLI: `--servo-deployment`. |
| `eval.server.servo_python` | Python >= 3.12 interpreter with `servo-client`, used when this runtime cannot import `servo`. CLI: `--servo-python`; env `SERVO_PYTHON`. |
| `eval.server.observation_encoding` | Defaults to `h264`; raw RGB pixels reach the SDK session encoder. CLI: `--observation-encoding`. |
| `eval.server.image_size` | Optional client-side padded resize; default `null` so the Servo runtime owns preprocessing. |
| `eval.molmoact_server` | `http` mode only: address of the self-hosted `host_server_yam.py`. CLI: `--molmoact-server`. |
| `eval.local.*` | Checkpoint / device / dtype for `mode: local`. |
| `eval.camera_server.enabled` | `true` uses the ZMQ camera server; `false` opens cameras in-process. |
| `eval.live_view_enabled` | `false` disables the cv2 window (headless runs). |
| `eval.rerun.*` | Post-rollout Rerun export (`image_stride`, JPEG quality, policy chunk size). |
| `max_steps` | Per-rollout timeout in control steps. |
| `storage.*` | Output location, instruction, PNG save settings. |
| `lerobot.*` | End-of-session dataset conversion knobs. |

## Camera server, standalone

Sanity-check the cameras independently of the eval loop:

```bash
python examples/yam/camera_client.py --mode sub      # subscribe to the PUB stream
```
