"""The rollout measures where time goes from a worker response to the robot executing its first row."""
from concurrent.futures import Future
from types import SimpleNamespace

import numpy as np
import pytest

PREDICT_SECONDS = 0.100
IK_SECONDS = 0.030
FK_SECONDS = 0.002


class _Clock:
    """Simulated time: only the fakes and the loop's sleep move it forward."""

    def __init__(self):
        self.now = 0.0

    def monotonic(self):
        return self.now

    def sleep(self, seconds):
        self.now += max(0.0, seconds)


class _Session:
    grant = SimpleNamespace(io_spec=SimpleNamespace(cameras={"top": (4, 4)}))

    def __init__(self, clock):
        self.clock = clock
        self.observed = 0

    def observe(self, *, cameras, state, instruction):
        self.observed += 1

    def predict(self, *, state, **_kwargs):
        self.clock.now += PREDICT_SECONDS
        chunk = np.tile(np.asarray(state, dtype=np.float32), (16, 1))
        chunk[:, :6] += 0.01 * np.arange(16)[:, None]
        return SimpleNamespace(actions=chunk)


class _InlinePredictor:
    """Runs each submitted call at once, so a run is deterministic; ``running`` is set during the call."""

    running = False

    def __init__(self, *args, **kwargs):
        pass

    def submit(self, function, *args, **kwargs):
        future = Future()
        _InlinePredictor.running = True
        try:
            future.set_result(function(*args, **kwargs))
        except Exception as error:
            future.set_exception(error)
        finally:
            _InlinePredictor.running = False
        return future

    def shutdown(self, **kwargs):
        pass


class _PoseActions:
    """Stands in for a tool-pose adapter; the identity map, with the cost of FK and IK."""

    def __init__(self, clock, *, unreachable_calls=()):
        self.clock = clock
        self.solved = 0
        self.state_calls = 0
        self.state_calls_on_control_thread = 0
        self.unreachable_calls = set(unreachable_calls)

    def state(self, joints, *, matrix_policy=False):
        self.clock.now += FK_SECONDS
        self.state_calls += 1
        self.state_calls_on_control_thread += not _InlinePredictor.running
        return np.asarray(joints, dtype=np.float32)

    def actions(self, rows, start, *, matrix_policy=False):
        from tool_pose_actions import UnreachableToolPose

        self.clock.now += IK_SECONDS
        self.solved += 1
        if self.solved in self.unreachable_calls:
            raise UnreachableToolPose("test pose out of reach")
        return np.asarray(rows, dtype=np.float32)


class _Env:
    def __init__(self):
        self.position = np.zeros(14, dtype=np.float32)
        self.commands = 0

    def get_obs(self):
        return {"joint_positions": self.position.copy(), "front_camera_rgb": np.zeros((4, 4, 3), np.uint8)}

    def get_robot_state(self):
        return {"joint_positions": self.position.copy()}

    def step_command_only(self, command, wait=False):
        self.position = np.asarray(command, dtype=np.float32).copy()
        self.commands += 1


class _Mask:
    def validate_state(self, state, name):
        return np.asarray(state, dtype=np.float32)

    def command_target(self, target, measured):
        return target


class _Trace:
    def __init__(self):
        self.predictions = []  # PREDICTION_STAGES per chunk, in order
        self.commands = []  # COMMAND_STAGES per accepted chunk, in order
        self.events = []
        self.tick_ms = []

    def prediction(self, *args, **kwargs):
        pass

    def event(self, message, *args):
        self.events.append(message % args if args else message)

    def tick(self, *args, holding, tick_ms=None):
        if tick_ms is not None:
            self.tick_ms.append(tick_ms)

    def latency(self, stages, event=None):
        (self.commands if "response_to_command_ms" in stages else self.predictions).append(stages)
        if event is not None:
            self.events.append(event)


class _SpanRecorder:
    def __init__(self):
        self.spans = []

    def __call__(self, name, start_ns, end_ns, *, attributes=None):
        self.spans.append(SimpleNamespace(name=name, duration_ms=(end_ns - start_ns) / 1e6, attributes=attributes))

    def named(self, name):
        return [span for span in self.spans if span.name == name]


@pytest.fixture
def clock(monkeypatch):
    rollout_loop = pytest.importorskip("rollout_loop")
    clock = _Clock()
    monkeypatch.setattr(rollout_loop, "ThreadPoolExecutor", _InlinePredictor)
    monkeypatch.setattr(rollout_loop.time, "monotonic", clock.monotonic)
    monkeypatch.setattr(rollout_loop.time, "sleep", clock.sleep)
    return clock


@pytest.fixture
def spans(monkeypatch):
    from orchestral.core import tracing

    recorder = _SpanRecorder()
    monkeypatch.setattr(tracing, "record_span", recorder)
    return recorder


def _run(clock, *, session=None, pose=None, env=None, **kwargs):
    import rollout_loop

    trace = _Trace()
    rollout_loop.run(env or _Env(), session or _Session(clock), _Mask(), seconds=3, instruction="go",
                      camera_bindings={"top": "front_camera_rgb"}, history_offsets=(), lead_steps=4,
                      pose_actions=pose or _PoseActions(clock), trace=trace, max_plan_rows=8, model_label="rho",
                      **{"shadow": False, **kwargs})
    return trace


def test_checked_chunk_times_each_stage_and_keeps_ik_out_of_postprocess(clock):
    from rollout_loop import checked_chunk, command_latency, prediction_latency

    timings = {}
    submitted_at = clock.now
    clock.now += 0.004  # behind an earlier observe call on the predictor thread
    _, elapsed = checked_chunk(_Session(clock), {}, np.zeros(7, np.float32), "go", episode_start=True,
                               pose_actions=_PoseActions(clock), timings=timings)
    clock.now += 0.010  # waiting for the control loop
    accepted_at = clock.now
    clock.now += 0.001
    stages = {**prediction_latency(timings, submitted_at),
              **command_latency(timings, observed_at=0.0, accepted_at=accepted_at, commanded_at=clock.now)}
    assert elapsed == pytest.approx(FK_SECONDS + PREDICT_SECONDS + IK_SECONDS)
    assert stages["queue_ms"] == pytest.approx(4)
    assert stages["prepare_ms"] == pytest.approx(1000 * FK_SECONDS)
    assert stages["predict_ms"] == pytest.approx(1000 * PREDICT_SECONDS)
    assert stages["ik_ms"] == pytest.approx(1000 * IK_SECONDS)
    assert stages["postprocess_ms"] == pytest.approx(0)
    assert stages["ready_to_accept_ms"] == pytest.approx(10)
    assert stages["accept_to_command_ms"] == pytest.approx(1)
    assert stages["response_to_command_ms"] == pytest.approx(1000 * IK_SECONDS + 11)
    assert stages["observation_to_command_ms"] == pytest.approx(
        1000 * (FK_SECONDS + PREDICT_SECONDS + IK_SECONDS) + 15)


def test_joint_worker_has_no_ik_stage(clock):
    from rollout_loop import checked_chunk, prediction_latency

    timings = {}
    checked_chunk(_Session(clock), {}, np.zeros(7, np.float32), "go", episode_start=True, timings=timings)
    assert "ik_seconds" not in timings
    assert "ik_ms" not in prediction_latency(timings, submitted_at=0.0)


@pytest.mark.parametrize("mode", ["splice", "crossfade"])
def test_rollout_traces_every_prediction_and_each_first_command(clock, mode, capsys):
    env = _Env()
    trace = _run(clock, env=env, chunk_fusion=mode)
    assert len(trace.commands) >= 3
    assert len(trace.predictions) >= len(trace.commands)
    for stages in trace.predictions:
        assert stages["predict_ms"] == pytest.approx(1000 * PREDICT_SECONDS)
        assert stages["ik_ms"] == pytest.approx(1000 * IK_SECONDS)
    for prediction, command in zip(trace.predictions, trace.commands):
        assert command["response_to_command_ms"] == pytest.approx(
            prediction["ik_ms"] + prediction["postprocess_ms"] + command["ready_to_accept_ms"]
            + command["accept_to_command_ms"])
        assert command["observation_to_command_ms"] >= (
            prediction["queue_ms"] + prediction["prepare_ms"] + prediction["predict_ms"]
            + command["response_to_command_ms"])
    assert len(trace.tick_ms) == env.commands
    summary = capsys.readouterr().out
    assert f"over {len(trace.predictions)} predictions, {len(trace.commands)} commanded" in summary
    for line in ("queue_ms", "ik_ms", "response_to_command_ms", "tick_to_command_ms",
                 f"outcomes: accepted {len(trace.commands)}"):
        assert line in summary


def test_ik_of_a_dropped_chunk_is_still_measured_and_sent(clock, spans, capsys):
    trace = _run(clock, pose=_PoseActions(clock, unreachable_calls={2}))
    assert "Prediction 2 rejected: unreachable plan" in trace.events
    assert len(trace.predictions) == len(trace.commands) + 1
    assert trace.predictions[1]["ik_ms"] == pytest.approx(1000 * IK_SECONDS)
    dropped = [span for span in spans.named("yam.rollout.ik") if span.attributes["yam.rollout.prediction_id"] == 2]
    assert len(dropped) == 1
    assert dropped[0].attributes["yam.rollout.outcome"] == "unreachable"
    assert dropped[0].duration_ms == pytest.approx(1000 * IK_SECONDS, abs=1e-3)
    assert 2 not in [span.attributes["yam.rollout.prediction_id"]
                     for span in spans.named("yam.rollout.response_to_command")]
    assert "unreachable 1" in capsys.readouterr().out


def test_streamed_observation_kinematics_stay_off_the_control_tick(clock):
    pose, session = _PoseActions(clock), _Session(clock)
    trace = _run(clock, pose=pose, session=session, stream_observations=True)
    assert session.observed > len(trace.predictions)
    assert pose.state_calls >= session.observed + len(trace.predictions)
    assert pose.state_calls_on_control_thread == 0


@pytest.mark.parametrize("shadow", [False, True])
def test_each_prediction_is_a_grafana_span(clock, spans, shadow):
    session = _Session(clock)
    session._trace_attributes = lambda: {"orc.deployment_id": "dep-1", "orc.session_id": "s-1"}
    trace = _run(clock, session=session, shadow=shadow)
    ik, commanded = spans.named("yam.rollout.ik"), spans.named("yam.rollout.response_to_command")
    assert len(ik) == len(trace.predictions) and len(commanded) == len(trace.commands) >= 3
    for span in ik:
        assert span.duration_ms == pytest.approx(1000 * IK_SECONDS, abs=1e-3)
        assert span.attributes["yam.rollout.outcome"] == "accepted"
    for span, command in zip(commanded, trace.commands):
        assert span.duration_ms == pytest.approx(command["response_to_command_ms"], abs=1e-3)
        assert span.attributes["yam.rollout.ik_ms"] == pytest.approx(1000 * IK_SECONDS)
        assert span.attributes["yam.rollout.predict_ms"] == pytest.approx(1000 * PREDICT_SECONDS)
    for span in ik + commanded:
        assert span.attributes["orc.deployment_id"] == "dep-1"
        assert span.attributes["yam.rollout.family"] == "rho"
        assert span.attributes["yam.rollout.shadow"] is shadow
        assert span.attributes["yam.rollout.chunk_fusion"] == "splice"
    assert [span.attributes["yam.rollout.prediction_id"] for span in commanded] == list(
        range(1, len(commanded) + 1))


def test_a_failing_span_never_stops_the_rollout(clock, monkeypatch, capsys):
    from orchestral.core import tracing

    calls = []

    def broken(*args, **kwargs):
        calls.append(args)
        raise RuntimeError("exporter down")

    monkeypatch.setattr(tracing, "record_span", broken)
    trace = _run(clock)
    assert len(trace.commands) >= 3
    assert len(calls) == 1
    assert "Stopped recording latency spans: exporter down" in capsys.readouterr().out


def test_an_sdk_without_span_recording_keeps_the_rerun_trace(clock, monkeypatch, capsys):
    from orchestral.core import tracing

    monkeypatch.delattr(tracing, "record_span")
    trace = _run(clock)
    assert len(trace.commands) >= 3
    assert "cannot record spans" in capsys.readouterr().out


def test_a_short_truncated_plan_finishes_before_replanning_instead_of_missing_prefetch(clock, capsys):
    class TruncatingPoseActions(_PoseActions):
        def actions(self, rows, start, *, matrix_policy=False):
            solved = super().actions(rows, start, matrix_policy=matrix_policy)
            # First two calls return only 1 row (67 ms < 132 ms predict+IK), then full chunks.
            return solved[:1] if self.solved <= 2 else solved

    trace = _run(clock, pose=TruncatingPoseActions(clock))
    out = capsys.readouterr().out
    assert "Tool IK kept 1 of 8 rows." in out
    assert "missed_boundary" not in out
    assert len(trace.commands) >= 4
