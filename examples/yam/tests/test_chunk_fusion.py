"""Regressions for joining successive action chunks into one command stream."""
from concurrent.futures import Future
from types import SimpleNamespace

import numpy as np
import pytest

from chunk_fusion import MODES, ChunkPlan


def ramp(start, rows=16, step=0.01, gripper=0.0):
    """A chunk whose six joints all move `step` per row from `start`."""
    chunk = np.empty((rows, 7), dtype=np.float32)
    chunk[:, :6] = (start + step * np.arange(rows))[:, None]
    chunk[:, 6] = gripper
    return chunk


def drain(plan):
    rows = []
    while plan:
        rows.append(plan.pop())
    return rows


def run_stream(mode, offset, *, blend_rows=4):
    """Command 8 rows of one chunk, then add a second that disagrees with it by `offset`."""
    plan = ChunkPlan(mode, blend_rows=blend_rows)
    plan.add(ramp(0.0), 0)
    sent = [plan.pop()[1] for _ in range(8)]
    plan.add(ramp(0.08 + offset, gripper=1.0), 8)
    return np.array(sent + [row for _, row in drain(plan)])


def test_splice_appends_the_new_chunk_after_the_queued_rows():
    plan = ChunkPlan("splice")
    assert plan.add(ramp(0.0), 0) == 0
    for _ in range(8):
        plan.pop()
    # Requested with 8 rows queued, and added once those rows have been commanded.
    request_slot = plan.slot
    for _ in range(8):
        plan.pop()
    assert plan.add(ramp(1.0), request_slot) == 8
    assert [index for index, _ in drain(plan)] == list(range(8, 16))


def test_rows_for_slots_already_commanded_are_discarded():
    plan = ChunkPlan("crossfade")
    plan.add(ramp(0.0), 0)
    for _ in range(3):
        plan.pop()
    assert plan.add(ramp(0.0), 0) == 3
    assert plan.add(ramp(0.0, rows=3), 0) is None


@pytest.mark.parametrize("mode", MODES)
def test_agreeing_chunks_give_the_same_stream_in_every_mode(mode):
    stream = run_stream(mode, offset=0.0)
    np.testing.assert_allclose(stream[:, 0], 0.01 * np.arange(len(stream)), atol=1e-5)


@pytest.mark.parametrize("mode", ["crossfade", "ensemble", "rail"])
def test_fusing_modes_shrink_the_step_onto_a_disagreeing_chunk(mode):
    offset = 0.3
    largest_step = np.abs(np.diff(run_stream(mode, offset)[:, 0])).max()
    assert largest_step < 0.6 * offset


@pytest.mark.parametrize("mode", ["crossfade", "rail"])
def test_ramped_modes_end_on_the_new_chunk(mode):
    stream = run_stream(mode, offset=0.3, blend_rows=4)
    expected = 0.38 + 0.01 * np.arange(16)
    # Slots 8..11 are the ramp; from slot 12 on the stream is the new chunk.
    np.testing.assert_allclose(stream[12:, 0], expected[4:], atol=1e-4)


def test_ensemble_averages_every_prediction_for_a_slot():
    stream = run_stream("ensemble", offset=0.3)
    # Two predictions cover slots 8..15, nearly equally weighted; one covers slots 16 on.
    np.testing.assert_allclose(stream[8:16, 0], 0.01 * np.arange(8, 16) + 0.15, atol=1e-3)
    np.testing.assert_allclose(stream[16:, 0], 0.38 + 0.01 * np.arange(8, 16), atol=1e-5)


def test_rail_replaces_a_noisy_chunk_with_its_cubic_fit():
    noisy = ramp(0.0)
    noisy[:, :6] += np.random.default_rng(0).normal(0, 0.02, (16, 6)).astype(np.float32)
    plan = ChunkPlan("rail")
    plan.add(noisy, 0)
    smooth = np.array([row for _, row in drain(plan)])
    assert np.abs(np.diff(smooth[:, :6], n=2, axis=0)).max() < 0.25 * np.abs(np.diff(noisy[:, :6], n=2, axis=0)).max()


def test_rail_bridge_keeps_velocity_continuous():
    stream = run_stream("rail", offset=0.3, blend_rows=6)[:, 0]
    acceleration = np.abs(np.diff(stream, n=2))
    # A hard splice would show the whole 0.3 rad offset as one acceleration spike.
    assert acceleration.max() < 0.1


@pytest.mark.parametrize("mode", ["crossfade", "ensemble", "rail"])
def test_gripper_follows_the_newest_chunk(mode):
    stream = run_stream(mode, offset=0.3)
    np.testing.assert_array_equal(stream[:8, 6], 0.0)
    np.testing.assert_array_equal(stream[8:, 6], 1.0)


class _Session:
    """Returns chunks anchored on the state it is given, like the YAM joint checkpoints."""

    grant = SimpleNamespace(io_spec=SimpleNamespace(cameras={"top": (4, 4)}))

    def __init__(self):
        self.requests = []
        self.observations = []

    def observe(self, *, cameras, state, instruction):
        self.observations.append(cameras["top"][0, 0, 0])

    def predict(self, *, cameras, state, instruction, episode_start, committed_actions=None, **sensors):
        self.observe(cameras=cameras, state=state, instruction=instruction)
        self.requests.append((episode_start, committed_actions, sensors))
        chunk = np.tile(np.asarray(state, dtype=np.float32), (16, 1))
        chunk[:, :6] += 0.01 * np.arange(16)[:, None]
        if committed_actions is not None:
            n = len(committed_actions)
            chunk[:n] = committed_actions
            chunk[n:] = committed_actions[-1]
            chunk[n:, :6] += 0.01 * np.arange(1, len(chunk) - n + 1)[:, None]
        return SimpleNamespace(actions=chunk)


class _Env:
    def __init__(self):
        self.position = np.zeros(14, dtype=np.float32)
        self.commands = []
        self.frame = 0

    def get_obs(self):
        frame = np.full((4, 4, 3), self.frame % 256, dtype=np.uint8)
        self.frame += 1
        return {"joint_positions": self.position.copy(), "front_camera_rgb": frame}

    def get_robot_state(self):
        return {"joint_positions": self.position.copy()}

    def step_command_only(self, command, wait=False):
        self.position = np.asarray(command, dtype=np.float32).copy()
        self.commands.append(self.position[:7].copy())


class _Mask:
    def validate_state(self, state, name):
        return np.asarray(state, dtype=np.float32)

    def command_target(self, target, measured):
        return target


class _InlinePredictor:
    """Finishes each prediction before the loop's next tick, so a run is deterministic."""

    def __init__(self, *args, **kwargs):
        pass

    def submit(self, function, *args, **kwargs):
        future = Future()
        future.set_result(function(*args, **kwargs))
        return future

    def shutdown(self, **kwargs):
        pass


@pytest.mark.parametrize("mode", MODES)
def test_rollout_loop_commands_a_continuous_stream(mode, monkeypatch):
    rollout_loop = pytest.importorskip("rollout_loop")
    monkeypatch.setattr(rollout_loop, "ThreadPoolExecutor", _InlinePredictor)
    clock = iter(np.arange(0, 1000, 0.001))
    monkeypatch.setattr(rollout_loop.time, "monotonic", lambda: next(clock))
    monkeypatch.setattr(rollout_loop.time, "sleep", lambda seconds: None)
    env = _Env()
    rollout_loop.run(env, _Session(), _Mask(), seconds=0.5, instruction="go", shadow=False,
                      camera_bindings={"top": "front_camera_rgb"}, history_offsets=(),
                      chunk_fusion=mode, lead_steps=4)
    commands = np.array(env.commands)
    assert len(commands) > 40
    steps = np.diff(commands[:, 0])
    assert steps.min() > -1e-4 and steps.max() < 0.03


@pytest.mark.parametrize("bimanual,stateful", [(False, False), (True, False), (False, True)])
@pytest.mark.parametrize("noise_seed", [None, 0])
def test_rollout_commits_the_queued_targets_before_appending_a_suffix(bimanual, stateful, noise_seed, monkeypatch):
    rollout_loop = pytest.importorskip("rollout_loop")
    monkeypatch.setattr(rollout_loop, "ThreadPoolExecutor", _InlinePredictor)
    clock = iter(np.arange(0, 1000, 0.001))
    monkeypatch.setattr(rollout_loop.time, "monotonic", lambda: next(clock))
    monkeypatch.setattr(rollout_loop.time, "sleep", lambda seconds: None)
    env, session = _Env(), _Session()
    parked = np.array([0, 0, 0, 0, 0, 0, 1], dtype=np.float32) if bimanual else None
    rollout_loop.run(env, session, _Mask(), seconds=0.4, instruction="go", shadow=False,
                      camera_bindings={"top": "front_camera_rgb"}, history_offsets=(), lead_steps=4,
                      action_shape=(16, 14 if bimanual else 7), state_padding=parked,
                      timestamp_observations=bimanual, committed_actions=True, stream_observations=stateful,
                      noise_seed=noise_seed)
    assert session.requests[0][:2] == (True, None)
    _, prefix, sensors = session.requests[1]
    assert len(prefix) == 4
    np.testing.assert_array_equal(np.array(env.commands)[12:16], prefix[:, :7])
    np.testing.assert_allclose(np.diff(np.array(env.commands)[:, 0]), 0.01, atol=1e-6)
    if bimanual:
        np.testing.assert_array_equal(prefix[:, 7:], np.tile(parked, (4, 1)))
        assert sensors["timestamp"][0] > 0
    if stateful:
        assert session.observations == list(range(env.frame))
    for _, _, prediction_kwargs in session.requests:
        if noise_seed is None:
            assert "noise_seed" not in prediction_kwargs
        else:
            assert prediction_kwargs["noise_seed"] == noise_seed


def test_checked_chunk_clips_flux_gripper_overshoot_before_queueing():
    import rollout_loop

    raw = ramp(0)
    raw[:, 6] = np.linspace(0.9199, 1.0504, len(raw))
    session = SimpleNamespace(predict=lambda **kwargs: SimpleNamespace(actions=raw))

    def predict():
        chunk, _ = rollout_loop.checked_chunk(session, {}, np.zeros(7), "go", episode_start=True,
                                               model_label="flux3-action")
        return chunk

    chunk = predict()
    np.testing.assert_array_equal(chunk[:, :6], raw[:, :6])
    np.testing.assert_array_equal(chunk[:, 6], np.minimum(raw[:, 6], 1.0))
    assert raw[-1, 6] > 1.05
    for invalid in (1.061, -0.051):
        raw[-1, 6] = invalid
        with pytest.raises(ValueError, match="outside the accepted aperture range"):
            predict()


def test_checked_chunk_rejects_changed_committed_rows():
    rollout_loop = pytest.importorskip("rollout_loop")
    session = SimpleNamespace(predict=lambda **kwargs: SimpleNamespace(actions=np.zeros((16, 7))))
    with pytest.raises(ValueError, match="did not preserve the committed action prefix"):
        rollout_loop.checked_chunk(session, {}, np.zeros(7), "go", episode_start=False,
                                    committed_actions=np.ones((1, 7), np.float32))


def test_rollout_replans_if_the_safety_mask_changes_a_committed_command(monkeypatch):
    rollout_loop = pytest.importorskip("rollout_loop")
    monkeypatch.setattr(rollout_loop, "ThreadPoolExecutor", _InlinePredictor)
    clock = iter(np.arange(0, 1000, 0.001))
    monkeypatch.setattr(rollout_loop.time, "monotonic", lambda: next(clock))
    monkeypatch.setattr(rollout_loop.time, "sleep", lambda seconds: None)
    env, session = _Env(), _Session()

    class ClipOnce(_Mask):
        def command_target(self, target, measured):
            if len(env.commands) == 12:
                target[0] = measured[0]
            return target

    rollout_loop.run(env, session, ClipOnce(), seconds=0.3, instruction="go", shadow=False,
                      camera_bindings={"top": "front_camera_rgb"}, history_offsets=(),
                      lead_steps=4, committed_actions=True)
    assert session.requests[1][1].shape == (4, 7)
    assert session.requests[2][1] is None
