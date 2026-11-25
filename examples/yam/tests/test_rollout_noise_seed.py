import sys
from threading import Lock
from types import SimpleNamespace

import numpy as np
import pytest

import managed_rollout
import rtc_rollout

# A control plane whose checkpoints declare no semantics, so the rollout falls back to the family tables.
UNDECLARED_HTTP = SimpleNamespace(
    post=lambda path, json=None, timeout=None: SimpleNamespace(raise_for_status=lambda: None,
                                                               json=lambda: {"ioSpec": {"stateDim": 7}}),
    close=lambda: None)


@pytest.mark.parametrize("family", ["pi05", "flux3-action", "longwam"])
def test_seeded_probe_checks_both_prediction_paths(family):
    requests = []
    observations = []

    def predict(**kwargs):
        requests.append(kwargs)
        return SimpleNamespace(actions=np.full((15, 7), 0.5, dtype=np.float32))

    host = SimpleNamespace(
        identity={"state_dim": 7, "action_horizon": 15, "action_dim": 7, "camera_mapping": {"top": "scene"}},
        supports_committed_actions=lambda: True,
        begin_episode=lambda: None,
        session=SimpleNamespace(
            predict=predict,
            supports_observe=True,
            observe=lambda **kwargs: observations.append(kwargs),
            grant=SimpleNamespace(io_spec=SimpleNamespace(cameras={"scene": (4, 4)})),
        ),
    )
    managed_rollout.probe(host, family, "go", noise_seed=2**64 - 1)

    first, continued = requests
    assert first["noise_seed"] == continued["noise_seed"] == 2**64 - 1
    assert first["episode_start"]
    assert "committed_actions" not in first
    np.testing.assert_array_equal(continued["committed_actions"], np.full((1, 7), 0.5))
    if family == "longwam":
        assert len(observations) == 1
        assert "noise_seed" not in observations[0]


@pytest.mark.parametrize("bad_prediction", [1, 2])
def test_flux_probe_rejects_non_finite_gripper_plan(bad_prediction):
    calls = []

    def predict(**kwargs):
        calls.append(kwargs)
        actions = np.zeros((16, 7), dtype=np.float32)
        actions[:, 6] = 1.0504
        if "committed_actions" in kwargs:
            assert kwargs["committed_actions"][0, 6] == 1.0
            actions[:1] = kwargs["committed_actions"]
        if len(calls) == bad_prediction:
            actions[-1, 6] = np.nan
        return SimpleNamespace(actions=actions)

    host = SimpleNamespace(
        identity=dict(state_dim=7, action_dim=7, action_horizon=16, camera_mapping={"top": "scene"}),
        session=SimpleNamespace(predict=predict, grant=SimpleNamespace(io_spec=SimpleNamespace(cameras={}))),
        supports_committed_actions=lambda: True,
    )
    with pytest.raises(SystemExit, match="finite"):
        managed_rollout.probe(host, "flux3-action", "go")
    assert len(calls) == bad_prediction


def test_praxis_probe_preserves_the_rotation_and_zero_seed():
    calls = []
    pose = np.array([.3, .1, .3, 0, 1, 0, -1, 0, 0, 0, 0, 1, .75], np.float32)
    actions = np.tile(pose, (24, 1))

    def predict(**kwargs):
        calls.append(kwargs)
        return SimpleNamespace(actions=actions)

    host = SimpleNamespace(identity={"state_dim": 8, "action_dim": 13, "action_horizon": 24,
                                     "camera_mapping": {"top": "exterior", "left": "wrist"}},
                           supports_committed_actions=lambda: True, begin_episode=lambda: None,
                           session=SimpleNamespace(predict=predict, grant=SimpleNamespace(
                               io_spec=SimpleNamespace(cameras={"exterior": (4, 4), "wrist": (4, 4)}))))
    managed_rollout.probe(host, "praxis1", "pick", noise_seed=0)
    assert [call["noise_seed"] for call in calls] == [0, 0]
    np.testing.assert_array_equal(calls[-1]["committed_actions"], actions[:1])


@pytest.mark.parametrize("family", ["gr00t", "isaac05", "molmoact2"])
def test_managed_cli_rejects_noise_seed_before_opening_session(family, monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["managed_rollout.py", "--noise-seed", "0"])
    monkeypatch.setattr(
        managed_rollout,
        "Orchestral",
        lambda: SimpleNamespace(
            deployments=SimpleNamespace(get=lambda _: SimpleNamespace(model_id=family)),
            _http=UNDECLARED_HTTP,
        ),
    )
    with pytest.raises(SystemExit) as error:
        managed_rollout.main()
    assert error.value.code == 2
    assert f"{family} does not support --noise-seed" in capsys.readouterr().err


def test_managed_pi_dispatch_forwards_parsed_seed(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["managed_rollout.py", "--noise-seed=0"])
    monkeypatch.setattr(
        managed_rollout,
        "Orchestral",
        lambda: SimpleNamespace(
            deployments=SimpleNamespace(get=lambda _: SimpleNamespace(model_id="pi05")),
            _http=UNDECLARED_HTTP,
        ),
    )
    dispatched_args = []
    monkeypatch.setattr(managed_rollout.runpy, "run_path", lambda *args, **kwargs: dispatched_args.extend(sys.argv))

    managed_rollout.main()

    assert dispatched_args[0].endswith("rtc_rollout.py")
    assert dispatched_args[-2:] == ["--noise-seed", "0"]


def test_pi_rtc_attaches_seed_to_each_queued_prediction(monkeypatch):
    requests = []
    ticks = 0

    def sleep(_seconds):
        nonlocal ticks
        ticks += 1

    state = np.zeros(14, dtype=np.float32)
    image = np.zeros((4, 4, 3), dtype=np.uint8)
    host = SimpleNamespace(
        identity={"state_dim": 7, "camera_mapping": {"top": "scene"}},
        policy_adapter="pi05_yam_red_cap",
        session=SimpleNamespace(clock=SimpleNamespace(now=lambda: ticks)),
        predict_inputs=lambda images, positions, **kwargs: ({"scene": images["top"], "state": positions}, positions),
    )
    env = SimpleNamespace(
        get_robot_state=lambda: {"joint_positions": state},
        get_obs=lambda: {"joint_positions": state, "front_camera_rgb": image},
    )
    mask = SimpleNamespace(active_arm_side="left", validate_state=lambda positions, **kwargs: positions)
    chunking = SimpleNamespace(
        observations=SimpleNamespace(put=lambda *, t_captured=None: None),
        actions=SimpleNamespace(lock=Lock()),
        action=lambda: None,
        should_replan=lambda: True,
        observe=lambda **kwargs: requests.append(kwargs),
        t_next=1,
        late=0,
        delays=[],
        k=[],
    )
    monkeypatch.setattr(rtc_rollout, "make_chunking", lambda *args, **kwargs: chunking)
    monkeypatch.setattr(rtc_rollout.time, "sleep", sleep)
    rtc_rollout.run(env, host, mask, control_hz=15, instruction="go", s_min=7, d_init=2, seconds=2,
                    telemetry=rtc_rollout.RolloutTelemetry(15), noise_seed=0)

    assert [request["noise_seed"] for request in requests] == [0, 0]
