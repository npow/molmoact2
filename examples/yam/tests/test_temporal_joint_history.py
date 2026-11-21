import numpy as np

from temporal_joint_history import TemporalJointHistory


def test_history_pairs_observations_with_preceding_commands():
    history = TemporalJointHistory(3)
    frame = np.zeros((2, 2, 3), dtype=np.uint8)
    state = np.arange(7, dtype=np.float32)
    history.observe({"scene": frame}, state)
    history.commanded(state + 10)
    frame.fill(1)
    history.observe({"scene": frame}, state + 1)
    cameras, tensors = history.inputs()
    np.testing.assert_array_equal(cameras["scene"][:, 0, 0, 0], [0, 0, 1])
    np.testing.assert_array_equal(tensors["state_history"][:, 0], [0, 0, 1])
    np.testing.assert_array_equal(tensors["command_history"][:, 0], [0, 0, 10])

    history.observe({"scene": frame}, state + 2)
    history.commanded(state + 20)
    _, next_tensors = history.inputs()
    np.testing.assert_array_equal(next_tensors["command_history"][:, 0], [0, 10, 10])
    np.testing.assert_array_equal(tensors["command_history"][:, 0], [0, 0, 10])
