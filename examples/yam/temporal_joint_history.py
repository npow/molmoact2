"""Align each measured observation with the joint command preceding it."""
from collections import deque

import numpy as np


class TemporalJointHistory:
    def __init__(self, length):
        self.observations = deque(maxlen=length)
        self.previous_command = None

    def observe(self, cameras, joints):
        if self.previous_command is None:
            self.commanded(joints)
        item = ({name: frame.copy() for name, frame in cameras.items()},
                joints.copy(), self.previous_command)
        repeats = self.observations.maxlen if not self.observations else 1
        self.observations.extend([item] * repeats)

    def commanded(self, joints):
        self.previous_command = np.asarray(joints, dtype=np.float32).copy()

    def inputs(self):
        frames, states, commands = zip(*self.observations)
        cameras = {name: np.stack([row[name] for row in frames]) for name in frames[-1]}
        return cameras, {
            "state_history": np.asarray(states, dtype=np.float32),
            "command_history": np.asarray(commands, dtype=np.float32),
        }
