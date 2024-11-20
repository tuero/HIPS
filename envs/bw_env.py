import gym
import numpy as np
from numpy.typing import NDArray

import torch

from gym.spaces.discrete import Discrete

from hptslite.env.boxworld import BoxWorldTreeState


def reconstruct_bw_state(obs: NDArray[np.float32]) -> BoxWorldTreeState:
    board_str = ""
    # Board
    for r in range(20):
        for c in range(20):
            colour = np.flatnonzero(obs[0:14, r, c])
            board_str += "|{:02d}".format(14 if len(colour) == 0 else colour[0])
    board_str = f"20|20" + board_str
    state = BoxWorldTreeState(board_str)
    # Key
    k = np.flatnonzero(np.mean(obs[14:].reshape(-1, 20*20), -1))
    if len(k) > 0:
        state.set_key(k[0])
    return state


class BoxWorldEnv(gym.Env):
    """
    The BoxWorld environment
    """
    def __init__(self, state):
        if isinstance(state, str):
            self.state = BoxWorldTreeState(state)
        else:
            if isinstance(state, torch.Tensor):
                state = state.cpu().long().numpy()
                self.state = reconstruct_bw_state(state)
            elif isinstance(state, np.ndarray):
                state = state.astype(int)
                self.state = reconstruct_bw_state(state)
            else:
                raise Exception("Unknown input type")
        self.action_space = Discrete(len(ACTION_LOOKUP))

    def reset(self):
        return self.state.get_observation()

    def step(self, action: int):
        if isinstance(action, np.ndarray):
            action = action[0]
        action = int(action)
        assert action in ACTION_LOOKUP
        self.state.apply_action(action)
        reward = 1 if self.state.is_solution() else 0
        obs = self.state.get_observation()
        term = self.state.is_solution()
        return obs, reward, term, {}

    def img(self):
        return self.state.to_image()

    def set_maxsteps(self, num_steps):
        self.max_steps = num_steps

    def get_action_lookup(self):
        return ACTION_LOOKUP

    def get_action_meanings(self):
        return ACTION_LOOKUP
    

ACTION_LOOKUP = {
    0: 'move up',
    1: 'move right',
    2: 'move down',
    3: 'move left',
}