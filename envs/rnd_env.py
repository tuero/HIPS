import gym
import numpy as np
from numpy.typing import NDArray

import torch

from gym.spaces.discrete import Discrete

from hptslite.env.rnd import RNDSimpleTreeState


def reconstruct_rnd_state(obs: NDArray[np.float32]) -> RNDSimpleTreeState:
    board_str = ""
    num_diamonds = 0
    vis_to_hidden = {
        4: 5,
        5: 7,
        6: 8,
        7: 9,
        10: 18,
        11: 19,
        16: 27,
        17: 28,
        18: 29,
        19: 30,
        20: 31,
        21: 32,
        22: 33,
        23: 34,
        24: 35,
        25: 36,
        26: 37,
        27: 38,
    }
    for r in range(14):
        for c in range(14):
            vis_id = np.flatnonzero(obs[:, r, c])[0]
            if vis_id == 4:
                num_diamonds += 1
            hid_id = vis_id if vis_id not in vis_to_hidden else vis_to_hidden[vis_id]
            board_str += "|{:02d}".format(hid_id)
    board_str = f"14|14|9999|{num_diamonds}" + board_str
    return RNDSimpleTreeState(board_str)


class RNDEnv(gym.Env):
    """
    The RND environment
    """
    def __init__(self, state):
        if isinstance(state, str):
            self.state = RNDSimpleTreeState(state)
        else:
            if isinstance(state, torch.Tensor):
                state = state.cpu().long().numpy()
                self.state = reconstruct_rnd_state(state)
            elif isinstance(state, np.ndarray):
                state = state.astype(int)
                self.state = reconstruct_rnd_state(state)
            else:
                raise Exception("Unknown input type")
        self.action_space = Discrete(len(ACTION_LOOKUP))

    def reset(self):
        return self.state.get_observation()

    def img(self):
        return self.state.to_image()

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