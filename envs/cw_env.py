import gym
import numpy as np
from numpy.typing import NDArray

import torch

from gym.spaces.discrete import Discrete

from hptslite.env.craftworld import CraftWorldTreeState


kNumRecipeTypes = 11
kNumEnvironment = 8
kNumPrimitive = 7
kNumGoals = kNumRecipeTypes
kNumInventory = kNumPrimitive + kNumRecipeTypes


def reconstruct_craftworld_state(obs: NDArray[np.float32]):
    board_str = ""
    # Background board
    for r in range(14):
        for c in range(14):
            el_id = np.flatnonzero(obs[0:15, r, c])
            el_id = 26 if len(el_id) == 0 else el_id[0]
            board_str += "|{:02d}".format(el_id)
    goal = 25
    for i in range(kNumGoals):
        channel = kNumEnvironment + kNumPrimitive + 2*kNumInventory + i
        if int(np.mean(obs[channel].flatten())) == 1:
            goal = channel - 2*kNumInventory
    board_str = f"14|14|{goal}" + board_str
    state = CraftWorldTreeState(board_str)

    # Inventory
    for i in range(kNumPrimitive + kNumRecipeTypes):
        el = kNumEnvironment + i
        idx = kNumEnvironment + kNumPrimitive + 2*i        
        if int(np.mean(obs[idx].flatten())) == 1:
            state.add_to_inventory(el, 1)
        if int(np.mean(obs[idx + 1].flatten())) == 1:
            state.add_to_inventory(el, 1)
    return state


class CraftWorldEnv(gym.Env):
    """
    The CraftWorld environment
    """
    def __init__(self, state):
        if isinstance(state, str):
            self.state = CraftWorldTreeState(state)
        else:
            if isinstance(state, torch.Tensor):
                state = state.cpu().long().numpy()
                self.state = reconstruct_craftworld_state(state)
            elif isinstance(state, np.ndarray):
                state = state.astype(int)
                self.state = reconstruct_craftworld_state(state)
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
    4: 'use',
}