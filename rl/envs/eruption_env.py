from abc import ABC
from typing import Dict, Any
import gym
import numpy as np
from gym import spaces
from gym.core import ObsType
from ray.rllib.env.env_context import EnvContext

from rl.games.shroom_collector_game import ShroomCollectorGame, Constants
from rl.utils import env_utils
from rl.utils import level_util


class EruptionEnv(gym.Env, ABC):

    def __init__(self, render=False):
        # Action space: 0 = up, 1 = right, 2 = down, 3 = left
        self.action_space: spaces.Discrete = spaces.Discrete(4)

        vision_size = Constants.COLLECTOR_VISION_BOX
        # vision_size = 5

        # TODO question: What observational space do we need?
        self.observation_space = spaces.Dict(
            {   # TODO question: what observation space makes sense?
                # "collector_vision": spaces.Box(low=-1, high=6, shape=(vision_size, vision_size), dtype=int),
            }
        )
        self.render: bool = render
        self.game: ShroomCollectorGame = level_util.get_eruption(ui_enabled=self.render)

        self.reset()

    def reset(self) -> Dict[str, Any]:
        self.game.reset()
        return self._get_observation()

    def step(self, action) -> tuple[Dict[str, Any], float, bool, dict]:
        available_actions = self.game.get_available_action()
        before_shroom_count = self.game.shrooms_collected
        if action in available_actions:
            self.game.step(action)

        done, reason = self.game.check_game_over_state()

        # TODO question: What reward do we need?
        reward = 0

        return self._get_observation(), reward, done, {}

    def _get_observation(self) -> Dict[str, Any]:
        return {
                    "place_holder": 1
               }

