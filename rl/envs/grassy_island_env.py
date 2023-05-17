from abc import ABC

import gym
import numpy as np
from gym import spaces
from ray.rllib.env.env_context import EnvContext

from rl.games.shroom_collector_game import ShroomCollectorGame, Constants
from rl.utils import env_utils
from rl.utils import level_util


class GrassyIslandEnv(gym.Env, ABC):

    def __init__(self, config: EnvContext, render=False):
        # Action space: 0 = up, 1 = right, 2 = down, 3 = left
        self.action_space: spaces.Discrete = spaces.Discrete(4)

        # Observation space: NxN one-hot encoded grid for both layers:
        # terrain layer
        # mushroom layer
        # Auto generated model limits to one array input
        vision_size = Constants.COLLECTOR_VISION_BOX
        # vision_size = 5

        # TODO question: What observational space do we need?
        self.observation_space = spaces.Dict(
            {
                "collector_vision": spaces.Box(low=-1, high=6, shape=(vision_size, vision_size), dtype=int),
            }
        )
        self.render: bool = render
        self.game: ShroomCollectorGame = level_util.get_grassy_island(ui_enabled=self.render)

        self.reset()

    def reset(self):
        self.game.reset()
        return self._get_observation()

    def step(self, action):
        available_actions = self.game.get_available_action()
        if action in available_actions:
            self.game.step(action)

        done, reason = self.game.check_game_over_state()

        # TODO question: What reward do we need?
        reward = 0

        return self._get_observation(), reward, done, {}


    def _get_observation(self):

        # Example getting layers directly and packaging them together
        layers = np.zeros((20, 20, 2), dtype=int)
        layers[:, :, 0] = self.game.terrain_layer.copy()
        layers[:, :, 1] = self.game.mushroom_layer.copy()


        # Example getting vision around the shroom collector
        collector_vision = np.zeros((Constants.COLLECTOR_VISION_BOX, Constants.COLLECTOR_VISION_BOX, 2), dtype=int)
        x = self.game.character_position[0]
        y = self.game.character_position[1]

        collector_vision[:, :, 1] = env_utils.extract_collector_vision(self.game.mushroom_layer, x, y, Constants.COLLECTOR_VISION_BOX)
        vision = env_utils.extract_collector_vision(self.game.mushroom_layer, x, y, Constants.COLLECTOR_VISION_BOX)
        return {
            "collector_vision": vision
               }
