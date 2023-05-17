from abc import ABC

import gym
import numpy as np
from gym import spaces
from rl.games.shroom_collector_game import ShroomCollectorGame, Constants
from ray.rllib.env.env_context import EnvContext
from rl.utils import env_utils
from rl.utils import level_util


class MazeEnv(gym.Env, ABC):

    def __init__(self, config: EnvContext, render=False):
        # Action space: 0 = up, 1 = right, 2 = down, 3 = left
        self.action_space: spaces.Discrete = spaces.Discrete(4)

        vision_size = Constants.COLLECTOR_VISION_BOX

        # TODO question: What observational space do we need?
        self.observation_space = spaces.Dict(
            {
                "collector_vision": spaces.Box(low=-1, high=6, shape=(vision_size, vision_size), dtype=int),
            }
        )
        self.render: bool = render
        self.game: ShroomCollectorGame = level_util.get_maze(ui_enabled=self.render)

        self.reset()

    def reset(self):
        self.game.reset()
        return self._get_observation()

    def step(self, action):
        available_actions = self.game.get_available_action()
        before_shroom_count = self.game.shrooms_collected
        if action in available_actions:
            self.game.step(action)

        done, reason = self.game.check_game_over_state()

        # TODO question: What reward do we need?
        reward = 0

        if self.game.shrooms_collected > before_shroom_count:
            # We know we collected a shroom.
            reward = 1.0
        else:
            reward = -0.005


        return self._get_observation(), reward, done, {}






    def _get_observation(self):
        layers = np.zeros((20, 20, 2), dtype=int)
        layers[:, :, 0] = self.game.terrain_layer.copy()
        layers[:, :, 1] = self.game.mushroom_layer.copy()

        collector_vision = np.zeros((Constants.COLLECTOR_VISION_BOX, Constants.COLLECTOR_VISION_BOX, 2), dtype=int)
        x = self.game.character_position[0]
        y = self.game.character_position[1]

        collector_vision[:, :, 1] = env_utils.extract_collector_vision(self.game.mushroom_layer, x, y, Constants.COLLECTOR_VISION_BOX)
        vision = env_utils.extract_collector_vision(self.game.mushroom_layer, x, y, Constants.COLLECTOR_VISION_BOX)
        return {
                "collector_vision": vision,
               }
