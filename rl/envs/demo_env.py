import gym
from typing import Dict, Any
import numpy as np
from gym import spaces
from rl.games.shroom_collector_game import ShroomCollectorGame, Constants
from ray.rllib.env.env_context import EnvContext
from rl.utils import env_utils
from rl.utils import level_util


class DemoEnv(gym.Env):
    def __init__(self, render=False):

        # Action space: 0 = up, 1 = right, 2 = down, 3 = left
        self.action_space: spaces.Discrete = spaces.Discrete(4)

        vision_size = Constants.COLLECTOR_VISION_BOX
        self.observation_space = spaces.Dict(
            {
                "collector_vision": spaces.Box(low=-1, high=6, shape=(vision_size, vision_size), dtype=int),
                "energy": spaces.Box(low=0, high=2000, shape=(1,), dtype=int)
            }
        )
        self.render: bool = render
        self.game: ShroomCollectorGame = level_util.get_demo(self.render)
        self.reset()

    def reset(self) -> Dict[str, Any]:
        self.game.reset()
        return self._get_observation()

    def step(self, action) -> tuple[Dict[str, Any], float, bool, dict]:

        reward = 0
        available_actions = self.game.get_available_action()
        before_shroom_count = self.game.shrooms_collected
        if action in available_actions:
            self.game.step(action)
        else:
            reward = -0.005

        done, reason = self.game.check_game_over_state()

        if done:
            if reason == Constants.REASON_LAVA:
                reward = -1.0
                reward += self.game.shrooms_collected * 0.5
            elif reason == Constants.REASON_ENERGY:
                reward = -1.0
                reward += self.game.shrooms_collected * 1
            elif reason == Constants.REASON_SHROOMS_COLLECTED:
                reward = 1.0
                reward += (self.game.energy / 100)

        elif self.game.shrooms_collected > before_shroom_count:
            # We know we collected a shroom.
            reward = 1.0
        else:
            reward = -0.005

        return self._get_observation(), reward, done, {}

    def _get_observation(self) -> Dict[str, Any]:

        layers = np.zeros((20, 20, 2), dtype=int)
        layers[:, :, 0] = self.game.terrain_layer.copy()
        layers[:, :, 1] = self.game.mushroom_layer.copy()

        collector_vision = np.zeros((Constants.COLLECTOR_VISION_BOX, Constants.COLLECTOR_VISION_BOX, 2), dtype=int)
        x = self.game.character_position[0]
        y = self.game.character_position[1]
        collector_vision[:, :, 0] = env_utils.extract_collector_vision(self.game.terrain_layer, x, y, Constants.COLLECTOR_VISION_BOX)
        collector_vision[:, :, 1] = env_utils.extract_collector_vision(self.game.mushroom_layer, x, y, Constants.COLLECTOR_VISION_BOX)
        vision = env_utils.extract_collector_vision(self.game.mushroom_layer, x, y, Constants.COLLECTOR_VISION_BOX)
        return {
                "collector_vision": vision,
                "energy": np.array([self.game.energy])
               }
