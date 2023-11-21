import gym
from typing import Dict, Any
import numpy as np
from gym import spaces
from rl.games.shroom_collector_game import ShroomCollectorGame, Constants
from ray.rllib.env.env_context import EnvContext
from rl.utils import env_utils
from rl.utils import level_util

VISION_DISTANCE = 5
# Used for calculating the area around the collector to grab
VISION_BOX = VISION_DISTANCE * 2 + 1


class DemoEnv(gym.Env):
    def __init__(self, render=False):

        # Action space: 0 = up, 1 = right, 2 = down, 3 = left
        self.action_space: spaces.Discrete = spaces.Discrete(4)

        self.observation_space = spaces.Dict(
            {
                # Multiple Layer Example
                # "collector_vision": spaces.Box(low=-1, high=6, shape=(VISION_BOX, VISION_BOX, 2), dtype=int),
                "collector_vision": spaces.Box(low=-1, high=6, shape=(VISION_BOX, VISION_BOX), dtype=int),
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
            # punitive if trying to do an action it can't do - e.g. walk into water
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

        # Minor hit for every move
        reward -= 0.0005

        return self._get_observation(), reward, done, {}

    def _get_observation(self) -> Dict[str, Any]:

        # Get character position
        x = self.game.character_position[0]
        y = self.game.character_position[1]

        # Just grab a single vision box out of the mushroom layer
        vision = env_utils.extract_collector_vision(self.game.mushroom_layer, x, y, VISION_BOX)

        # EXAMPLE of how to use multiple layers
        # Lets get vision for both the terrain_layer and mushroom_layer
        # Setup a 3d array of 2 vision layers deep
        # collector_vision = np.zeros((VISION_BOX, VISION_BOX, 2), dtype=int)
        # collector_vision[:, :, 0] = env_utils.extract_collector_vision(self.game.terrain_layer, x, y, VISION_BOX)
        # collector_vision[:, :, 1] = env_utils.extract_collector_vision(self.game.mushroom_layer, x, y, VISION_BOX)

        # max energy = 100 base + 5 * 50 for each shroom. Unattainable but technically the max.
        normalized_energy = env_utils.normalize_value(self.game.energy, 0, 350)
        return {
                "collector_vision": vision,
                "energy": np.array([normalized_energy])
               }
