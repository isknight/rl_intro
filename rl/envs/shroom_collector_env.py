import gym
import numpy as np
from gym import spaces
from rl.games.shroom_collector_game import ShroomCollectorGame, Constants
from ray.rllib.env.env_context import EnvContext
from rl.utils import env_utils


class ShroomCollectorEnv(gym.Env):
    def __init__(self, config: EnvContext, render=False):

        # Action space: 0 = up, 1 = right, 2 = down, 3 = left
        self.action_space: spaces.Discrete = spaces.Discrete(4)

        # Observation space: NxN one-hot encoded grid for both layers:
        # terrain layer
        # mushroom layer
        # Auto generated model limits to one array input

        vision_size = Constants.COLLECTOR_VISION_BOX
        # vision_size = 5
        self.observation_space = spaces.Dict(
            {
                "collector_vision": spaces.Box(low=-1, high=6, shape=(vision_size, vision_size), dtype=int),
            }
        )
        self.render: bool = render
        self.game: ShroomCollectorGame = ShroomCollectorGame(ui_enabled=self.render)
        self.reset()

    def reset(self):
        # self.game = ShroomCollectorGame(ui_enabled=self.render)
        self.game.reset()

        return self._get_observation()

    def step(self, action):

        reward = 0
        available_actions = self.game.get_available_action()
        before_shroom_count = self.game.shrooms_collected
        if action in available_actions:
            self.game.step(action)
        else:
            # self.game.energy -= 1
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
        # elif self.game.character_position == self.game.prev_character_position:
        #     reward = -0.05
        elif self.game.shrooms_collected > before_shroom_count:
            # We know we collected a shroom.
            reward = 1.0
        else:
            reward = -0.005

        # elif self.game.character_position != self.game.prev_character_position and self.game.get_surface_under_character() == Constants.TILE_ONE_HOT[Constants.SAND_COLOR]:
        #     print("SAND!!!!")
        #     reward = 1



        # surface = self.game.get_surface_under_character()
        # if surface == -1:
        #     reward = -0.005
        # elif surface == Constants.TILE_ONE_HOT[Constants.LAVA_COLOR]:
        #     print("LAVA!!!")
        #     reward = -10.0
        #     done = True
        # elif surface == Constants.TILE_ONE_HOT[Constants.WATER_COLOR]:
        #     reward = 0.025
        # elif surface == Constants.TILE_ONE_HOT[Constants.GRASS_COLOR]:
        #     reward = 0.025
        # if self.game.energy <= 0:
        #     done = True
        #     reward = 0

        # if self.game.character_position[0] == 19 and self.game.character_position[1] == 0:
        #     done = True
        #     reward = 10.0
        #     reward += (self.game.energy / 100)
        #
        return self._get_observation(), reward, done, {}

    def _get_observation(self):
        layers = np.zeros((20, 20, 2), dtype=int)
        layers[:, :, 0] = self.game.terrain_layer.copy()
        layers[:, :, 1] = self.game.mushroom_layer.copy()

        collector_vision = np.zeros((Constants.COLLECTOR_VISION_BOX, Constants.COLLECTOR_VISION_BOX, 2), dtype=int)
        x = self.game.character_position[0]
        y = self.game.character_position[1]
        collector_vision[:, :, 0] = env_utils.extract_collector_vision(self.game.terrain_layer, x, y, Constants.COLLECTOR_VISION_BOX)
        collector_vision[:, :, 1] = env_utils.extract_collector_vision(self.game.mushroom_layer, x, y, Constants.COLLECTOR_VISION_BOX)
        vision = env_utils.extract_collector_vision(self.game.mushroom_layer, x, y, Constants.COLLECTOR_VISION_BOX)
        # print(f"shape of vision={vision.shape}")
        return {
                "collector_vision": vision,
               }

    def _calculate_reward(self, old_position, new_position):
        # Define your reward calculation logic here
        reward = 0
        return reward
