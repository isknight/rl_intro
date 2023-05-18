from abc import ABC

import gym
from gym import spaces
from ray.rllib.env.env_context import EnvContext

from rl.games.shroom_collector_game import ShroomCollectorGame, Constants
from rl.utils import level_util


class GrassyIslandEnv(gym.Env, ABC):

    def __init__(self, config: EnvContext, render=False):
        # Action space: 0 = up, 1 = right, 2 = down, 3 = left
        self.action_space: spaces.Discrete = spaces.Discrete(4)

        # A way to get vision sizes around the shroom collector
        # vision_size = Constants.COLLECTOR_VISION_BOX

        # TODO question: What observational space schema do we need?
        self.observation_space = spaces.Dict(
            {
                "place_holder": spaces.Discrete(1),
            }
        )
        self.render: bool = render
        self.game: ShroomCollectorGame = level_util.get_grassy_island(ui_enabled=self.render)
        self.reset()

    def reset(self) -> None:
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

        # TODO - What observational space should we have?
        return {
                    "place_holder": 1
               }
