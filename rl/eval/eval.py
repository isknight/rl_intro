"""Example of a custom experiment wrapped around an RLlib Algorithm."""
import os
import sys
from typing import Type
import gym
import pygame
# from ray.air.integrations.wandb import WandbLoggerCallback
# from artist.utils.custom_wandb import WandbLoggerCallback
from pygame.locals import *

from rl.utils import training_utils, config_utils

import ray.rllib.algorithms.ppo as ppo
from ray.tune.registry import register_env


def manual_eval(eval_algo: str, env_type: Type[gym.Env]) -> None:
    pygame.init()
    clock = pygame.time.Clock()
    env = env_type(render=True)

    obs = env.reset()
    env.game.render()
    pygame.display.update()
    is_done = False

    while True:
        while not is_done:
            for event in pygame.event.get():
                if event.type == QUIT:
                    pygame.quit()
                    sys.exit()
            action = eval_algo.compute_single_action(obs)
            obs, reward, is_done, _ = env.step(action)
            env.game.render()
            pygame.display.update()
            clock.tick(10)
        # env = ShroomCollectorEnv(config=None, render=True)
        obs = env.reset()
        is_done = False

