"""Example of a custom experiment wrapped around an RLlib Algorithm."""
import os
import sys
import warnings

import pygame
# from ray.air.integrations.wandb import WandbLoggerCallback
# from artist.utils.custom_wandb import WandbLoggerCallback
from pygame.locals import *

from rl.utils import training_utils, config_utils


def fxn():
    warnings.warn("deprecated", DeprecationWarning)

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    fxn()

import ray.rllib.algorithms.ppo as ppo
from ray.tune.registry import register_env


def manual_eval(eval_algo, env_type):
    pygame.init()
    clock = pygame.time.Clock()
    env = env_type(config=None, render=True)

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


if __name__ == "__main__":
    runtime_env = {"PYTHONWARNINGS": "ignore"}
    # ray.init(accelerator_type="TITAN")
    register_env("shroom_collector_env", ShroomCollectorEnv)


    # config = ppo.PPOConfig().environment("shroom_collector_env")
    # config = config.to_dict()
    #
    # # config["train-iterations"] = train_iterations
    # config["num_cpus_per_worker"] = 1.0
    # config["num_workers"] = 0
    # config["num_gpus"] = 0
    # config["model"] = {
    #     "custom_model": None,
    #     "custom_model_config": {},
    #     "custom_preprocessor": None,
    #     "conv_filters": [
    #         [32, [Constants.COLLECTOR_VISION_BOX, Constants.COLLECTOR_VISION_BOX], 1],  # 32 filters, 3x3 kernel size, stride 1
    #     ]
    #     # 20
    #     # "conv_filters": [
    #     #     [16, [3, 3], 2],  # 16 filters, 3x3 kernel size, stride 2
    #     #     [32, [3, 3], 2],  # 32 filters, 3x3 kernel size, stride 2
    #     #     [64, [4, 4], 2],  # 64 filters, 4x4 kernel size, stride 2
    #     # ],
    # }
    config = config_utils.get_config()

    algo = ppo.PPO(config=config, env="shroom_collector_env")
    script_path = os.path.dirname(os.path.abspath(__file__))
    load_path = os.path.join(script_path, '../../ray_results', 'test')
    checkpoint_path = training_utils.find_latest_checkpoint(load_path)
    print(checkpoint_path)
    algo.load_checkpoint(os.path.join(load_path, checkpoint_path))
    # # print(algo)
    manual_eval(algo)
