import os
import string
import sys
from typing import Type

import click
import gym
import pygame
import ray.rllib.algorithms.ppo as ppo
from colorama import Fore, Style
from pygame.locals import *
from ray import tune
from ray.tune.registry import register_env

from rl.envs.eruption_env import EruptionEnv
# from ray.air.integrations.wandb import WandbLoggerCallback
# from artist.utils.custom_wandb import WandbLoggerCallback
from rl.envs.grassy_island_env import GrassyIslandEnv
from rl.envs.maze_env import MazeEnv
from rl.eval.eval import manual_eval
from rl.games.shroom_collector_game import Constants
from rl.training.train import experiment
from rl.utils import config_utils
from rl.utils import training_utils


def get_env_by_name(name: str) -> Type[gym.Env]:
    if name == "grassy_island_env":
        return GrassyIslandEnv
    elif name == "maze_env":
        return MazeEnv
    elif name == "eruption_env":
        return EruptionEnv

    return None


# Dummy function that represents your actual RL training function
def train_agent(experiment_name, environment, iterations):
    print(
        f"{Fore.LIGHTBLUE_EX}Training {Fore.RED}'{experiment_name}' {Fore.LIGHTBLUE_EX}on {Fore.RED}'{environment}' {Fore.LIGHTBLUE_EX}for {Fore.RED}{iterations} {Fore.LIGHTBLUE_EX}iterations.{Style.RESET_ALL}")

    def get_trial_name(results_dir):
        return f"{results_dir}_results"

    # Get the env based on name mapping
    env = get_env_by_name(environment)
    print(f"Loading env={environment}")
    if env is None:
        print(f"Env name {env} does not map to an existing env.")
        sys.exit(0)

    # Registering env
    register_env("shroom_collector_env", env)

    # Get the configuration
    config = config_utils.get_config()

    # Launch a tuner manually.
    logdir = config_utils.get_path(f"ray_results/{experiment_name}")
    print(f"Logging to: {logdir}")
    tune.run(
        tune.with_parameters(
            tune.with_resources(
                experiment,
                ppo.PPO.default_resource_request(config)
            ),
            experiment_name=experiment_name,
            iterations=iterations
        ),
        config=config,
        stop={"training_iteration": 100},
        local_dir=logdir,
        trial_name_creator=get_trial_name,
        # resume="AUTO"
    )


# Dummy function that represents your actual RL evaluation function
def eval_agent(experiment_name, environment):
    print(f"Evaluating agent '{experiment_name}' on '{environment}'.")
    env = get_env_by_name(environment)

    register_env("shroom_collector_env", env)
    config = config_utils.get_config()

    algo = ppo.PPO(config=config, env="shroom_collector_env")

    load_path = config_utils.get_path(f"ray_results/{experiment_name}")

    checkpoint_path = training_utils.find_latest_checkpoint(load_path)
    print(checkpoint_path)
    algo.load_checkpoint(os.path.join(load_path, checkpoint_path))
    # # print(algo)
    manual_eval(algo, env)


def play_game(environment: string) -> None:
    env_type = get_env_by_name(environment)
    env = env_type(config=None, render=True)
    # game = level_util.get_grassy_island(ui_enabled=True)
    game = env.game
    clock = pygame.time.Clock()

    action = None
    while True:
        for event in pygame.event.get():
            if event.type == QUIT:
                pygame.quit()
                sys.exit()

            # Handle character movement
            if event.type == KEYDOWN:
                if event.key == K_w:
                    action = Constants.ACTION_UP
                elif event.key == K_s:
                    action = Constants.ACTION_DOWN
                elif event.key == K_a:
                    action = Constants.ACTION_LEFT
                elif event.key == K_d:
                    action = Constants.ACTION_RIGHT

        # Draw the map and character
        if not action in game.get_available_action():
            action = None

        game.step(action)
        action = None

        pygame.display.update()
        over, reason = game.check_game_over_state()
        clock.tick(10)

        if over:
            game.reset()

    return None


@click.group()
def rl():
    pass


@rl.command()
@click.option('-x', '--experiment_name', required=True, help='Name of the experiment.')
@click.option('-e', '--env', required=True, help='Environment of the experiment.')
@click.option('-i', '--iterations', default=10, help='Number of training iterations.')
def train(experiment_name, env, iterations):
    train_agent(experiment_name, env, iterations)


@rl.command()
@click.option('-x', '--experiment_name', required=True, help='Name of the experiment.')
@click.option('-e', '--env', required=True, help='Environment of the experiment.')
def eval(experiment_name, env):
    eval_agent(experiment_name, env)


@rl.command()
@click.option('-e', '--env', required=False, help='Name of the env.')
def play(env):
    play_game(env)


if __name__ == '__main__':
    rl()
