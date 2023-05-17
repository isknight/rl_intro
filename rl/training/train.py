"""Example of a custom experiment wrapped around an RLlib Algorithm."""
import warnings
from ray.tune.logger import DEFAULT_LOGGERS
# from ray.air.integrations.wandb import WandbLoggerCallback
# from artist.utils.custom_wandb import WandbLoggerCallback
from rl.envs.shroom_collector_env import ShroomCollectorEnv
from rl.games.shroom_collector_game import Constants
from rl.utils import config_utils

import os
def fxn():
    warnings.warn("deprecated", DeprecationWarning)

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    fxn()
from ray.tune import Callback


from ray import tune
from ray.rllib.models.catalog import ModelCatalog, MODEL_DEFAULTS

import ray.rllib.algorithms.ppo as ppo
import csv
from ray.tune.registry import register_env
from ray.rllib.utils.pre_checks.env import check_gym_environments
from ray.rllib.agents.ppo import PPOTrainer
from rl.utils import training_utils

def experiment(config, experiment_name, iterations: int = 50):
    print(f"Running Experiment: {experiment_name} for {iterations} iterations")
    script_path = os.path.dirname(os.path.abspath(__file__))

    # checkpoints_path = os.path.join(script_path, '../../ray_results', experiment_name)
    checkpoints_path = config_utils.get_path(f"ray_results/{experiment_name}")
    # delete the checkpoints/experiments
    training_utils.delete_checkpoint_folders(checkpoints_path)
    #TODO resume the checkpoints/experiments
    print(f"checkpoints_path={checkpoints_path}")

    algo = ppo.PPO(config=config, env="shroom_collector_env")

    # Very manual train loop
    for i in range(iterations):
        print(f"experiment_name={experiment_name}")
        print(f"iteration: {i}")
        train_results = algo.train()

        tune.report(**train_results)

        # save checkpoint every 5 iterations
        if i % 1 == 0:
            checkpoint = algo.save(checkpoints_path)
    algo.stop()


