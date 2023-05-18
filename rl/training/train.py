"""Example of a custom experiment wrapped around an RLlib Algorithm."""
import warnings

# from ray.air.integrations.wandb import WandbLoggerCallback
# from artist.utils.custom_wandb import WandbLoggerCallback
from rl.utils import config_utils


def fxn():
    warnings.warn("deprecated", DeprecationWarning)

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    fxn()

from ray import tune
import ray.rllib.algorithms.ppo as ppo


def experiment(config, experiment_name, iterations: int = 50) -> None:
    print(f"Running Experiment: {experiment_name} for {iterations} iterations")
    checkpoints_path = config_utils.get_path(f"ray_results/{experiment_name}")

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


