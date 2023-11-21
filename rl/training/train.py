import os
from rl.utils import config_utils
from typing import Any, Sequence
from numbers import Number
from ray import tune
import wandb
import numpy as np
import ray.rllib.algorithms.ppo as ppo
from wandb.sdk.data_types.base_types.wb_value import WBValue
from wandb.util import json_dumps_safer

if 'WANDB_API_KEY' in os.environ:
    wandb.init(
        project="rl_intro"
    )

def experiment(config, experiment_name, iterations: int = 50) -> None:
    print(f"Running Experiment: {experiment_name} for {iterations} iterations")
    checkpoints_path = config_utils.get_path(f"ray_results/{experiment_name}")

    algo = ppo.PPO(config=config, env="shroom_collector_env")

    # Very manual train loop
    for i in range(iterations):
        print(f"experiment_name={experiment_name}")
        print(f"iteration: {i}")
        train_results = algo.train()
        # filtered_result = {k: v for k, v in train_results.items() if is_serializable(v)}
        if 'WANDB_API_KEY' in os.environ:
            proces_wandb_dict(train_results)

        tune.report(**train_results)

        # save checkpoint every N iterations... I guess 1 for now!
        if i % 1 == 0:
            checkpoint = algo.save(checkpoints_path)
    algo.stop()

def proces_wandb_dict(train_results):

    clean_result = {}
    process_dict(train_results, ["episode"], clean_result)
    process_dict(train_results["info"]["learner"]["default_policy"]["learner_stats"], ["total_loss"], clean_result)
    wandb.log(clean_result)

def process_dict(dictionary, filters, clean_result):
    result = _clean_log(dictionary)
    for key in result:
        if any(k in key for k in filters):
            v = result[key]
            if not type(v) == type(True):
                clean_result[key] = v


def _is_allowed_type(obj):
    """Return True if type is allowed for logging to wandb"""
    if isinstance(obj, np.ndarray) and obj.size == 1:
        return isinstance(obj.item(), Number)
    if isinstance(obj, Sequence) and len(obj) > 0:
        return isinstance(obj[0], WBValue)
    return isinstance(obj, (Number, WBValue))


def _clean_log(obj: Any):
    # Fixes https://github.com/ray-project/ray/issues/10631
    if isinstance(obj, dict):
        return {k: _clean_log(v) for k, v in obj.items()}
    elif isinstance(obj, (list, set)):
        return [_clean_log(v) for v in obj]
    elif isinstance(obj, tuple):
        return tuple(_clean_log(v) for v in obj)
    elif _is_allowed_type(obj):
        return obj

    # Else

    try:
        # This is what wandb uses internally. If we cannot dump
        # an object using this method, wandb will raise an exception.
        json_dumps_safer(obj)

        # This is probably unnecessary, but left here to be extra sure.
        pickle.dumps(obj)

        return obj
    except Exception:
        # give up, similar to _SafeFallBackEncoder
        fallback = str(obj)

        # Try to convert to int
        try:
            fallback = int(fallback)
            return fallback
        except ValueError:
            pass

        # Try to convert to float
        try:
            fallback = float(fallback)
            return fallback
        except ValueError:
            pass

        # Else, return string
        return fallback