import os
import ray.rllib.algorithms.ppo as ppo
from ray.rllib.models import ModelCatalog

from rl.games.shroom_collector_game import Constants
from ray.rllib.models.tf.fcnet import FullyConnectedNetwork


def get_path(local_path: str) -> str:
    script_path = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(script_path, '../../', local_path)


def get_config():
    config = ppo.PPOConfig().environment("shroom_collector_env")
    ModelCatalog.register_custom_model("my_fc_model", FullyConnectedNetwork)

    config = config.to_dict()

    config["num_cpus_per_worker"] = 1.0
    config["num_workers"] = 0
    config["num_gpus"] = 0
    config["model"] = {
        "custom_model": "my_fc_model",
        "custom_model_config": {},
        "custom_preprocessor": None,
    }
    config["model"]["post_fcnet_hiddens"] = [128, 128]
    config["model"]["post_fcnet_activation"] = "relu"

    return config
