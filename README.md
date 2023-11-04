# So you want to make AI bots? A gentle intro into reinforcement learning.

## Prerequisites:

```text
Python 3.9.13
```

If you are using [pyenv](https://github.com/pyenv/pyenv) (a great way to manage python versions on your machine)

```bash
pyenv install 3.9.13
pyenv global 3.9.13
```

then pip install `requirements.txt` into a virtual environment using your preferred tool of venv/poetry/pipenv etc.

To test if everything is running, from the top level folder run:

```bash
python rl_cli.py play --env grassy_island_env
```

This should pop up a human playable version of the first level we'll solve. Move the shroom collector with WSAD.



### M1 Macs

Note: If you have an M1 Mac, you will need to use an alternate approach


## What is reinforcement learning?

### TODO :) 

## The Problem

This project contains three "unsolved" environments built around variations of a simple game "Shroom Collector".

For this lab we'll ignore tuning the policy / model and focus on what we can do adjusting the observational_space and rewards. 


### The labs

In each of the following labs, you will edit the *_env.py file as needed. At the very least you will need to define an observational space and reward signal.

For example, in the grassy_island_env.py, this means

1. defining the shape of your observation_space here. Read more about gym [spaces](https://www.gymlibrary.dev/api/spaces/) here.
2. Populating said observation_space with game state data which will be your policy/model's input and what the agent can "see".
3. Determining a reward based upon different conditions.

For each of these you can look at [demo_env.py](rl/envs/demo_env.py) for starting examples.

### grassy_island_env

Our intrepid shroom collector needs to gather mushrooms for battle!

Every move made costs energy, each mushroom gathered gains 50 energy-- the objective is to gather 5 mushrooms for battle.

What **observational_space** & **rewards** can help train our AI to consistently win this game?

Once you have an observational space and reward you're happy with you can train with:

```bash
python rl_cli.py train --experiment_name=grassy_island --env=grassy_island_env --iterations=20
```

Note, experiment_name can be anything. It's just the folder your policy checkpoints / logs /metrics are stored.

Watch the reward metrics. Are the rewards trending up?

Watch the episode_len_mean. Is it trending down?

If you have a good observation_space defined, and a good strategy with rewards you will see reward averages go up and episode_len go down.

If you want to see your agent play:

```bash 
rl_cli.py eval --experiment_name=grassy_island --env=grassy_island_env
```
Note, experiment_name can be anything. It's just the folder your policy checkpoints / logs /metrics are stored.

### eruption_env

Oh no! the island has erupted spewing lava! If our brave shroom collector steps in the lava, he will die!

On top of that he needs to collect 10 shrooms now since he needs to save up until the lava subsides.

What observational_space and rewards can help train our AI to consistently win this game?

Once you have an observational space and reward you're happy with you can train with:
```bash
python rl_intro.py train --experiment_name=eruption --env=eruption_env --iterations=20
```
Note, experiment_name can be anything. It's just the folder your policy checkpoints / logs /metrics are stored.

Watch the reward metrics. Are the rewards trending up?

Watch the episode_len_mean. Is it trending down?

If you want to see your agent play:

```bash 
rl_intro.py eval --experiment_name=eruption --env=eruption_env
```

### maze_env.

This one is tricky. Same as above, just use `maze_env`. 