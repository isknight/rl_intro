# So you want to make AI bots? A gentle intro into reinforcement learning.

Prerequisites:

```text
Python 3.9.13
```

If you are using pyenv (a great way to manage python versions on your machine)

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


## What is reinforcement learning?

### TODO :) 

## The Problem

This project contains three "unsolved" environments built around variations of a simple game "Shroom Collector".

For the purpose of this intro lab, solve the following by training a policy 


### grassy_island_env

Our intrepid shroom collector needs to gather mushrooms for battle!

Every move made costs, an energy, each mushroom gathered gains 50 energy-- the objective is to gather 5 mushrooms.

What **observational_space** & **rewards** can help train our AI to consistently win this game?

Once you have an observational space and reward you're happy with you can train with:
```bash
python rl_intro.py train --experiment_name=grassy_island --env=grassy_island_env --iterations=20
```
Note, experiment_name can be anything. It's just the folder your policy checkpoints / logs /metrics are stored.
Watch the reward metrics. Are the rewards trending up?

Watch the episode_len_mean. Is it trending down?

If you want to see your agent play:

```bash 
rl_intro.py eval --experiment_name=grassy_island --env=grassy_island_env
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