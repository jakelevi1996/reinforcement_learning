# reinforcement_learning

This repository contains implementations of RL (reinforcement learning) algorithms, specifically from "Sutton, Richard S., and Andrew G. Barto. Reinforcement learning: An introduction. MIT press, 2018." Eventually I hope to also add implementations of more cutting edge RL algorithms directly from the literature.

## Contents

- [reinforcement_learning](#reinforcement_learning)
  - [Contents](#contents)
  - [K-armed bandits](#k-armed-bandits)
    - [Results](#results)
    - [Parameter sweeps](#parameter-sweeps)
    - [Profiling](#profiling)

## K-armed bandits

The K-armed bandit is arguably the simplest form of reinforcement learning problem, consisting of a single state, and a choice of multiple actions to take with unknown distributions over their rewards, the long-term goal being to maximise total cumulative reward. [More information about K-armed bandits is available on Wikipedia](https://en.wikipedia.org/wiki/Multi-armed_bandit).

### Results

*Coming soon*

### Parameter sweeps

Parameter sweeps for the bandit algorithms can be performed using the script `scripts/param_sweep_bandits.py`. The parameter sweeps below for the algorithms epsilon-greedy, epsilon-greedy with constant step size, and the gradient bandit algorithm were performed with the command `python scripts/param_sweep_bandits.py --num_repeats 200 --num_values 20`, which ran in about 10 minutes using a single CPU process. Parameter sweeps were not performed for either of the Bayesian sampling algorithms, because both of these algorithms are non-parametric.

The epsilon-greedy algorithm only has one parameter, epsilon. Below are the results for mean reward achieved after 1000 time steps on 200 different randomly generated 10-armed bandit task, for each of 20 different values of epsilon:

![Varying parameter epsilon](https://github.com/jakelevi1996/reinforcement_learning/blob/main/scripts/Results/Protected/Param_sweeps/Bandit/200_repeats_1000_steps_20_values/Epsilon_greedy/Parameter_sweep_results_for__Epsilon_greedy_,_varying_parameter__epsilon_.png?raw=true "Varying parameter epsilon")

### Profiling

It is possible to profile the `compare_bandits.py` script and save the profiling information to a file with a unique timestamped filename with the following commands:

```
python -m cProfile -o .profile ./scripts/compare_bandits.py --no_plot --no_save
python -c "import pstats; p = pstats.Stats('.profile'); p.sort_stats('cumtime'); p.print_stats(50)" > ".profile $(date '+%Y-%m-%d %H-%M-%S').txt"
```
