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

Shown below are 3 graphs which demonstrate the relative performance of 5 different bandit algorithms tested on 2000 different randomly generated bandit tasks (which ran in total in just under 5 minutes using a single CPU process), which respectively show:

- The rewards over time, averaged across all tasks
- The percentage of actions chosen that were optimal over time, averaged across the same tasks
- A bar chart of the total mean rewards, showing that the Bayesian sampling algorithms are best overall (with a value-based prior being marginally better than a broad prior)

![Mean rewards over time](https://github.com/jakelevi1996/reinforcement_learning/blob/main/scripts/Results/Protected/Bandit/2000_repeats_1000_steps/10-armed_bandit_mean_rewards__1000_steps,_2000_repeats_.png?raw=true "Mean rewards over time")

![Percentage optimal actions](https://github.com/jakelevi1996/reinforcement_learning/blob/main/scripts/Results/Protected/Bandit/2000_repeats_1000_steps/10_armed_bandit_percentage_of_optimal_actions__1000_steps,_2000_repeats_.png?raw=true "Percentage optimal actions")

![Bar chart of total mean rewards](https://github.com/jakelevi1996/reinforcement_learning/blob/main/scripts/Results/Protected/Bandit/2000_repeats_1000_steps/10_armed_bandit_total_mean_rewards__1000_steps,_2000_repeats_.png?raw=true "Bar chart of total mean rewards")

### Parameter sweeps

- Parameter sweeps for the bandit algorithms can be performed using the script `scripts/param_sweep_bandits.py`
- The parameter sweeps below for the algorithms epsilon-greedy, epsilon-greedy with constant step size, and the gradient bandit algorithm were performed with the command `python scripts/param_sweep_bandits.py --num_repeats 200 --num_values 20`, which ran in about 10 minutes using a single CPU process
- Parameter sweeps were not performed for either of the Bayesian sampling algorithms, because both of these algorithms are non-parametric
- The parameter sweep results show mean reward achieved after 1000 time steps on 200 different randomly generated 10-armed bandit tasks, for each of 20 different values of each parameter being considered
- We want to choose the optimal value for each parameter as one which *reliably* produces a high mean reward
  - Therefore, the optimal value for each parameter is considered to be not the value with the highest mean reward, but the value with the highest of a particular linear combination of mean and standard deviation of the mean reward over different randomly generated bandit tasks
  - This is to ensure that an optimal value is not chosen as one that has high mean reward if it also has excessively high variance, which would imply that it does not *reliably* produce a high reward
- For algorithms with multiple parameters, each parameter is set to its optimal value when it is not being swept

The epsilon-greedy algorithm only has one parameter, epsilon. Below are the parameter sweep results for the parameter epsilon:

![Varying parameter epsilon](https://github.com/jakelevi1996/reinforcement_learning/blob/main/scripts/Results/Protected/Param_sweeps/Bandit/200_repeats_1000_steps_20_values/Epsilon_greedy/Parameter_sweep_results_for__Epsilon_greedy_,_varying_parameter__epsilon_.png?raw=true "Varying parameter epsilon")

The epsilon-greedy algorithm with constant step size has two parameters, epsilon and step size. Below are the parameter sweep results for both of these parameters:

![Varying parameter epsilon](https://github.com/jakelevi1996/reinforcement_learning/blob/main/scripts/Results/Protected/Param_sweeps/Bandit/200_repeats_1000_steps_20_values/Epsilon_greedy_constant_step_size/Varying_epsilon.png?raw=true "Varying parameter epsilon")

![Varying parameter step size](https://github.com/jakelevi1996/reinforcement_learning/blob/main/scripts/Results/Protected/Param_sweeps/Bandit/200_repeats_1000_steps_20_values/Epsilon_greedy_constant_step_size/Varying_step_size.png?raw=true "Varying parameter step size")

The gradient bandit algorithm only has one parameter, step size. Below are the parameter sweep results for the parameter step size:

![Varying parameter step size](https://github.com/jakelevi1996/reinforcement_learning/blob/main/scripts/Results/Protected/Param_sweeps/Bandit/200_repeats_1000_steps_20_values/Gradient_bandit/Parameter_sweep_results_for__Gradient_bandit_,_varying_parameter__step_size_.png?raw=true "Varying parameter step size")

### Profiling

It is possible to profile the `compare_bandits.py` script and save the profiling information to a file with a unique timestamped filename with the following commands:

```
python -m cProfile -o .profile ./scripts/compare_bandits.py --no_plot --no_save
python -c "import pstats; p = pstats.Stats('.profile'); p.sort_stats('cumtime'); p.print_stats(50)" > ".profile $(date '+%Y-%m-%d %H-%M-%S').txt"
```
