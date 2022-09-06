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

*Coming soon*

### Profiling

It is possible to profile the `bandit.py` script and save the profiling information to a file with a unique timestamped filename with the following commands:

```
python -m cProfile -o .profile ./scripts/bandit.py --no_plot --no_save
python -c "import pstats; p = pstats.Stats('.profile'); p.sort_stats('cumtime'); p.print_stats(50)" > ".profile $(date '+%Y-%m-%d %H-%M-%S').txt"
```
