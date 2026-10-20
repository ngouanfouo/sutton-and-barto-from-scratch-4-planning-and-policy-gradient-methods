# Sutton and Barto from Scratch 4: Planning and Policy-Gradient Methods

A hands-on, from-scratch implementation of planning and policy-gradient reinforcement-learning methods from *Reinforcement Learning: An Introduction* by Richard S. Sutton and Andrew G. Barto.

This project explores how an agent can learn not only from direct experience, but also from a learned model of its environment. It then moves from value-based planning methods to policy-based methods, including REINFORCE, REINFORCE with a learned baseline, and one-step actor-critic methods. Finally, it demonstrates differential semi-gradient Sarsa on a continuing access-control queueing problem.

## Overview

Reinforcement learning agents improve by interacting with an environment, observing rewards, and adjusting their behavior to maximize long-term return. This repository implements several important families of algorithms without relying on high-level reinforcement-learning libraries.

The project is divided into three main areas:

- **Model-based planning:** Learn a tabular environment model from experience and use it to perform additional simulated updates, accelerating learning in a maze environment.
- **Improved planning:** Handle changing environments and focus computational effort on the most useful updates using Dyna-Q+ and prioritized sweeping.
- **Policy-gradient methods:** Directly learn parameterized policies using REINFORCE, variance reduction with a learned baseline, and actor-critic learning.
- **Continuing tasks:** Apply differential semi-gradient Sarsa to an access-control queue, where the objective is average reward rather than discounted episodic return.

## What You Will See

- A tabular Dyna-Q agent learns a maze model and uses planning updates to discover the optimal 14-step path more efficiently than direct experience alone.
- Dyna-Q+ encourages exploration of actions whose environment model may have become outdated after a maze changes.
- Prioritized sweeping performs backups in order of urgency, concentrating planning on state-action pairs likely to produce the largest value updates.
- REINFORCE learns a stochastic policy for the short-corridor environment from complete episode returns.
- REINFORCE with baseline reduces gradient variance by learning a state-value function alongside the policy.
- One-step actor-critic combines a policy network or table with a value-function critic for more sample-efficient learning.
- Differential Sarsa solves the access-control problem by optimizing average reward in a continuing task.

## Features

- Pure Python implementations of core Sutton and Barto algorithms.
- Tabular Dyna-Q, Dyna-Q+, and prioritized sweeping.
- REINFORCE, REINFORCE with baseline, and one-step actor-critic.
- Differential semi-gradient Sarsa for continuing tasks.
- Clear, incremental implementation steps.
- Educational focus: each function maps closely to a concept from the book.

## Requirements

- Python 3.8 or later
- No external reinforcement-learning framework is required

Install dependencies if a `requirements.txt` file is provided:

```bash
pip install -r requirements.txt
```

## How to Run

Run the complete scaffold and experiments with:

```bash
python scaffold.py
```

This executes the implemented functions, runs the learning experiments, and prints progress or results for the maze, corridor, and access-control environments.

## Project Structure

```text
.
├── scaffold.py          # Main implementation and experiment runner
├── README.md            # Project documentation
└── requirements.txt     # Optional Python dependencies
```

## Implemented Steps

- [x] **1.** `build_maze`
- [x] **2.** `maze_step`
- [x] **3.** `epsilon_greedy_tabular`
- [x] **4.** `q_learning_update`
- [x] **5.** `planning_updates`
- [x] **6.** `dyna_q_episode`
- [x] **7.** `greedy_path_length`
- [x] **8.** `update_model_plus`
- [x] **9.** `dyna_q_plus_planning`
- [x] **10.** `run_blocking_maze`
- [x] **11.** `push_priority`
- [x] **12.** `prioritized_sweeping_episode`
- [x] **13.** `corridor_step`
- [x] **14.** `corridor_policy`
- [x] **15.** `corridor_true_value`
- [x] **16.** `reinforce_episode`
- [x] **17.** `reinforce_baseline_episode`
- [x] **18.** `actor_critic_corridor_episode`
- [x] **19.** `access_control_step`
- [x] **20.** `differential_sarsa_update`
- [x] **21.** `run_access_control`
- [x] **22.** `access_control_policy_table`

## Algorithms Covered

| Method | Type | Main Idea |
|---|---|---|
| Dyna-Q | Model-based planning | Learn a model from experience and use it for extra simulated Q-learning updates |
| Dyna-Q+ | Model-based planning | Encourage exploration of state-action pairs that have not been tried recently |
| Prioritized Sweeping | Model-based planning | Prioritize backups with the largest expected value change |
| REINFORCE | Monte Carlo policy gradient | Update policy parameters using complete-episode returns |
| REINFORCE with baseline | Policy gradient | Add a learned value baseline to reduce gradient variance |
| Actor-Critic | Policy gradient with bootstrapping | Use a critic to evaluate actions and guide policy updates |
| Differential Sarsa | Continuing-task control | Optimize average reward using differential temporal-difference error |

## Learning Outcomes

By working through this project, you will understand how to:

- Implement tabular Q-learning and epsilon-greedy action selection.
- Build and use a simple environment model for planning.
- Apply Dyna-style planning to accelerate reinforcement learning.
- Adapt planning methods to nonstationary environments.
- Use prioritized sweeping to make planning more efficient.
- Derive intuition for policy-gradient methods.
- Implement REINFORCE and reduce its variance with a baseline.
- Combine an actor and critic for one-step policy learning.
- Formulate and solve a continuing reinforcement-learning task using average reward.

## References

- Sutton, R. S., & Barto, A. G. *Reinforcement Learning: An Introduction*, 2nd Edition.
- Deep-ML, *Sutton and Barto from Scratch 4: Planning and Policy-Gradient Methods*.

## Author

**Tiayo Durel**

---

Built on Deep-ML.
