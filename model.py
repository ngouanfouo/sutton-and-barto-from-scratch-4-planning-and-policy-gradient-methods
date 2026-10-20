"""
Sutton and Barto from Scratch 4: Planning and Policy-Gradient Methods

Assembled from your step-by-step solutions.
"""

import numpy as np

# Step 1 - build_maze
def build_maze(walls=None, start=(2, 0), goal=(0, 8), rows=6, cols=9):
    if walls is None:
        walls = [(1, 2), (2, 2), (3, 2), (4, 5), (0, 7), (1, 7), (2, 7)]
    return {
        'rows': rows,
        'cols': cols,
        'start': tuple(start),
        'goal': tuple(goal),
        'walls': set(tuple(w) for w in walls),
        'n_actions': 4,
    }

# Step 2 - maze_step
def maze_step(maze, state, action):
    deltas = {0: (-1, 0), 1: (0, 1), 2: (1, 0), 3: (0, -1)}
    dr, dc = deltas[action]
    r, c = state
    nr, nc = r + dr, c + dc

    # Bump into a wall or the grid boundary: stay put.
    if not (0 <= nr < maze['rows'] and 0 <= nc < maze['cols']) or (nr, nc) in maze['walls']:
        nr, nc = r, c

    next_state = (nr, nc)
    done = next_state == maze['goal']
    reward = 1.0 if done else 0.0
    return next_state, reward, done

# Step 3 - epsilon_greedy_tabular
import numpy as np


def epsilon_greedy_tabular(q_table, state, epsilon, rng):
    values = q_table[state]
    n_actions = len(values)

    if rng.random() < epsilon:
        return int(rng.integers(0, n_actions))

    candidates = np.flatnonzero(values == values.max())
    return int(rng.choice(candidates))

# Step 4 - q_learning_update
import numpy as np


def q_learning_update(q_table, state, action, reward, next_state, alpha, gamma):
    new_table = q_table.copy()
    target = reward + gamma * np.max(q_table[next_state])
    new_table[state][action] += alpha * (target - q_table[state][action])
    return new_table

# Step 5 - planning_updates
import numpy as np


def planning_updates(q_table, model, n_planning, alpha, gamma, rng):
    keys = list(model.keys())
    if not keys or n_planning <= 0:
        return q_table.copy()

    indices = rng.integers(0, len(keys), size=n_planning)
    for i in indices:
        state, action = keys[i]
        next_state, reward = model[(state, action)]
        q_table = q_learning_update(q_table, state, action, reward, next_state, alpha, gamma)
    return q_table

# Step 6 - dyna_q_episode
import numpy as np


def dyna_q_episode(maze, q_table, model, n_planning, alpha, gamma, epsilon, rng):
    model = dict(model)
    state = maze['start']
    n_steps = 0
    done = False

    while not done:
        action = epsilon_greedy_tabular(q_table, state, epsilon, rng)
        next_state, reward, done = maze_step(maze, state, action)
        q_table = q_learning_update(q_table, state, action, reward, next_state, alpha, gamma)
        model[(state, action)] = (next_state, reward)
        q_table = planning_updates(q_table, model, n_planning, alpha, gamma, rng)
        state = next_state
        n_steps += 1

    return q_table, model, n_steps

# Step 7 - greedy_path_length (not yet solved)
# TODO: implement

# Step 8 - update_model_plus (not yet solved)
# TODO: implement

# Step 9 - dyna_q_plus_planning (not yet solved)
# TODO: implement

# Step 10 - run_blocking_maze (not yet solved)
# TODO: implement

# Step 11 - push_priority (not yet solved)
# TODO: implement

# Step 12 - prioritized_sweeping_episode (not yet solved)
# TODO: implement

# Step 13 - corridor_step (not yet solved)
# TODO: implement

# Step 14 - corridor_policy (not yet solved)
# TODO: implement

# Step 15 - corridor_true_value (not yet solved)
# TODO: implement

# Step 16 - reinforce_episode (not yet solved)
# TODO: implement

# Step 17 - reinforce_baseline_episode (not yet solved)
# TODO: implement

# Step 18 - actor_critic_corridor_episode (not yet solved)
# TODO: implement

# Step 19 - access_control_step (not yet solved)
# TODO: implement

# Step 20 - differential_sarsa_update (not yet solved)
# TODO: implement

# Step 21 - run_access_control (not yet solved)
# TODO: implement

# Step 22 - access_control_policy_table (not yet solved)
# TODO: implement

