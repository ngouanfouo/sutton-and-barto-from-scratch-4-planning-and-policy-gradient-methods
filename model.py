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

# Step 7 - greedy_path_length
import numpy as np


def greedy_path_length(maze, q_table, max_steps=100):
    state = maze['start']
    for step in range(1, max_steps + 1):
        action = int(np.argmax(q_table[state]))
        state, reward, done = maze_step(maze, state, action)
        if done:
            return step
    return max_steps

# Step 8 - update_model_plus
def update_model_plus(model, state, action, next_state, reward, time, n_actions=4):
    new_model = dict(model)
    for a in range(n_actions):
        new_model.setdefault((state, a), (state, 0.0, 0))
    new_model[(state, action)] = (next_state, reward, time)
    return new_model

# Step 9 - dyna_q_plus_planning
import numpy as np


def dyna_q_plus_planning(q_table, model, n_planning, time, kappa, alpha, gamma, rng):
    keys = list(model.keys())
    if not keys or n_planning <= 0:
        return q_table.copy()

    indices = rng.integers(0, len(keys), size=n_planning)
    for i in indices:
        state, action = keys[i]
        next_state, reward, last_time = model[(state, action)]
        bonus = kappa * np.sqrt(time - last_time)
        q_table = q_learning_update(q_table, state, action, reward + bonus, next_state, alpha, gamma)
    return q_table

# Step 10 - run_blocking_maze
import numpy as np


def run_blocking_maze(use_plus, max_steps, change_step, n_planning, kappa, alpha, gamma, epsilon, rng):
    rows, cols = 6, 9
    start = (5, 3)
    goal = (0, 8)
    initial_walls = [(3, c) for c in range(8)]
    new_walls = [(3, c) for c in range(1, 9)]

    maze = build_maze(initial_walls, start=start, goal=goal, rows=rows, cols=cols)

    q_table = np.zeros((rows, cols, 4))
    model = {}
    state = maze['start']
    cumulative = np.zeros(max_steps, dtype=float)
    total = 0.0

    for t in range(1, max_steps + 1):
        # Wall moves after change_step: rebuild the layout, keep position/Q/model.
        if t == change_step + 1:
            maze = build_maze(new_walls, start=start, goal=goal, rows=rows, cols=cols)

        action = epsilon_greedy_tabular(q_table, state, epsilon, rng)
        next_state, reward, done = maze_step(maze, state, action)
        q_table = q_learning_update(q_table, state, action, reward, next_state, alpha, gamma)

        if use_plus:
            model = update_model_plus(model, state, action, next_state, reward, t, 4)
            q_table = dyna_q_plus_planning(q_table, model, n_planning, t, kappa, alpha, gamma, rng)
        else:
            model[(state, action)] = (next_state, reward)
            q_table = planning_updates(q_table, model, n_planning, alpha, gamma, rng)

        total += reward
        cumulative[t - 1] = total

        if done:
            state = maze['start']
        else:
            state = next_state

    return cumulative

# Step 11 - push_priority
import numpy as np
import heapq


def push_priority(queue, q_table, state, action, reward, next_state, gamma, theta):
    priority = abs(reward + gamma * np.max(q_table[next_state]) - q_table[state][action])
    if priority > theta:
        heapq.heappush(queue, (-float(priority), tuple(state), int(action)))
    return queue

# Step 12 - prioritized_sweeping_episode
import numpy as np
import heapq


def prioritized_sweeping_episode(maze, q_table, model, n_planning, theta, alpha, gamma, epsilon, rng):
    model = dict(model)
    queue = []
    state = maze['start']
    n_steps = 0
    n_updates = 0
    done = False

    while not done:
        action = epsilon_greedy_tabular(q_table, state, epsilon, rng)
        next_state, reward, done = maze_step(maze, state, action)

        # Real experience only feeds the model and the priority queue.
        model[(state, action)] = (next_state, reward)
        push_priority(queue, q_table, state, action, reward, next_state, gamma, theta)

        # Planning: focused backward wave of up to n_planning backups.
        for _ in range(n_planning):
            if not queue:
                break
            _, s, a = heapq.heappop(queue)
            s_next, r = model[(s, a)]
            q_table = q_learning_update(q_table, s, a, r, s_next, alpha, gamma)
            n_updates += 1
            for (ps, pa), (pn, pr) in model.items():
                if pn == s:
                    push_priority(queue, q_table, ps, pa, pr, s, gamma, theta)

        state = next_state
        n_steps += 1

    return q_table, model, n_steps, n_updates

# Step 13 - corridor_step
def corridor_step(state, action):
    if state == 1:
        action = 1 - action
    if action == 0:
        next_state = max(state - 1, 0)
    else:
        next_state = state + 1
    return next_state, -1.0, next_state == 3

# Step 14 - corridor_policy
import numpy as np


def corridor_policy(theta):
    theta = np.asarray(theta, dtype=float)
    shifted = theta - np.max(theta)
    exp_theta = np.exp(shifted)
    return exp_theta / np.sum(exp_theta)

# Step 15 - corridor_true_value
def corridor_true_value(p_right):
    return float(-2.0 * (2.0 - p_right) / (p_right * (1.0 - p_right)))

# Step 16 - reinforce_episode
import numpy as np


def reinforce_episode(theta, alpha, gamma, rng, max_steps=10000):
    # TODO: Sample an episode (at most max_steps transitions), compute G_t backward, apply theta += alpha*gamma^t*G_t*(onehot - probs) for each t
    pass
import numpy as np


def reinforce_episode(theta, alpha, gamma, rng, max_steps=10000):
    theta = np.array(theta, dtype=float)  # copy, don't mutate caller's array

    state = 0
    actions = []
    rewards = []
    done = False

    while not done and len(actions) < max_steps:
        probs = corridor_policy(theta)
        action = int(rng.choice(2, p=probs))
        next_state, reward, done = corridor_step(state, action)
        actions.append(action)
        rewards.append(reward)
        state = next_state

    T = len(actions)

    # Backward pass: G_t = R_t + gamma * G_{t+1}
    returns = np.zeros(T)
    g = 0.0
    for t in range(T - 1, -1, -1):
        g = rewards[t] + gamma * g
        returns[t] = g

    # Forward pass: sequential updates using current preferences at each t
    for t in range(T):
        probs = corridor_policy(theta)
        grad = -probs.copy()
        grad[actions[t]] += 1.0
        theta = theta + alpha * (gamma ** t) * returns[t] * grad

    return theta, float(np.sum(rewards))

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

