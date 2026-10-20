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

# Step 2 - maze_step (not yet solved)
# TODO: implement

# Step 3 - epsilon_greedy_tabular (not yet solved)
# TODO: implement

# Step 4 - q_learning_update (not yet solved)
# TODO: implement

# Step 5 - planning_updates (not yet solved)
# TODO: implement

# Step 6 - dyna_q_episode (not yet solved)
# TODO: implement

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

