import pandas as pd
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree
import matplotlib.animation as animation
import pickle

def periodic_boundaries(pos, boundary):
    pos[0] = ((pos[0] + boundary) % (2 * boundary)) - boundary
    pos[1] = ((pos[1] + boundary) % (2 * boundary)) - boundary
    return pos

def levy_alpha_stable(alpha, beta, mu, c, delta_t):
    theta = np.random.uniform(-np.pi/2, np.pi/2)
    W = np.random.exponential(1)
    if W == 0:
        W = np.finfo(float).eps
    numerator = np.sin(alpha * theta)
    denominator = (np.cos(theta)) ** (1/alpha)
    cos_term = np.cos((1 - alpha) * theta)
    cos_term = np.abs(cos_term)
    with np.errstate(divide='ignore', invalid='ignore'):
        correction_factor = (cos_term/W) ** ((1 - alpha)/alpha)
    step_length = numerator/denominator * correction_factor
    if np.isnan(step_length) or np.isinf(step_length):
        step_length = 0.0
    return c * step_length + mu * delta_t

def levy_flight_step_with_angle(alpha, beta, mu, c, delta_t):
    while True:
        jump_length = np.abs(levy_alpha_stable(alpha, beta, mu, c, delta_t))
        if jump_length <= 1e7:
            break
    angle = np.random.uniform(0, 2*np.pi)
    dx = jump_length * np.cos(angle)
    dy = jump_length * np.sin(angle)
    return dx, dy, jump_length

def compute_msd(positions):
    pos_array = np.array(positions)
    deltas = pos_array - pos_array[0]
    squared_deltas = deltas[:, 0]**2 + deltas[:, 1]**2
    return squared_deltas

# Parameters
alpha = 1.4
beta = 0
mu = 0
c = 1.0
delta_t = 1
max_time = 100000
vision_radius = 3.0
num_runs = 6000
boundary = 12000
num_targets = 600

# Target setup
grid_size = int(np.sqrt(num_targets))
x_coords = np.linspace(-boundary, boundary, grid_size)
y_coords = np.linspace(-boundary, boundary, grid_size)
targets = np.array([[x, y] for x in x_coords for y in y_coords])
target_tree = cKDTree(targets)

# Starting position
grid_size = int(np.sqrt(len(targets)))
middle_index = (grid_size//2) * grid_size + (grid_size//2)
starting_target_pos = targets[middle_index]

# Data storage
all_run_data = []
all_msd = []

# Simulation loop
for run_idx in tqdm(range(num_runs), desc="Simulating Runs", unit="run"):
    walker_pos = starting_target_pos.copy()
    step_count = 0
    total_distance = 0.0
    distances_this_run = []
    trajectory_this_run = [walker_pos.copy()]
    times_return_to_original = []
    times_hit_other_targets = []
    visited_targets = {middle_index}
    first_return_time = None
    first_hit_time = None
    while step_count < max_time:
        dx_total, dy_total, jump_length = levy_flight_step_with_angle(alpha, beta, mu, c, delta_t)
        dd = np.linalg.norm([dx_total, dy_total])
        
        num_increments = max(1, int(dd))
        while dd/num_increments >= 1.0:
            num_increments += 1

        dx_step = dx_total/num_increments
        dy_step = dy_total/num_increments

        detected = False

        for _ in range(num_increments):
            walker_pos[0] += dx_step
            walker_pos[1] += dy_step
            walker_pos = periodic_boundaries(walker_pos, boundary)

            total_distance += np.linalg.norm([dx_step, dy_step])
            distances_this_run.append(total_distance)
            step_count += 1
            trajectory_this_run.append(walker_pos.copy())
            
            dist_to_nearest_target, idx_nearest_target = target_tree.query(walker_pos)
            if dist_to_nearest_target <= vision_radius :
                if (idx_nearest_target == middle_index and step_count > 3) and first_return_time is None:
                    first_return_time = step_count
                elif idx_nearest_target != middle_index and first_hit_time is None:
                    first_hit_time = step_count
                    visited_targets.add(idx_nearest_target)
            
            if first_return_time is not None and first_hit_time is not None:
                break

            if step_count >= max_time:
                break

        
        if first_return_time is not None and first_hit_time is not None:
            break
        if step_count >= max_time:
            break
        
    all_run_data.append({
        "run_index": run_idx,
        
        "first_return_time": first_return_time,
        "first_hit_time": first_hit_time,

        "visited_targets": visited_targets,
    })



    print(f"Run {run_idx + 1}/{num_runs}: "
          f"First Return = {first_return_time}, "
          f"First Hit = {first_hit_time}")

with open("simulation_results.pkl", "wb") as f:
    pickle.dump(all_run_data, f)
