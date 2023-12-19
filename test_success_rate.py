
import random
import numpy as np
import os
from statistics import mean
from animation_refined_pso import Anchor, plot_anchors_animation
from map_to_grid import transfer_data_grid

def generate_random_coordinates(num_coordinates):
    coordinates = []
    for _ in range(num_coordinates):
        x = random.randint(0, 255)
        y = random.randint(0, 255)
        coordinates.append([x, y])
    return coordinates


def Recond_trajectories_value(map_csv_path,num_particles, signal_threshold):
    grid = transfer_data_grid(map_csv_path)

    inertia_w_max=0.9 # max=0.9 min = 0.4 by default
    inertia_w_min=0.4
    cognitive_c=2 #cognitive_c,social_c=2 by default
    social_c=2
    cur_cycle=1
    cycle_max=50
    # Tc=5
    Sc =6
    min_distance_neighbor = 3
    radius = 30
    max_retries = 10
    window_size = 3 # the historical data size 
    Found_signal_source = False

    particals_cord = generate_random_coordinates(num_particles)
    anchors = [Anchor(tuple(coord), value=grid[coord[1]][coord[0]], num_direction=8) for coord in particals_cord]

    trajectories = [[] for _ in range(len(anchors))]
    ans_p = [[] for _ in range(cycle_max)]
    anchors_p = particals_cord.copy()

    for cur_cycle in range(cycle_max):
        for idx, anchor in enumerate(anchors):
            anchor.update_global_bests_process(anchors_p, grid, radius)
            params = (inertia_w_max, inertia_w_min, cognitive_c, social_c, cur_cycle, cycle_max, Sc)

            anchor.update_position(params, anchors, grid, min_distance_neighbor,max_retries,window_size)

            # Record the position for this anchor at this time cycle
            trajectories[idx].append(anchor.position.copy())
            # Recond anchors positions in cur_cycle
            ans_p[cur_cycle].append(anchor.position.copy())
            if anchor.value >= signal_threshold:
                Found_signal_source = True

        anchors_p = ans_p[cur_cycle]
        cur_cycle += 1
        if Found_signal_source:
            break


    trajectories_value = [[grid[t[1]][t[0]] for t in node] for node in trajectories]
    trajectories = np.array(trajectories)
    trajectories_value = np.array(trajectories_value)

  
    plot_anchors_animation(grid, trajectories, trajectories_value,'plot_refined_pso_animation.gif') 
    return trajectories, trajectories_value



# def find_particle_reaching_sinal_first(power_data, threshold):
#     num_steps = len(power_data[0])

#     for step in range(num_steps):
#         for particle_index in range(len(power_data)):
#             if power_data[particle_index][step] >= threshold:
#                 return particle_index, step, True

#     return None, None, False  # No particle reaches the threshold

def determine_locating_steps(power_data, threshold):
    # determine weather it locate the signal source and the steps, if fails, steps = -1
    steps = len(power_data[0])

    # Extract the last movement sensing power for each particle
    last_values = [row[-1] for row in power_data if len(row) > 0]
    max_last_value = max(last_values) if last_values else None
    if max_last_value >= threshold:
        return steps, True
    else:
        return -1, False

def test_algorithm(times_to_test, map_csv_path, num_particles, threshold):
    success_count = 0
    his_steps = np.array([])
    for _ in range(times_to_test):
        _, power_data = Recond_trajectories_value(map_csv_path, num_particles,threshold)
        # _, step, success = find_particle_reaching_sinal_first(power_data, threshold)
        step,success = determine_locating_steps(power_data, threshold)
        if success:
            success_count += 1
            his_steps = np.append(his_steps,step)
        else:
            his_steps = np.append(his_steps,-1)
    print(f'his_steps={his_steps}')

    # Extracting the base name
    base_name = os.path.splitext(os.path.basename(map_csv_path))[0]
    # Constructing the new file name
    npy_file_name = f'./csv_files/{base_name}_historical_steps.npy'
    np.save(npy_file_name,his_steps)
    historical_steps = np.load(npy_file_name)

    # Print the contents of the array
    print("Contents of the .npy file:")
    print(historical_steps)
    success_rate = success_count / times_to_test

    successful_steps = his_steps[his_steps != -1]  # Filter out unsuccessful steps
    mean_step = np.mean(successful_steps) if successful_steps.size > 0 else None

    return success_rate, mean_step

def main():

    times_to_test = 10
    csv_map = './csv_files/map6.csv' 
    num_particles = 10
    threshold = -30
    success_rate,av_step = test_algorithm(times_to_test, csv_map, num_particles, threshold)
    print(f"Success Rate: {success_rate * 100}%, av_step:{av_step}")



if __name__ == '__main__':
    main()
