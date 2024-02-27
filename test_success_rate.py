
import random
import numpy as np
import os
from statistics import mean
from animation_refined_pso import Anchor, plot_anchors_animation
from map_to_grid import transfer_data_grid
import matplotlib.pyplot as plt


def generate_random_coordinates(num_coordinates, grid_map):
    coordinates = []
    while len(coordinates) < num_coordinates:
        x = random.randint(0, len(grid_map)-1)
        y = random.randint(0, len(grid_map)-1)
        
        # Check if the coordinate maps to a valid location in grid_map
        if not np.isnan(grid_map[y, x]):
            coordinates.append([x, y])

    return coordinates

def Recond_trajectories_value(map_csv_path,num_particles, signal_threshold):
    grid = transfer_data_grid(map_csv_path)
    # print(f"grid.shape = {grid.shape}")
    grid = grid[100:len(grid)-100, 100:len(grid)-100]
    # print(f"grid.shape = {grid.shape}")

    # to find this map max signal strength value to set the threshold
    # check_m_grid = grid
    # if np.isnan(grid).any():       
    #     check_m_grid[np.isnan(check_m_grid)] = -70  # Replace NaNs with 0 in this example
    #     max_value = np.amax(check_m_grid)
    #     min_value = np.amin(check_m_grid)
    #     print(f"max_power{max_value}")
    #     print(f"min_value{min_value}")

    inertia_w_max=0.9 # max=0.9 min = 0.4 by default
    inertia_w_min=0.4
    cognitive_c=2 #cognitive_c,social_c=2 by default
    social_c=2
    cur_cycle=1
    cycle_max=70
    # Tc=5
    Sc = 7
    min_distance_neighbor = 1
    radius = 50
    max_retries = 10
    window_size = 3 # the historical data size 
    Found_signal_source = False

    particals_cord = generate_random_coordinates(num_particles,grid)
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

  
    # plot_anchors_animation(grid, trajectories, trajectories_value,'plot_refined_pso_animation.gif') 
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
   
    # print(f'his_steps={his_steps}')

    # Extracting the base name
    base_name = os.path.splitext(os.path.basename(map_csv_path))[0]
    # Constructing the new file name
    npy_file_name = f'./csv_files/{base_name}_test_his.npy'
    np.save(npy_file_name,his_steps)
    historical_steps = np.load(npy_file_name)

    # Print the contents of the array
    # print("Contents of the .npy file:")
    # print(historical_steps)
    success_rate = success_count / times_to_test

    successful_steps = his_steps[his_steps != -1]  # Filter out unsuccessful steps
    mean_step = np.mean(successful_steps) if successful_steps.size > 0 else None
    std = np.std(successful_steps)
    print(f"successful_steps = {successful_steps}")
    print(f"standard_deviation{std}")

    return success_rate, mean_step

def main():

    times_to_test = 300
    # csv_map = './ts_map3.csv'
    # csv_map = './csv_files/poster_map1.csv'
    # file_csv_maps = ['./csv_files/poster_map1.csv','./csv_files/poster_map2.csv','./csv_files/poster_map3.csv']
    file_csv_maps = ['./csv_files/poster_map2.csv']

    array_success_rates = [[] for _ in file_csv_maps]   
    for i in range(len(file_csv_maps)):
        # num_particles = [5,10,15,20,25]
        num_particles = [20]
        print(f"map is {file_csv_maps[i]}")

        for n_p in num_particles:
            threshold = -47# max is -32db in poster_map1, -183db min
            success_rate,av_step = test_algorithm(times_to_test, file_csv_maps[i], n_p, threshold)
            array_success_rates[i].append(success_rate)
            # print(f"times_to_test = {times_to_test}   num_particles = {n_p}   Rate: {success_rate * 100}%, av_step:{av_step}")

    print(f"array_success_rates {array_success_rates}")

    # plot the success rate
    markers = ['o', 's', '^']
    line_styles = ['-', '--', ':']
    for i, success_rates in enumerate(array_success_rates):
        marker = markers[i % len(markers)]  # Cycle through markers
        line_style = line_styles[i % len(line_styles)] 
        plt.plot(num_particles, success_rates, marker=marker, linestyle=line_style, label=f'Map {i + 1}')

    # Add labels and legend
    plt.xlabel('Number of Particles')
    plt.ylabel('Success Rate')
    plt.legend(loc='best')  # Add a legend in the best location
    plt.grid(True)

    # Show the plot
    plt.title('Success Rate for Different Maps and Particles')
    plt.show()



if __name__ == '__main__':
    main()
# 500 samples, 10 particles, Success Rate: 51.2%, av_step:20.150390625
#  1000 samples, 10 particles,  Success Rate: 47.4%, av_step:20.421940928270043
# 1000 times, 15 particles  Success Rate: 67.10000000000001%, av_step:19.341281669150522
# 1000 times, 20 particles     Success Rate: 83.0%, av_step:16.14457831325301  
# 1000 times, 25 particles  Success Rate: 90.4%, av_step:14.847345132743364

#   csv_map = './ts_map1.csv' , hard
#   times_to_test = 1000  num_particles = 5,     Success Rate: 35.099999999999994%, av_step:18.36182336182336
#   times_to_test = 1000  num_particles = 10,    Success Rate: 64.0%, av_step:15.0984375
#   times_to_test = 1000  num_particles = 15,    Success Rate: 80.5%, av_step:12.698136645962732
#   times_to_test = 1000  num_particles = 20,    Success Rate: 91.9%, av_step:11.486398258977149  
#   times_to_test = 1000  num_particles = 25,    Success Rate: 96.3%, av_step:9.499480789200415
    


#   csv_map = './ts_map2.csv' , easy
#   times_to_test = 1000  num_particles = 5,     Success Rate: 40.0%, av_step:17.155
#   times_to_test = 1000  num_particles = 10,    Success Rate: 73.4%, av_step:15.215258855585832
#  times_to_test = 1000   num_particles = 15     Rate: 86.7%, av_step:11.675893886966552
#  times_to_test = 1000   num_particles = 20    Rate: 93.5%, av_step:10.935828877005347
#   times_to_test = 1000   num_particles = 25   Rate: 97.0%, av_step:8.596907216494845




#   csv_map = './ts_map3.csv' , 
#   times_to_test = 1000   num_particles = 5   Rate: 34.300000000000004%, av_step:17.03206997084548
#   times_to_test = 1000   num_particles = 10   Rate: 65.0%, av_step:15.503076923076923
#   times_to_test = 1000   num_particles = 15   Rate: 81.39999999999999%, av_step:13.71007371007371
#   times_to_test = 1000   num_particles = 20   Rate: 91.60000000000001%, av_step:11.555676855895197
#   times_to_test = 1000   num_particles = 25   Rate: 96.2%, av_step:9.23908523908524