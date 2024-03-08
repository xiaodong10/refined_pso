
import random
import numpy as np
import os
from statistics import mean
from animation_refined_pso import Anchor, plot_anchors_animation
from map_to_grid import transfer_data_grid
import matplotlib.pyplot as plt



def generate_random_coordinates(num_coordinates, grid_map):
    coordinates = np.zeros((num_coordinates, 2), dtype=int)
    count = 0

    while count < num_coordinates:
        x = random.randint(0, grid_map.shape[1] - 1)
        y = random.randint(0, grid_map.shape[0] - 1)

        # Check if the coordinate maps to a valid location in grid_map
        if not np.isnan(grid_map[y, x]):
            coordinates[count] = [x, y]
            count += 1

    return coordinates

def calculate_total_steps(num_steps, sc, cyc_max):
    
    total_steps = 0
    for iteration in range(1, int(num_steps) + 1):
        step = sc * (np.exp(1 - (iteration / cyc_max)))
        total_steps += step
       
    return total_steps

def Recond_trajectories_value(parameters,grid,num_particles, signal_threshold):
    

    inertia_w_max,inertia_w_min,cognitive_c,social_c,cycle_max,Sc,min_distance_neighbor,radius,max_retries,window_size = parameters 

    particals_cord = generate_random_coordinates(num_particles,grid)
    anchors = [Anchor(tuple(coord), value=grid[coord[1], coord[0]], num_direction=8) for coord in particals_cord]


    # trajectories = [[] for _ in range(len(anchors))]
    # ans_p = [[] for _ in range(cycle_max)]
    trajectories = np.zeros((num_particles, cycle_max, 2))  # Assuming 2D coordinates
    ans_p = np.zeros((cycle_max, num_particles, 2))
    anchors_p = particals_cord.copy() # anchors' current positions
    process_complete = False  # Flag to indicate if the process should be terminated


    for cur_cycle in range(cycle_max):
        if process_complete:
            break
        for idx, anchor in enumerate(anchors):
            anchor.update_global_bests_process(anchors_p, grid, radius)
            params = (inertia_w_max, inertia_w_min, cognitive_c, social_c, cur_cycle, cycle_max, Sc)

            anchor.update_position(params, anchors, grid, min_distance_neighbor,max_retries,window_size)

            # Record the position for this anchor at this time cycle
            trajectories[idx, cur_cycle] = anchor.position.copy()
            # Recond anchors positions in cur_cycle
            # ans_p[cur_cycle].append(anchor.position.copy())
            ans_p[cur_cycle, idx] = anchor.position.copy()
            if anchor.value >= signal_threshold:
                # print(f"success !!!!!{anchor.value}")
                process_complete = True
                break  # Break out of the inner loop

        anchors_p = ans_p[cur_cycle].copy()
       


    # trajectories_value = [[grid[t[1]][t[0]] for t in node] for node in trajectories]
    # trajectories = np.array(trajectories)
    # trajectories_value = np.array(trajectories_value)
        
    trajectories = trajectories[:, :cur_cycle + 1, :]
    ans_p = ans_p[:cur_cycle + 1, :, :]
    # Compute trajectories_value
    # how to store the value?????
    trajectories_value = np.array([[grid[int(t[1]), int(t[0])] for t in trajectory] for trajectory in trajectories.swapaxes(0, 1)])
    trajectories_value = trajectories_value.transpose()
    # print(f"trajectories (num_particles, cycle_max, 2)= {np.shape(trajectories)}")
    # print(f"ans_p is the (cycle_max, num_particles, 2) = {np.shape(ans_p)}")
    # print(f"trajectories_value = {np.shape(trajectories_value)}")

  
    plot_anchors_animation(grid, trajectories, trajectories_value,'plot_refined_pso_animation.gif') 
    return trajectories, trajectories_value, process_complete, cur_cycle



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

def test_algorithm(parameters,times_to_test, grid, num_particles, threshold):
    success_count = 0
    his_steps = np.array([])
    for _ in range(times_to_test):
        p_traj, power_data ,success, steps= Recond_trajectories_value(parameters,grid, num_particles,threshold)
        # print(f"trajectories value= {power_data}")
        # print(f"traj = {p_traj}")
        # print(f"trajectories value shape= {np.shape(power_data)}")
        # print(f"traj shape= {np.shape(p_traj)}")
        # _, step, success = find_particle_reaching_sinal_first(power_data, threshold)
        if success:
            success_count += 1
            his_steps = np.append(his_steps,steps)
        else:
            his_steps = np.append(his_steps,-1)
   
   
    success_rate = success_count / times_to_test

    successful_steps = his_steps[his_steps != -1]  # Filter out unsuccessful steps
    mean_step = np.mean(successful_steps) if successful_steps.size > 0 else None
    std = np.std(successful_steps)
    # print(f"successful_steps = {len(successful_steps)}")
    # print(f"standard_deviation{std}")

    return success_rate, mean_step

def main():

    times_to_test = 10
    file_csv_maps = ['./csv_files/GaussianCampusParkingLot.csv', './csv_files/GaussianCampusStreet.csv','./csv_files/GaussianDentonDowntown.csv']
    # file_csv_maps = ['./csv_files/GaussianCampusStreet.csv']
    inertia_w_max = 0.9
    inertia_w_min = 0.4
    cognitive_c = 2.05
    social_c = 2.05
    Sc = 20
   
    min_distance_neighbor = 2
    radius=50
    max_retries =10
    window_size = 3

    # cy_maxes = [30,60,90,120,150,180,210,240,270,300]  # Different values of Sc to test
    cy_maxes = [150]
    results = {cyc_max: {'success_rates': [], 'average_steps': [], 'total_distance': []} for cyc_max in cy_maxes}
    heatmaps_files = []
    for i in range(len(file_csv_maps)):
        processed_map_file = transfer_data_grid(file_csv_maps[i])
        heatmaps_files.append(processed_map_file)


    for cyc_max in cy_maxes:
        array_success_rates = [[] for _ in heatmaps_files]
        average_steps = [[] for _ in heatmaps_files] 
        total_distance = [[] for _ in heatmaps_files] 
        for i in range(len(heatmaps_files)):
            # num_particles = [5,10,15,20,25]
            num_particles = [15,20]
            grid = np.load(heatmaps_files[i])

            for n_p in num_particles:
                threshold = -33# max is -30db 
                
                para = [inertia_w_max,inertia_w_min,cognitive_c,social_c,cyc_max,Sc,min_distance_neighbor,radius,max_retries,window_size]
                success_rate,av_step = test_algorithm(para,times_to_test, grid, n_p, threshold)
                total_dis = calculate_total_steps(num_steps = av_step, sc=Sc, cyc_max=cyc_max)
                
                array_success_rates[i].append(success_rate)
                average_steps[i].append(av_step)
                total_distance[i].append(total_dis)
                # print(f"av_step{av_step}")
        
        print(f"array_success_rates {array_success_rates}")
        print(f"average_steps {average_steps}")
        results[cyc_max]['success_rates'] = array_success_rates
        results[cyc_max]['average_steps'] = average_steps
        results[cyc_max]['total_distance'] = total_distance

    
        # Print the results
    for cyc_max, data in results.items():
        print(f"Results for cyc_max = {cyc_max}:")
        print("Success Rates:")
        for i, success_rate in enumerate(data['success_rates']):
            print(f"  CSV file {i}: {success_rate}")
        print("Average Steps:")
        for i, avg_step in enumerate(data['average_steps']):
            print(f"  CSV file {i}: {avg_step}")
        print("\n")  # New line for better readability

   
    selected_map_index = 0  # Example: 0 for the first map
    selected_num_particles_index = 0  # Example: 0 for the first num_particles value

    # Plotting success rates for different Sc values for the selected map
    for i, map_name in enumerate(file_csv_maps):
   
        success_rates_for_map = [results[cyc]['success_rates'][i][selected_num_particles_index] for cyc in cy_maxes]
        plt.plot(cy_maxes, success_rates_for_map, marker='o', linestyle='-', label=f'Map: {map_name}')

    # plt.figure(figsize=(10, 6))
    # success_rates = [results[sc]['success_rates'][selected_map_index][selected_num_particles_index] for sc in Sc_values]
    # plt.plot(Sc_values, success_rates, marker='o', linestyle='-')
    plt.xlabel('Max Cycle Time')
    plt.ylabel('Success Rate')
    plt.title(f'Impact of Max Cycle Time(Particle number:{num_particles[selected_num_particles_index]})')
    plt.grid(True)
    plt.legend() 
    plt.show()

    # Plotting average steps for different Sc values for the selected map
    for i, map_name in enumerate(file_csv_maps):
   
        average_steps = [results[cyc]['average_steps'][i][selected_num_particles_index] for cyc in cy_maxes]
        plt.plot(cy_maxes, average_steps, marker='o', linestyle='-', label=f'Map: {map_name}')
    # plt.figure(figsize=(10, 6))
    # average_steps = [results[sc]['average_steps'][selected_map_index][selected_num_particles_index] for sc in Sc_values]
    # plt.plot(Sc_values, average_steps, marker='o', linestyle='-')
    plt.xlabel('Max Cycle Time')
    plt.ylabel('Average Number of Steps')
    plt.title(f'Impact of Max Cycle Time(Particle number:{num_particles[selected_num_particles_index]})')
    plt.grid(True)
    plt.legend() 
    plt.show()

    for i, map_name in enumerate(file_csv_maps):
   
        total_distance = [results[cyc]['total_distance'][i][selected_num_particles_index] for cyc in cy_maxes]
        plt.plot(cy_maxes, total_distance, marker='o', linestyle='-', label=f'Map: {map_name}')
    # plt.figure(figsize=(10, 6))
    # total_distance = [results[sc]['total_distance'][selected_map_index][selected_num_particles_index] for sc in Sc_values]
    # plt.plot(Sc_values, total_distance, marker='o', linestyle='-')
    plt.xlabel('Max Cycle Time')
    plt.ylabel('Distance Traveled')
    plt.title(f'Impact of Max Cycle Time(Particle number:{num_particles[selected_num_particles_index]})')
    plt.grid(True)
    plt.legend() 
    plt.show()
    




if __name__ == '__main__':
    main()
def main():

    times_to_test = 1000
    file_csv_maps = ['./csv_files/CampusParkingLot.csv','./csv_files/CampusStreet.csv','./csv_files/DentonDowntown.csv']
    # file_csv_maps = ['./csv_files/GaussianCampusStreet.csv']
    inertia_w_max = 0.9
    inertia_w_min = 0.4
    cognitive_c = 2.05
    social_c = 2.05
    cycle_max = 200
   
    min_distance_neighbor = 2
    radius=50
    max_retries =10
    window_size = 3

    Sc_values = [2,4,6,8,10,12,14]  # Different values of Sc to test
    results = {sc: {'success_rates': [], 'average_steps': [], 'total_distance': []} for sc in Sc_values}

    for Sc in Sc_values:
        array_success_rates = [[] for _ in file_csv_maps]
        average_steps = [[] for _ in file_csv_maps] 
        total_distance = [[] for _ in file_csv_maps] 
        for i in range(len(file_csv_maps)):
            # num_particles = [5,10,15,20,25]
            num_particles = [15,20]
            grid = transfer_data_grid(file_csv_maps[i])
            grid = grid[100:len(grid)-100, 100:len(grid)-100]

            for n_p in num_particles:
                threshold = -35# max is -30db 
                
                para = [inertia_w_max,inertia_w_min,cognitive_c,social_c,cycle_max,Sc,min_distance_neighbor,radius,max_retries,window_size]
                success_rate,av_step = test_algorithm(para,times_to_test, grid, n_p, threshold)
                total_dis = calculate_total_steps(num_steps = av_step, sc=Sc, cyc_max=cycle_max)
                
                array_success_rates[i].append(success_rate)
                average_steps[i].append(av_step)
                total_distance[i].append(total_dis)
                # print(f"av_step{av_step}")
        
        print(f"array_success_rates {array_success_rates}")
        print(f"average_steps {average_steps}")
        results[Sc]['success_rates'] = array_success_rates
        results[Sc]['average_steps'] = average_steps
        results[Sc]['total_distance'] = total_distance

    
        # Print the results
    for Sc, data in results.items():
        print(f"Results for Sc = {Sc}:")
        print("Success Rates:")
        for i, success_rate in enumerate(data['success_rates']):
            print(f"  CSV file {i}: {success_rate}")
        print("Average Steps:")
        for i, avg_step in enumerate(data['average_steps']):
            print(f"  CSV file {i}: {avg_step}")
        print("\n")  # New line for better readability

   
    selected_map_index = 0  # Example: 0 for the first map
    selected_num_particles_index = 0  # Example: 0 for the first num_particles value

    # Plotting success rates for different Sc values for the selected map
    for i, map_name in enumerate(file_csv_maps):
   
        success_rates_for_map = [results[sc]['success_rates'][i][selected_num_particles_index] for sc in Sc_values]
        plt.plot(Sc_values, success_rates_for_map, marker='o', linestyle='-', label=f'Map: {map_name}')

    # plt.figure(figsize=(10, 6))
    # success_rates = [results[sc]['success_rates'][selected_map_index][selected_num_particles_index] for sc in Sc_values]
    # plt.plot(Sc_values, success_rates, marker='o', linestyle='-')
    plt.xlabel('Sc Value')
    plt.ylabel('Success Rate')
    plt.title(f'Impact of Scale Factor on Success Rate Through Initial Steps (Map: {file_csv_maps[selected_map_index]}, Particles: {num_particles[selected_num_particles_index]})')
    plt.grid(True)
    plt.show()

    # Plotting average steps for different Sc values for the selected map
    for i, map_name in enumerate(file_csv_maps):
   
        average_steps = [results[sc]['average_steps'][i][selected_num_particles_index] for sc in Sc_values]
        plt.plot(Sc_values, average_steps, marker='o', linestyle='-', label=f'Map: {map_name}')
    # plt.figure(figsize=(10, 6))
    # average_steps = [results[sc]['average_steps'][selected_map_index][selected_num_particles_index] for sc in Sc_values]
    # plt.plot(Sc_values, average_steps, marker='o', linestyle='-')
    plt.xlabel('Sc Value')
    plt.ylabel('Average Steps')
    plt.title(f'Impact of Scale Factor on Average Step Through Initial Steps (Map: {file_csv_maps[selected_map_index]}, Particles: {num_particles[selected_num_particles_index]})')
    plt.grid(True)
    plt.show()

    for i, map_name in enumerate(file_csv_maps):
   
        total_distance = [results[sc]['total_distance'][i][selected_num_particles_index] for sc in Sc_values]
        plt.plot(Sc_values, total_distance, marker='o', linestyle='-', label=f'Map: {map_name}')
    # plt.figure(figsize=(10, 6))
    # total_distance = [results[sc]['total_distance'][selected_map_index][selected_num_particles_index] for sc in Sc_values]
    # plt.plot(Sc_values, total_distance, marker='o', linestyle='-')
    plt.xlabel('Sc Value')
    plt.ylabel('Total_Distance')
    plt.title(f'Impact of Scale Factor on Total Distance Through Initial Steps (Map: {file_csv_maps[selected_map_index]}, Particles: {num_particles[selected_num_particles_index]})')
    plt.grid(True)
    plt.show()
    




if __name__ == '__main__':
    main()
