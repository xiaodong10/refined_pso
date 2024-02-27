import math
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation
from scipy.interpolate import griddata,Rbf
from source_map_generator import generate_map,plot_map,plot_3d
from map_to_grid import transfer_data_grid
from plot_animation import plot_anchors_animation

# map_csv_path = './csv_files/pd_data.csv'
def get_anchor_value_by_position(grid, position):
   
    return grid[int(position[1]), int(position[0])]


def choose_random_direction(num_direction=8):
    """Generate a direction vector based on num_direction possiable direction"""
    step_angle = 2 * np.pi / num_direction
    dirt_index = np.random.choice(num_direction)
    angle = dirt_index * step_angle
    dirt_angle = np.array([np.cos(angle), np.sin(angle)])
    magnitude = np.linalg.norm(dirt_angle)
    norm_drection = dirt_angle / magnitude
    return norm_drection



class Anchor:
    def __init__(self, position, value, num_direction):
        self.position = np.array(position)
        self.value = value
        self.previous_best_value = value
        self.previous_best_position = np.array(position).copy()
        self.global_best_value = float("-inf")
        self.global_best_position = np.array([])
        self.step = choose_random_direction(num_direction)
        self.historical_neighbors = [] #[[t1's self.position, neighbors_positions...],[t2's self...,nei...],[]]


    # def PSO_determine_v_direction(self, inertia_w_max, inertia_w_min, cognitive_c, social_c, cur_cycle, cycle_max, S_c):
    #     S = S_c*(np.exp(1-(cur_cycle/cycle_max)))
    #     inertia_w = inertia_w_max - (inertia_w_max-inertia_w_min)/cycle_max*cur_cycle

    #     # rand_0 = np.random.uniform(0.5, 1.5)
    #     rand_0 = 1
    #     rand_1 = np.random.uniform(0, 1)
    #     rand_2 = np.random.uniform(0, 1)

    #     personal_best_diff = self.previous_best_position-self.position
    #     global_best_diff = self.global_best_position-self.position

    #     # For personal_best_diff
    #     if np.linalg.norm(personal_best_diff) < 0.1:
    #         personal_best_component = np.array([0, 0])
    #     else:
    #         personal_best_component = (personal_best_diff/np.linalg.norm(personal_best_diff))

    #     # For global_best_diff
    #     if np.linalg.norm(global_best_diff) < 0.1:
    #         global_best_component = np.array([0, 0])
    #     else:
    #         global_best_component = (global_best_diff/np.linalg.norm(global_best_diff))

    #     d_v = inertia_w * rand_0 * self.step + cognitive_c * rand_1 * personal_best_component + social_c * rand_2 * global_best_component
    #     magnitude = np.linalg.norm(d_v)
    #     if magnitude == 0:  # Avoid division by zero
    #         norm_velocity = choose_random_direction(num_direction=8) 
    #     else:
    #         norm_velocity = d_v/magnitude
    #     next_move = S*norm_velocity
    #     next_move_int = np.rint(next_move).astype(int)
    #     return norm_velocity, next_move_int
    
    def PSO_determine_v_direction(self, inertia_w_max, inertia_w_min, cognitive_c, social_c, cur_cycle, cycle_max, S_c):
        S = S_c*(np.exp(1-(cur_cycle/cycle_max)))
        # S = S_c
        inertia_w = inertia_w_max - (inertia_w_max-inertia_w_min)/cycle_max*cur_cycle

        # rand_0 = np.random.uniform(0.5, 1.5)
        rand_0 = 1
        rand_1 = np.random.uniform(0, 1)
        rand_2 = np.random.uniform(0, 1)

        d_v = inertia_w * rand_0 * self.step + cognitive_c * rand_1 * (self.previous_best_position - self.position) + social_c * rand_2 * (self.global_best_position - self.position)
        magnitude = np.linalg.norm(d_v)
        

        
        if (self.value==-200 or self.global_best_value == -200) or magnitude == 0: # Avoid division by zero and encourage explore when -inf
            norm_velocity = choose_random_direction(num_direction=8)
        else:
            norm_velocity = d_v/magnitude
        next_move = S*norm_velocity
        next_move_int = np.rint(next_move).astype(int)
        return d_v, next_move_int
    

    def PSO_determine_norm_v_direction(self, inertia_w_max, inertia_w_min, cognitive_c, social_c, cur_cycle, cycle_max, S_c):
        S = S_c*(np.exp(1-(cur_cycle/cycle_max)))
        # S = S_c
        inertia_w = inertia_w_max - (inertia_w_max-inertia_w_min)/cycle_max*cur_cycle

        # rand_0 = np.random.uniform(0.5, 1.5)
        rand_0 = 1
        rand_1 = np.random.uniform(0, 1)
        rand_2 = np.random.uniform(0, 1)

        dir_p_best = self.previous_best_position - self.position
        norm_p = np.linalg.norm(dir_p_best)
        dir_g_best = self.global_best_position - self.position
        norm_g = np.linalg.norm(dir_g_best)

        norm_p_best = dir_p_best/norm_p if norm_p != 0 else dir_p_best
        norm_g_best = dir_g_best/norm_g if norm_g != 0 else dir_g_best

        d_v = inertia_w * rand_0 * self.step + cognitive_c * rand_1 * norm_p_best + social_c * rand_2 * norm_g_best
        magnitude = np.linalg.norm(d_v)

        if (self.value==-200 or self.global_best_value == -200) or magnitude == 0: # Avoid division by zero and encourage explore when -inf
            norm_velocity = choose_random_direction(num_direction=8)
        else:
            norm_velocity = d_v/magnitude
        next_move = S*norm_velocity
        next_move_int = np.rint(next_move).astype(int)
        return norm_velocity, next_move_int
       

    def update_position(self, params, all_anchors,grid, min_distance=2.0, max_retries=10, window_size=3):

        # Check if the current value is better than the previous best value
        if self.value > self.previous_best_value:
            self.previous_best_value = self.value
            self.previous_best_position = self.position.copy()

        # Determine the next step and check weather it would collision, if does, re_calculate the next step till max_retries times or not collision.
        for _ in range(max_retries):
            # Problem of retries
    
            self.step,next_move = self.PSO_determine_norm_v_direction(*params)
            new_position = self.position + next_move
            if 0 <= new_position[0] < grid.shape[1] and 0 <= new_position[1] < grid.shape[0] and not math.isnan(grid[new_position[1]][new_position[0]]):
                # print(f"new_position{new_position}")

                if not self.is_collision(new_position, all_anchors, min_distance):
                    self.position = new_position
                    self.value = grid[new_position[1]][new_position[0]] 
                    break
            self.step = np.array([0, 0])


    def is_collision(self, new_position, all_anchors, min_distance):
        for a in all_anchors:
            diff = new_position - a.position
            if np.linalg.norm(diff) < min_distance:
                return True
        return False

    

    def find_neighbors(self, all_anchors_position, radius):
        # Find all neighbors within a certain radius, add anchor itself in the first position
        current_neighbors = []

        for anchor_position in all_anchors_position:
            if not np.array_equal(self.position, anchor_position):
                distance = np.linalg.norm(self.position - anchor_position)
                if distance <= radius:
                    current_neighbors.append(anchor_position)

        neighbor_positions_with_self = [self.position.tolist()] + [neighbor for neighbor in current_neighbors]

        self.historical_neighbors.append(neighbor_positions_with_self)





    def update_global_best(self, grid):
        """
        The global best is not real global, it's local best(neighbors' best)
        """
        # grab the latest neighbors including itself
        latest_neighbors_positions = self.historical_neighbors[-1]
        best_value = self.value
        best_position = self.position.copy()

        for neighbor_position in latest_neighbors_positions:
            # print(f"neighbor_position{neighbor_position}")
  
            neighbor_value = get_anchor_value_by_position(grid, neighbor_position)
            
            if neighbor_value > best_value:
                best_value = neighbor_value
                best_position = neighbor_position

        self.global_best_value = best_value
        self.global_best_position = best_position.copy()

    def update_global_bests_process(self, anchors_p, grid, radius):  
        self.find_neighbors(anchors_p, radius)
        self.update_global_best(grid)

    def print_value(self):
        print(self.value)


def main():
    # # the ideal rf signal
    # parameter_of_features = [[150,100],-np.pi,-40,0,2,1,np.pi/16]
    # grid = generate_map(parameter_of_features, [200,200])
    # # plot the original ideal rf signal
    # plot_map(grid)
    # plot_3d(grid)

    # monitor real environment
    # map_csv_path = './csv_files/pd_data.csv'
    map_csv_path = './campus_parking_lot.csv'
    grid = transfer_data_grid(map_csv_path)
    # print(np.shape(grid))

    inertia_w_max=0.9 # max=0.9 min = 0.4 by default
    inertia_w_min=0.4
    cognitive_c=2 #cognitive_c,social_c=2 by default
    social_c=2
    cur_cycle=1
    cycle_max=100
    # Tc=5
    Sc =6
    min_distance_neighbor = 3
    radius = 50
    max_retries = 10
    window_size = 3 # the historical data size 
    signal_threshold = -27 # signal strength of locating the source
    Found_signal_source = False # one partile find the source, this flag turn into True, end of collecting data

    anchors_coords = [[182,285], [259,187],[70,112],[60,287],[465,320],[363,290],[267,399],[476,190],[261,408],[466,217],[364,280],[170,498],[78,495],[75,384]]
    # anchors_coords = [[82,95], [79,87],[34,60],[40,50],[67,90],[30,50],[50,30],[120,60],[170,70],[160,50]
    # anchors_coords = [[20,60],[90,70],[60,50],[70,40],[90,30]]
    # anchors_coords = [[60,30],[100,90],[70,80]]
    # anchors_coords = [[160,120],[150,160],[80,50],[70,90],[68,80],[50,140]]
    # anchors_coords = [[200,122],[170,200],[93,189],[123,60],[173,38],[173,38],[200,42]] # good
    # anchors_coords = [[193,106],[160,210],[215,126],[155,170],[100,200],[105,195],[150,160]] #'C:/Users/xdwun/Research/Codes_work/AI_army/anchors_animation.gif
    # anchors_coords = [[193,106],[160,210],[215,126],[171,105],[146,60]]
    # anchors_coords = [[160,210],[215,126],[171,105],[56,200],[92,180],[85,205]]

    # anchors_coords = [[150,60],[160,100],[90,120],[80,70]]


    anchors = [Anchor(tuple(coord), value=grid[coord[1]][coord[0]], num_direction=8) for coord in anchors_coords]

    trajectories = [[] for _ in range(len(anchors))]
    ans_p = [[] for _ in range(cycle_max)]
    anchors_p = anchors_coords.copy()
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
        # print(f"anchors_p={anchors_p}")
        cur_cycle += 1
        if Found_signal_source:
            break


    trajectories_value = [[grid[t[1]][t[0]] for t in node] for node in trajectories]
    trajectories = np.array(trajectories)
    trajectories_value = np.array(trajectories_value)    
    plot_anchors_animation(grid, trajectories, trajectories_value,'plot_refined_pso_animation.gif')
   

if __name__ == '__main__':
    main()
