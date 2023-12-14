import math
import matplotlib.pyplot as plt
import numpy as np
import json
import os
import source_map_generator
from matplotlib.animation import FuncAnimation
from scipy.interpolate import griddata,Rbf
# from scipy.interpolate import Rbf
# def choose_random_direction():
#     ### need to change the ininial v, should have a v in time 0 and the best_neighbor is itself.. Update the psedocode.
#     directions = np.array([[0,1], [0,-1], [1,0], [1,1], [1,-1], [-1,0], [-1,-1], [-1,1]])
#     index = np.random.choice(directions.shape[0])
#     return directions[index]

def get_anchor_value_by_position(grid, position):
    return grid[position[1]][position[0]]



def choose_random_direction(num_direction):
    """Generate a direction vector based on num_direction possiable direction"""
    step_angle = 2 * np.pi / num_direction
    dirt_index = np.random.choice(num_direction)
    angle = dirt_index * step_angle
    dirt_angle = np.array([np.cos(angle), np.sin(angle)])
    magnitude = np.linalg.norm(dirt_angle)
    norm_drection = dirt_angle / magnitude
    return norm_drection


def calculate_gradient(anchors_positions, anchors_values):
    """
    
    """
    anchors_positions = np.array(anchors_positions)

    # print(anchors_positions.shape)
    x,y = anchors_positions[:,0],anchors_positions[:,1]
    z = anchors_values

    rbf = Rbf(x, y, z, function='multiquadric')#multiquadric

    current_anchor_position = anchors_positions[0]

    h = 1e-5
    f_x = (rbf(current_anchor_position[0] + h, current_anchor_position[1]) - rbf(current_anchor_position[0] - h, current_anchor_position[1])) / (2 * h)
    f_y = (rbf(current_anchor_position[0], current_anchor_position[1] + h) - rbf(current_anchor_position[0], current_anchor_position[1] - h)) / (2 * h)
    gradient = np.array([f_x, f_y])
    normalized_gradient = gradient / np.linalg.norm(gradient)

    return normalized_gradient


class Anchor:
    def __init__(self, position, value, num_direction):
        self.position = np.array(position)
        self.value = value
        self.previous_best_value = value
        self.previous_best_position = np.array(position).copy()
        # self.global_best_value = float('-inf')  
        # self.global_best_position = np.array([0, 0]) 
        self.global_best_value = value
        self.global_best_position = np.array(position).copy()
        # self.step = np.array([0, 0]) 
        self.step = choose_random_direction(num_direction)
        self.historical_neighbors = [] #[[t1's self.position, neighbors_positions...],[t2's self...,nei...],[]]


    def PSO_determine_v_direction(self, inertia_w_max, inertia_w_min, cognitive_c, social_c, cur_cycle, cycle_max, S_c):
        S = S_c*(np.exp(1-(cur_cycle/cycle_max)))
        inertia_w = inertia_w_max - (inertia_w_max-inertia_w_min)/cycle_max*cur_cycle

        # rand_0 = np.random.uniform(0.5, 1.5)
        rand_0 = 1
        rand_1 = np.random.uniform(0, 1)
        rand_2 = np.random.uniform(0, 1)

        v_0 = inertia_w*rand_0*self.step[0] + cognitive_c*rand_1*(self.previous_best_position[0]-self.position[0]) + social_c*rand_2*(self.global_best_position[0]-self.position[0])
        v_1 = inertia_w*rand_0*self.step[1] + cognitive_c*rand_1*(self.previous_best_position[1]-self.position[1]) + social_c*rand_2*(self.global_best_position[1]-self.position[1])

        magnitude = math.sqrt(v_0**2 + v_1**2)
        if magnitude == 0:  # Avoid division by zero
            return np.array([0, 0]),np.array([0, 0])
        
        norm_velocity = np.array([v_0 / magnitude, v_1 / magnitude])
        next_move = S*norm_velocity
        next_move_int = np.rint(next_move).astype(int)
        return np.array([v_0, v_1]), next_move_int
    

    # def calculate_centroid(self):
    #     if not self.historical_neighbors:
    #         return self.position
        
    #     data = self.historical_neighbors[-1].copy()
    #     if len(data)==1:
    #         return self.position
        
    #     sum_positions = np.array([0, 0])
    #     for neighbor_position in data:
    #         sum_positions += np.array(neighbor_position)
        
    #     centroid_coords = sum_positions / len(data)
    #     return centroid_coords

    def calculate_centroid(self,n_position):
        if len(n_position)==1:
            return self.position
        
        sum_positions = np.array([0, 0])
        for neighbor_position in n_position:
            sum_positions += np.array(neighbor_position)
        
        centroid_coords = sum_positions / len(n_position)
        return centroid_coords
    

    def Gradient_determine_step_with_centroid(self, cur_cycle, cycle_max, Sc, his_position):
        S = Sc*(np.exp(1-(cur_cycle/cycle_max)))
        print(f"his_position{his_position}")
        centroid = self.calculate_centroid(his_position)
        # dir_step = self.position - centroid
        dir_step = self.global_best_position - centroid
        print(f"dir_step={dir_step}")
        
        # need to random move, to explore better position
        if np.linalg.norm(dir_step) == 0:
            normalized_dir_s = choose_random_direction(num_direction=8)    
        else:
            normalized_dir_s = dir_step / np.linalg.norm(dir_step)
        step = S * normalized_dir_s
        step_r = np.round(step).astype(int)

        return step_r


    def Gradient_determine_step_with_extrapolation(self, cur_cycle, cycle_max, Sc, his_position,his_value):
        S = Sc * (np.exp(1- (cur_cycle /cycle_max)))
        normalized_dir_s = calculate_gradient(his_position, his_value)
        step = S * normalized_dir_s
        step_r = np.round(step).astype(int)

        return step_r

       

    def update_position(self, params, all_anchors,grid, min_distance=2.0, max_retries=10, window_size=3):

        # Check if the current value is better than the previous best value
        if self.value > self.previous_best_value:
            self.previous_best_value = self.value
            self.previous_best_position = self.position.copy()

        # Determine the next step and check weather it would collision, if does, re_calculate the next step till max_retries times or not collision.
        curent_cycle = params[4]
        his_position, his_value = self.historical_data(window_size, grid)
        print(f"his_position===={his_position}")
        for _ in range(max_retries):
            # Problem of retries
            if abs(self.value - self.global_best_value) < 0.5 and curent_cycle > 5:
                if np.size(his_position) > 4:
                    self.step = self.Gradient_determine_step_with_extrapolation(params[4], params[5], params[6], his_position,his_value)
                    next_move = self.step.copy()
                else:
                    self.step = self.Gradient_determine_step_with_centroid(params[4], params[5], params[6], his_position)
                    next_move = self.step.copy()


            else:
                self.step,next_move = self.PSO_determine_v_direction(*params)
            new_position = self.position + next_move
            if not self.is_collision(new_position, all_anchors, min_distance):
                self.position = new_position
                self.value = grid[new_position[1]][new_position[0]] 
                break
            self.step = np.array([0, 0])


    def historical_data(self, window_size, grid):

        """
        Grab the last window_size time, the anchor's all neighbors' inclinding itself's current info in the first position.
        Parameters:
            window_size (int): The number of historical moments to grab.
            grid (object): The map grid information.
        Returns:
            tuple(his_anchors_positions(1D list), his_anchors_power((1D numpy array))) and the first element is the current_node info,
            his_anchors_positions = [[1,10],[5,7],[6,8],[10,5],[4,6]], anchor current position is [1,10], the rest is its neighbors and its past positions.
        """

        if len(self.historical_neighbors) < window_size:
            his_anchors_stacks = self.historical_neighbors[:].copy()
            # print(f"his_anchors_stacks{his_anchors_stacks}")
        else:
            his_anchors_stacks = self.historical_neighbors[-window_size:].copy()
            print(f"his_anchors_stacks{his_anchors_stacks}")
        if not his_anchors_stacks:
            print("No historical data available.")
            # return [], np.array([])
            return self.position, self.value

        current_anchor_p = his_anchors_stacks[-1].pop(0)
        # if after pop no element, then remove the last sublist
        if not his_anchors_stacks[-1]:
            his_anchors_stacks.pop()

        first_node = [current_anchor_p]
        his_anchors_stacks.insert(0,first_node)

        his_anchors_positions = [p for stack in his_anchors_stacks for p in stack]
        # Convert list of lists to list of tuples, then remove duplicates, then convert back to list of lists.
        his_anchors_positions = [list(item) for item in set(tuple(sublist) for sublist in his_anchors_positions)]
        his_anchors_power = np.array([get_anchor_value_by_position(grid, p) for p in his_anchors_positions])

        return his_anchors_positions, his_anchors_power

    def is_collision(self, new_position, all_anchors, min_distance):
        for a in all_anchors:
            diff = new_position - a.position
            if np.linalg.norm(diff) < min_distance:
                return True
        return False

    

    def find_neighbors(self, all_anchors, radius):

        current_neighbors = [anchor for anchor in all_anchors if 0 < np.linalg.norm(self.position - anchor.position) <= radius]
        # Check if current_neighbors is empty
        if len(current_neighbors):
            self.historical_neighbors.append([self.position] + [neighbor.position for neighbor in current_neighbors])
        else:
            self.historical_neighbors.append([self.position])




    def update_global_best(self, grid):
        """
        The global best is not real global, it's local best(neighbors' best)
        """
        # grab the latest neighbors including itself
        latest_neighbors_positions = self.historical_neighbors[-1]
        best_value = self.value
        best_position = self.position.copy()

        for neighbor_position in latest_neighbors_positions:
            neighbor_value = get_anchor_value_by_position(grid, neighbor_position)
            if neighbor_value > best_value:
                best_value = neighbor_value
                best_position = neighbor_position

        self.global_best_value = best_value
        self.global_best_position = best_position.copy()
    
    def print_value(self):
        print(self.value)

    @classmethod
    def update_all_global_bests(cls, anchors, grid, radius):
        for anchor in anchors:
            anchor.find_neighbors(anchors, radius)
            anchor.update_global_best(grid)



def main():

    parameter_of_features = [[150,100],-np.pi,-40,0,2,1,np.pi/16]
    grid = source_map_generator.generate_map(parameter_of_features, [200,200])

    inertia_w_max=0.9 # max=0.9 min = 0.4 by default
    inertia_w_min=0.4
    cognitive_c=2 #cognitive_c,social_c=2 by default
    social_c=2
    cur_cycle=1
    cycle_max=50
    # Tc=5
    Sc =5
    min_distance_neighbor = 3
    radius = 30
    max_retries = 10
    window_size = 3 # the historical data size 

    # anchors_coords = [[82,85], [59,87],[70,112],[60,87],[65,120],[63,90],[67,99],[76,90],[61,108],[66,117],[64,80],[70,98],[78,95],[75,84]]
    # anchors_coords = [[82,95], [79,87],[34,60],[40,50],[67,90],[30,50],[50,30],[120,60],[170,70],[160,50]]
    # anchors_coords = [[20,60],[90,70],[60,50],[70,40],[90,30]]
    anchors_coords = [[60,30]]
    # anchors_coords = [[160,120],[150,160],,[80,50],[70,90],[68,80],[50,140]]


    anchors = [Anchor(tuple(coord), value=grid[coord[1]][coord[0]], num_direction=8) for coord in anchors_coords]

    trajectories = [[] for _ in range(len(anchors))]

    for cur_cycle in range(cycle_max):
        for idx, anchor in enumerate(anchors):
            params = (inertia_w_max, inertia_w_min, cognitive_c, social_c, cur_cycle, cycle_max, Sc)
            anchor.update_position(params, anchors, grid, min_distance_neighbor,max_retries,window_size)

            # Record the position for this anchor at this time cycle
            trajectories[idx].append(anchor.position.copy())

        #need update the anchors
        Anchor.update_all_global_bests(anchors, grid, radius)
        cur_cycle += 1


    trajectories_value = [[grid[t[1]][t[0]] for t in node] for node in trajectories]
    trajectories = np.array(trajectories)
    trajectories_value = np.array(trajectories_value)

    # Plot the trajectory 
    fig, ax = plt.subplots()
    ax.set_xlim(0, 200)
    ax.set_ylim(0, 200)

    X_contour = np.linspace(0, grid.shape[1], grid.shape[1])
    Y_contour = np.linspace(0, grid.shape[0], grid.shape[0])
    levels = 25
    contour = ax.contour(X_contour, Y_contour, grid, levels)
    # im = ax.imshow(grid,  origin='lower', cmap='viridis', vmin=-120, vmax =-40)

    # fig.colorbar(im, ax=ax)

    point_size = 6
    scatter = ax.scatter([], [], c=[], s=point_size, cmap='viridis')

    # Update the scatter plot for each anchor update
    def update(frame):
        x_points = [point[0] for point in trajectories[:,frame]]
        y_points = [point[1] for point in trajectories[:,frame]]
        vals = trajectories_value[:,frame] + [3]*len(trajectories_value[:,frame])

        scatter.set_offsets(np.c_[x_points, y_points])
        scatter.set_array(np.array(vals))
        ax.set_title(f'Step {frame + 1}')

    # Animation plot
    ani = FuncAnimation(fig, update, frames=len(trajectories[0]), interval=200, blit=False)
    test_ani_file_name = 'C:/Users/xdwun/Research/Codes_work/AI_army/test_animation.gif'
    ani.save(test_ani_file_name, writer='pillow' )
    plt.show()



if __name__ == '__main__':
    main()
