import math
import matplotlib.pyplot as plt
import numpy as np
import json
import os
import source_map_generator
def determine_step(last_step, global_a_best_position, local_a_best_position, a_cur_position, inertia_w_max, inertia_w_min, cognitive_c, social_c, cur_cycle, cycle_max, Vc):

  V_max = Vc*(np.exp(1-(cur_cycle/cycle_max)))
  inertia_w = inertia_w_max - (inertia_w_max-inertia_w_min)/cycle_max*cur_cycle

  rand_1 = np.random.uniform(0, 1)
  rand_2 = np.random.uniform(0, 1)

  v_0 = inertia_w*last_step[0] + cognitive_c*rand_1*(local_a_best_position[0]-a_cur_position[0]) + social_c*rand_2*(global_a_best_position[0]-a_cur_position[0])
  v_1 = inertia_w*last_step[1] + cognitive_c*rand_1*(local_a_best_position[1]-a_cur_position[1]) + social_c*rand_2*(global_a_best_position[1]-a_cur_position[1])

  magnitude = math.sqrt(v_0**2 + v_1**2)
  if magnitude > V_max:
      v_0 = (v_0 / magnitude) * V_max
      v_1 = (v_1 / magnitude) * V_max

  rounded_v0 = int(v_0)
  rounded_v1 = int(v_1)

  return [rounded_v0,rounded_v1]


def determine_position(move_step, cur_a_position):
  a_next_0 = move_step[0] + cur_a_position[0]
  a_next_1 = move_step[1] + cur_a_position[1]

  return [a_next_0,a_next_1]

def determine_best_position(as_position,grid):
  as_value = [grid[x[1]][x[0]] for x in as_position]
  max_index = as_value.index(max(as_value))
  return as_position[max_index]


def calculate_distance(s,a):
    x1,x2= s[0],a[0]
    y1,y2 = s[1],a[1]
    distance = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
    return distance

def find_neighbors(target_a,all_anchors,neighbor_dis):
  neighbors = []
  for x in all_anchors:
    if calculate_distance(target_a, x) < neighbor_dis:
      neighbors.append(x)
  return neighbors


def main():
  
  parameter_of_features = [[150,100],-np.pi,-40,0,2,1,np.pi/16]
  grid = source_map_generator.generate_map(parameter_of_features, [200,200])
  anchors_location = [[82,85], [59,87],[70,112],[60,87],[65,120],[63,90],[67,99],[76,90],[61,108],[66,117],[64,80],[70,98],[75,84]]
  cycle_max = 50

  update_anchors = [[anchors_location[i]] for i in range(len(anchors_location))]
  last_step_as = [[[0,0]] for i in range(len(anchors_location))]

  anchors_value = np.array([grid[x[1]][x[0]] for x in anchors_location])
  initial_max_index = anchors_value.argmax()

  max_global_index = anchors_location[initial_max_index]
  max_local_index = anchors_location[initial_max_index]

  g_best_value = anchors_value[initial_max_index]
  l_best_value = anchors_value[initial_max_index]

  anchors_process = []

  for t in range(cycle_max):   # Changed this line to have a variable t for the cycle
    a_i_neighbor_with_ai = anchors_location[:]
    neigh_anchors_value = np.array([grid[x[1]][x[0]] for x in a_i_neighbor_with_ai])
    max_update_index = neigh_anchors_value.argmax()

    if g_best_value < neigh_anchors_value[max_update_index]:
        g_best_value = neigh_anchors_value[max_update_index]
        max_global_index = a_i_neighbor_with_ai[max_update_index]
    for i in range(len(anchors_location)):
        previous_update_local = update_anchors[i][-1]
        previous_update_value = grid[previous_update_local[1]][previous_update_local[0]]
        if previous_update_value > l_best_value:
            max_local_index = previous_update_local

        a_move_step = determine_step(last_step=last_step_as[i][-1], global_a_best_position=max_global_index, local_a_best_position=max_local_index, a_cur_position=anchors_location[i], inertia_w_max=0.9, inertia_w_min=0.4, cognitive_c=2, social_c=2, cur_cycle=t, cycle_max=cycle_max, Vc=3)
        next_position = determine_position(a_move_step, anchors_location[i])

        # anti_collision
        stop_flag = 0
        min_distance = 2
        regular_neighbors = a_i_neighbor_with_ai[:i]+a_i_neighbor_with_ai[(i+1):]
        while find_neighbors(next_position,regular_neighbors, min_distance):
          # if there has neighbor close to min_dis, repeat the operation that find the next step
          a_move_step = determine_step(last_step=last_step_as[i][-1], global_a_best_position=max_global_index, local_a_best_position=max_local_index, a_cur_position=anchors_location[i], inertia_w_max=0.9, inertia_w_min=0.4, cognitive_c=2, social_c=2, cur_cycle=t, cycle_max=cycle_max, Vc=3)
          next_position = determine_position(a_move_step, anchors_location[i])
          stop_flag += 1

          if stop_flag > 5:
            next_position = anchors_location[i]
            # print(f"{i}th anchor doesn't move")
            break

        if 0 <= next_position[0] < len(grid[0]) and 0 <= next_position[1] < len(grid):
            anchors_location[i] = next_position
        else:
            print(f"Warning: Anchor {i} went out of bounds with position {next_position}. Reverting to previous position.")
            next_position = anchors_location[i]  # Retain last valid position
            a_move_step = [0, 0]

        last_step_as[i].append(a_move_step)
        update_anchors[i].append(next_position)

    anchors_process.append(anchors_location[:])
  whole_updated_process_points_vals = [[grid[node[1]][node[0]] for node in group] for group in anchors_process]

# store datas in the files
  dir_path = "C:\\Users\\xdwun\\Research\\Codes_work\\AI_army"
  a_position_file_path = os.path.join(dir_path,'anchors_positions.json')
  with open(a_position_file_path,'w') as f:
    json.dump(anchors_process,f)
  a_values_file_path = os.path.join(dir_path,'anchors_values.json')
  with open(a_values_file_path,'w') as f:
    json.dump(whole_updated_process_points_vals,f)
  print('done')

if __name__ == '__main__':
   main()

