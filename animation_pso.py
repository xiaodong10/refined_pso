
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation
import source_map_generator
import os
import json


# generate the source
parameter_of_features = [[150,100],-np.pi,-40,0,2,1,np.pi/16]
grid = source_map_generator.generate_map(parameter_of_features, [200,200])



# read datas in the files

# dir_path = os.path.dirname(os.path.abspath(__file__))
# ani_file_name = os.path.join(dir_path, './Plots/PSO_anchors_animation.gif')
# a_position_file_path = os.path.join(dir_path,'anchors_positions.json')
# a_values_file_path = os.path.join(dir_path,'anchors_values.json')

# Print the current working directory

print(os.getcwd())

ani_file_name = './Plots/PSO_anchors_animation.gif'
a_position_file_path = './Json_files/anchors_positions.json'
a_values_file_path = './Json_files/anchors_values.json'
with open(a_position_file_path, 'r') as f:
    anchors_process = json.load(f)
with open(a_values_file_path, 'r') as f:
    whole_updated_process_points = json.load(f)


# Plot the trajectory 
fig, ax = plt.subplots()
ax.set_xlim(0, 200)
ax.set_ylim(0, 200)

X_contour = np.linspace(0, grid.shape[1], grid.shape[1])
Y_contour = np.linspace(0, grid.shape[0], grid.shape[0])
levels = 25
contour = ax.contour(X_contour, Y_contour, grid, levels)
im = ax.imshow(grid,  origin='lower', cmap='viridis', vmin=-120, vmax =-40)

fig.colorbar(im, ax=ax)
# Create an empty scatter plot (will be updated in the animation)
point_size = 6
scatter = ax.scatter([], [], c=[], s=point_size, cmap='viridis')


# Update the scatter plot for each anchor update
def update(frame):
    x_points = [point[0] for point in anchors_process[frame]]
    y_points = [point[1] for point in anchors_process[frame]]
    vals = whole_updated_process_points[frame] + [3]*len(whole_updated_process_points[frame])

    scatter.set_offsets(np.c_[x_points, y_points])
    scatter.set_array(np.array(vals))
    ax.set_title(f'Step {frame + 1}')

# Animation plot
print(len(anchors_process))

ani = FuncAnimation(fig, update, frames=len(anchors_process), interval=200, blit=False)
# cbar = fig.colorbar(scatter, ax=ax)
ani.save(ani_file_name, writer='pillow' )
plt.show()


