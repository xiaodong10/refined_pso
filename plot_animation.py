import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation
import os 

def plot_anchors_animation(grid, trajectories, trajectories_value, plot_name):
    fig, ax = plt.subplots()
    ax.set_xlim(0, grid.shape[1])
    ax.set_ylim(0, grid.shape[0])

    X_contour, Y_contour = np.meshgrid(np.arange(grid.shape[1]), np.arange(grid.shape[0]))
    levels = np.linspace(np.min(grid), np.max(grid), 25)
    contour = ax.contour(X_contour, Y_contour, grid, levels)
    im = ax.imshow(grid,  origin='lower', cmap='viridis', vmin=-200, vmax =-20)

    point_size = 10
    # scatter = ax.scatter([], [], c=[], s=point_size, cmap='magma', vmin=np.min(grid), vmax=np.max(grid))
    scatter = ax.scatter([], [], c=[], s=point_size, cmap='Oranges', vmin=-250, vmax=0)

    def update(frame):
        try:
            x_points = [point[0] for point in trajectories[:, frame]]
            y_points = [point[1] for point in trajectories[:, frame]]
            vals = trajectories_value[:, frame]

            scatter.set_offsets(np.c_[x_points, y_points])
            scatter.set_array(vals)
            ax.set_title(f'Step {frame + 1}')
            return scatter,
        except Exception as e:
            print(f"Error updating frame {frame}: {e}")

    ani = FuncAnimation(fig, update, frames=len(trajectories[0]), interval=500, blit=False)


    ani_name = os.path.join("Plots", plot_name)
    ani.save(ani_name, writer='pillow')
    plt.show()

# # Make sure to call the function with the correct arguments
# plot_anchors_animation(grid, trajectories, trajectories_value, plot)
