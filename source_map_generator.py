import math
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D

def calculate_distance(s,a):
    x1,x2= s[0],a[0]
    y1,y2 = s[1],a[1]
    distance = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
    return distance
def calculate_angle(s, theta, a):
    """
    Calculate the angle from the source with respect to theta to the anchor.

    Parameters:
        s (tuple): The coordinates (x, y) of the source.
        theta (float): The orientation of the source in radians.
        a (tuple): The coordinates (x, y) of the anchor.

    Returns:
        float: The angle in radians from the source to the anchor with respect to theta.
    """
    ang = np.angle((a[0]-s[0])+1j*(a[1]-s[1]))
    angle = ang - theta
    if angle > np.pi:
      angle -= 2*np.pi
    if angle < -np.pi:
      angle += 2*np.pi
    return angle

def point_map(s, theta, a, P0, mu, alpha, d0, beta):
  # the sourse point return 0, power is the largest
  dist = calculate_distance(s,a)
  angle = calculate_angle(s, theta, a)
  if dist <= 0 or d0 <= 0:
    return 0
  Pi = P0 - mu - 10 * alpha * np.log10(dist / d0) - 5 * (angle)**2 / (beta ** 2 * np.log(10))

  return Pi

def generate_map(parameter_of_features, map_size):
  s, theta, P0, mu, alpha, d0, beta = parameter_of_features
  rows = map_size[1]
  cols = map_size[0]
  grid = np.zeros((cols, rows))

  for row in range(rows):
    for col in range(cols):
      a = [row,col]
      Pi = point_map(s, theta, a, P0, mu, alpha, d0, beta)
      grid[col][row] = Pi
  # the signal is too weak to be received, threshold is -120
  grid = np.where(grid < -120, -120, grid)
  return grid

def plot_map(grid):

  X_contour = np.linspace(0, grid.shape[1], grid.shape[1])
  Y_contour = np.linspace(0, grid.shape[0], grid.shape[0])

  # Create a figure
  fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

  # Plot imshow
  im = ax1.imshow(grid,  origin='lower', cmap='viridis', vmin=-120, vmax =-40)
  ax1.set_title('Source MAP')
  ax1.set_xlabel('X')
  ax1.set_ylabel('Y')
  fig.colorbar(im, ax=ax1)

  # Plot contour
  levels = 25
  contour = ax2.contour(X_contour, Y_contour, grid, levels)
  ax2.set_title('Source Contour MAP')
  ax2.set_xlabel('X')
  ax2.set_ylabel('Y')
  fig.colorbar(contour, ax=ax2)

  # Display the plots
  plt.show()


def plot_3d(grid):
  x = np.linspace(0, grid.shape[1], grid.shape[1])
  y = np.linspace(0, grid.shape[0], grid.shape[0])

  X, Y = np.meshgrid(x, y)
  Z = grid
  fig = plt.figure()
  ax = fig.add_subplot(111, projection='3d')

  # Plot the 3D surface
  surf = ax.plot_surface(X, Y, Z, cmap='viridis',  rstride=1, cstride=1)

  fig.colorbar(surf)
  ax.set_xlabel('X')
  ax.set_ylabel('Y')
  ax.set_zlabel('Z')
  ax.set_title('Source 3D Surface Plot')
  plt.show()


# parameter_of_features = [[150,100],-np.pi,-40,0,2,1,np.pi/16]
# grid = generate_map(parameter_of_features, [200,200])
# plot_map(grid)
# plot_3d(grid)