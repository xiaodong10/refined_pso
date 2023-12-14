import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Read the pre-processed CSV file
df = pd.read_csv('pre_heatmap_data.csv')

# Define the conversion function
def latitude_longitude_to_grid(lat, lon, base_lat, base_lon, grid_size=0.000001):
    grid_x = int(np.floor((lat - base_lat) / grid_size))
    grid_y = int(np.floor((lon - base_lon) / grid_size))
    return grid_x, grid_y

# Get base coordinates
base_lat = df['Latitude'].min()
base_lon = df['Longitude'].min()

# Apply the function to the dataframe
df['GridX'], df['GridY'] = zip(*df.apply(lambda row: latitude_longitude_to_grid(row['Latitude'], row['Longitude'], base_lat, base_lon), axis=1))

# Create a new dataframe with GridX and GridY, and Power
grid_group = df[['GridX', 'GridY', 'Power']]

# Pivot the grid_group DataFrame
pivot_df = grid_group.pivot(index='GridY', columns='GridX', values='Power')

# Convert all non-NaN values to floats
pivot_df = pivot_df.apply(pd.to_numeric, errors='ignore')

# Plotting
plt.figure(figsize=(10, 10))
# Use np.ma.masked_invalid to mask NaNs
heatmap_array = np.ma.masked_invalid(pivot_df.to_numpy())
# plt.pcolormesh(heatmap_array, cmap='viridis', shading='nearest')
plt.pcolormesh(heatmap_array, cmap='viridis', shading='nearest', aspect='equal')

plt.colorbar(label='Power')
plt.xlabel('GridX')
plt.ylabel('GridY')
plt.title('Heatmap of Power Values')
plt.gca().invert_yaxis()  # Invert y-axis to match the geographic convention
plt.show()
