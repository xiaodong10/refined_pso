import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load the CSV data into a DataFrame
df = pd.read_csv('./CVS_files/grid_data.csv')

# # Filter the data to include only GridY values from 15 to 265
# filtered_df = df.query('15 <= GridY <= 265')

# # Compute dimensions
# grid_size_x = filtered_df['GridX'].max() - filtered_df['GridX'].min() + 1
# grid_size_y = filtered_df['GridY'].max() - filtered_df['GridY'].min() + 1

# # Create an array for heatmap data
# heatmap_data = np.zeros((grid_size_x, grid_size_y))

# # Fill heatmap data
# for _, row in filtered_df.iterrows():
#     heatmap_data[row['GridX'] - filtered_df['GridX'].min(), row['GridY'] - filtered_df['GridY'].min()] = row['Power']


# 1. Create a full grid with default values
grid_size_x = df['GridX'].max() - df['GridX'].min() + 1
grid_size_y = df['GridY'].max() - df['GridY'].min() + 1

full_grid = pd.DataFrame({
    'GridX': np.repeat(np.arange(df['GridX'].min(), df['GridX'].max() + 1), grid_size_y),
    'GridY': np.tile(np.arange(df['GridY'].min(), df['GridY'].max() + 1), grid_size_x),
    'Power': -200  # default value
})

# 2. Overlay with the existing data
merged_df = full_grid.merge(df, on=['GridX', 'GridY'], how='left')
merged_df['Power'] = merged_df['Power_y'].combine_first(merged_df['Power_x'])

# 3. Use the merged_df for your heatmap

heatmap_data = np.zeros((grid_size_x, grid_size_y))
for _, row in merged_df.iterrows():
    heatmap_data[row['GridX'] - df['GridX'].min(), row['GridY'] - df['GridY'].min()] = row['Power']


# Plotting
plt.imshow(heatmap_data, origin='lower', cmap='jet', aspect='auto')
plt.colorbar(label='Power')
plt.title('Power Heatmap')
plt.xlabel('GridY')
plt.ylabel('GridX')
plt.show()
