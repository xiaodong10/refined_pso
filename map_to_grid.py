import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def latitude_longitude_to_grid(lat, lon, base_lat, base_lon, grid_size=0.000001):
    grid_x = int((lat - base_lat) / grid_size)
    grid_y = int((lon - base_lon) / grid_size)
    return grid_x, grid_y


def transfer_data_grid(map_csv_path):
    df = pd.read_csv(map_csv_path)
    # Replace -Inf with -200 
    df.replace([-np.inf], -200, inplace=True)
    df.replace("#NAME?", -200, inplace=True)

    base_lat = df['Latitude'].min()
    base_lon = df['Longitude'].min()

    # Get grid coordinates
    df['GridX'], df['GridY'] = zip(*df.apply(lambda row: latitude_longitude_to_grid(row['Latitude'], row['Longitude'], base_lat, base_lon), axis=1))

    grid_group = df[['GridX', 'GridY', 'Power']]
    # max_power = grid_group['Power'].max()
    # print(f"The signal largest power value is: {max_power}")

    reshape_df = grid_group.pivot('GridY', 'GridX', 'Power')

    array_2d = reshape_df.to_numpy()
    return array_2d

def plot_map(grid):
        
    plt.figure(figsize=(10, 10))
    plt.imshow(grid, cmap='viridis', origin='lower', aspect='auto')  # `origin='lower'` makes the [0,0] index appear at the bottom left corner.
    plt.colorbar(label='Power')
    plt.xlabel('GridX')
    plt.ylabel('GridY')
    plt.title('rf signal map')
    plt.show()

def main():
    map_csv_path = './CVS_files/pd_data.csv'
    map_grid = transfer_data_grid(map_csv_path)
    plot_map(map_grid)

if __name__ == "__main__":
    main()
    
