import pandas as pd

# Replace 'your_file.csv' with the path to your CSV file
file_path = './csv_files/CampusParkingLot.csv'
# Read the CSV file
data = pd.read_csv(file_path)

# Assuming the column for power values is named 'power'
max_power = data['Power'].max()

print("The maximum power value is:", max_power)
