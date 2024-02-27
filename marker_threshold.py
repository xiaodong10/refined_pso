import pandas as pd
import matplotlib.pyplot as plt

csv_file = './csv_files/GaussianCampusStreet.csv'
# csv_file = './csv_files/poster_map1.csv'

data = pd.read_csv(csv_file)

max_power = data['Power'].max()
print(f"max_power{max_power}")
high_power = data[data['Power'] > -32]
low_power = data[data['Power'] <= -32]


plt.figure(figsize=(10, 10))

plt.scatter(low_power['lon'], low_power['lat'], color='blue', label='Power ≤ 35')
plt.scatter(high_power['lon'], high_power['lat'], color='red', label='Power > 35')

plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.title('Distribution of Power Values')
plt.legend()
plt.show()
