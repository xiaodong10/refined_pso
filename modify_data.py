import pandas as pd
import numpy as np

# Load the CSV data into a DataFrame
df = pd.read_csv('./CVS_files/pd_data.csv')

# Replace -Inf with -200 (or any other value you'd like)
df.replace([-np.inf], -1000, inplace=True)

# Save the cleaned data back to a CSV file
df.to_csv('cleaned_file.csv', index=False)
