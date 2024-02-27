import matplotlib.pyplot as plt
import pandas as pd

# Example parameters
parameters = {
    # 'Parameter': ['tx frequency', 'tx height', 'tx power'],
    # 'Value': [28e9, 2, 1]
    'Parameter': ['M','N', 'inertia_max', 'inertia_min','c1','c2','s_c'],
    'Value': ['[5,10,15,20,25]',"70",'0.9','0.4','2.05','2.05','7']
    # Add more parameters and their values here
}

# Create a DataFrame
df = pd.DataFrame(parameters)

# Plotting the table
fig, ax = plt.subplots(figsize=(6, 2))  # Adjust the figure size as needed
ax.axis('off')  # Hide the axes

# Creating the table with custom styling
table = ax.table(cellText=df.values, colLabels=df.columns, loc='center', 
                 cellLoc='center', colColours=["#56B4E9"]*len(df.columns))

# Modifying the style of the table
table.auto_set_font_size(False)
table.set_fontsize(12)  # Adjust font size
table.auto_set_column_width(col=list(range(len(df.columns))))  # Adjust column width

# Show the table
plt.show()
