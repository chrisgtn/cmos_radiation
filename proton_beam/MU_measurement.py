import os
import cv2
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.ticker as ticker
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression

""" 
    Dose Measurement Analysis: Activated Pixels and Total Intensity against MU Class (including logarithmic scale)
"""

def process_images_in_directory(directory, threshold):
    total_activated_pixels = 0
    total_intensity = 0
    frame_count = 0 
    for filename in os.listdir(directory):
        if filename.endswith(".png"):
            image_path = os.path.join(directory, filename)
            image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            total_activated_pixels += np.sum(image > threshold)
            total_intensity += np.sum(image)
            frame_count += 1
        
            
            
    # Extract MU value from directory name
    try:
        mu_value = int(''.join(filter(str.isdigit, directory.split('/')[-1])))
    except ValueError:
        print(f"Error: MU value not found in directory name {directory}")
        mu_value = 0  
    return pd.DataFrame({
        'MU': [mu_value], 
        'Total Activated Pixels': [total_activated_pixels], 
        'Total Intensity': [total_intensity],
        'Frame Count': [frame_count]
    })

# threshold for activation
threshold = 0

# Directory containing the images
root_directory = 'runs_mu'
mu_directories = [d for d in os.listdir(root_directory) if os.path.isdir(os.path.join(root_directory, d))]

# Data frame to hold all data
all_data = pd.DataFrame()

# Process images and calculate averages
for mu_dir in mu_directories:
    full_path = os.path.join(root_directory, mu_dir)
    mu_data = process_images_in_directory(full_path, threshold)
    all_data = pd.concat([all_data, mu_data], ignore_index=True)
    
    
all_data['MU'] = pd.to_numeric(all_data['MU'])
all_data.sort_values('MU', inplace=True)

# Group data by MU, calculate mean and standard deviation
grouped_data = all_data.groupby('MU').agg({
    'Total Activated Pixels': ['mean', 'std'],
    'Total Intensity': ['mean', 'std']
}).reset_index()

# Flatten the MultiIndex for columns resulting from aggregation
grouped_data.columns = [' '.join(col).strip() for col in grouped_data.columns.values]

# Rename columns for convenience
grouped_data.rename(columns={
    'MU ': 'MU',
    'Total Activated Pixels mean': 'Average Activated Pixels',
    'Total Activated Pixels std': 'Activated Pixels Std',
    'Total Intensity mean': 'Average Intensity',
    'Total Intensity std': 'Intensity Std'
}, inplace=True)

# Make sure 'MU' is a string to plot as discrete categories
grouped_data['MU'] = grouped_data['MU'].astype(str)

# Plot the data
plt.figure(figsize=(14, 7))

# the numerical index for the x-position and label with the MU classes
plt.errorbar(
    x=np.arange(len(grouped_data)),  # x-position based on the numerical index
    y=grouped_data['Average Activated Pixels'],
    yerr=grouped_data['Activated Pixels Std'],
    fmt='o', color='blue', ecolor='lightblue'
)

# Label the x-axis with MU classes
plt.xticks(
    ticks=np.arange(len(grouped_data)),  # ticks to the numerical index
    labels=grouped_data['MU'],  # Label with the MU classes
    rotation=90
)

plt.title('Average Activated Pixels by MU Class')
plt.xlabel('MU (class)')
plt.ylabel('Average Activated Pixels')
plt.yscale('log')  # Log scale for y-axis
plt.tight_layout()
plt.show()


plt.figure(figsize=(14, 7))
plt.errorbar(
    x=np.arange(len(grouped_data)),
    y=grouped_data['Average Intensity'],
    yerr=grouped_data['Intensity Std'],
    fmt='o', color='red', ecolor='pink'
)

plt.xticks(
    ticks=np.arange(len(grouped_data)),
    labels=grouped_data['MU'],
    rotation=90
)

plt.title('Average Intensity by MU Class')
plt.xlabel('MU (class)')
plt.ylabel('Average Intensity')
plt.yscale('log')
plt.tight_layout()
plt.show()