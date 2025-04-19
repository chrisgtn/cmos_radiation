import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import logging
import os

""" 
    Dose Measurement Analysis: Total particle counts for each MU class visualisations, derived from the Image processing algorithm 
"""

logging.basicConfig(level=logging.DEBUG, filename='script_debug.log', filemode='w', format='%(asctime)s - %(levelname)s - %(message)s')

# Function to calculate total particle count for each class
def calculate_total_counts(csv_files):
    total_counts = []

    for csv_file in csv_files:

        class_mu = int(os.path.basename(csv_file).split('_')[2].replace('mu.csv', ''))
        data = pd.read_csv(csv_file)
        total_particle_count = data['Particle_Count'].sum()
        
        # Append to the list as a tuple 
        total_counts.append((class_mu, total_particle_count))
    
    # Sort by MU class
    total_counts.sort(key=lambda x: x[0])
    
    return total_counts

# Function to plot total counts against MU classes
def plot_total_counts(total_counts, output_path):

    mu_classes = [x[0] for x in total_counts]
    particle_counts = [x[1] for x in total_counts]
    
    plt.figure(figsize=(10, 6))
    plt.scatter(mu_classes, particle_counts, color='blue', label='Data Points')
    
    # Fit and plot a line between the data points
    plt.plot(mu_classes, particle_counts, color='red', linestyle='-', label='Trend Line')

    plt.xlabel('MU Class')
    plt.ylabel('Total Particle Count')
    plt.title('Total Particle Count vs MU Class')
    plt.xticks(rotation=45)
    
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    logging.info(f"Total counts plot has been saved at {output_path}.")
    
    
def plot_total_counts_zoomed(total_counts_df, output_path, mu_min, mu_max):
    # Filter the dataframe for the specified MU range
    filtered_df = total_counts_df[(total_counts_df['MU_Class'] >= mu_min) & (total_counts_df['MU_Class'] <= mu_max)]
    
    # Extract MU classes and their corresponding total counts
    mu_classes = filtered_df['MU_Class'].tolist()
    particle_counts = filtered_df['Total_Particle_Count'].tolist()

    # Create evenly spaced positions for x-ticks
    x_positions = np.arange(len(mu_classes))
    
    plt.figure(figsize=(8, 8))
    plt.scatter(x_positions, particle_counts, color='blue', label='Data Points')
    # Fit and plot a line between the data points
    plt.plot(x_positions, particle_counts, color='red', linestyle='-', label='Trend Line')

    plt.xlabel('MU Class')
    plt.ylabel('Total Particle Count')
    plt.title(f'Total Particle Count vs MU Class ({mu_min}-{mu_max})')
    
    # custom x-ticks and labels
    plt.xticks(x_positions, mu_classes, rotation=45)
    
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"Total counts plot has been saved at {output_path}.")


# total counts for each class
csv_files_directory = 'csv'  
csv_files = [os.path.join(csv_files_directory, f) for f in os.listdir(csv_files_directory) if f.startswith('counts_mu_') and f.endswith('.csv')]

total_counts = calculate_total_counts(csv_files)

# plot total counts against MU classes
output_path_total = 'total_counts_vs_mu_class.png'
plot_total_counts(total_counts, output_path_total)


# plot total counts for each class zoomed 
processed_csv = 'processed_results_with_frames.csv'
total_counts_df = pd.read_csv(processed_csv)

output_path_total = 'total_counts_vs_mu_low.png'
plot_total_counts_zoomed(total_counts_df, output_path_total, mu_min=1, mu_max=1000)
