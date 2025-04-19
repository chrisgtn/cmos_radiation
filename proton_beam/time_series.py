import matplotlib.pyplot as plt
import pandas as pd
import logging
from datetime import datetime
import os

""" 
    Energy dependency analysis: Particle counts over time for each Energy class visualisations.
"""

logging.basicConfig(level=logging.DEBUG, filename='script_debug.log', filemode='w', format='%(asctime)s - %(levelname)s - %(message)s')


def extract_timestamp(frame_name):
    try:
        parts = frame_name.split('_')
        logging.debug(f"Frame name parts: {parts}")
        if len(parts) != 5:
            raise ValueError(f"Unexpected frame name format: {frame_name}")
        date_str = parts[3]  
        time_str = parts[4].replace('.png', '') 
        datetime_str = f"{date_str}_{time_str}"
        return datetime.strptime(datetime_str, "%m.%d_%H.%M.%S")
    except (IndexError, ValueError) as e:
        logging.error(f"Error parsing frame name {frame_name}: {e}")
        return None


def plot_counts_over_time_for_class(csv_file, output_dir):
    
    data = pd.read_csv(csv_file)
    logging.debug(f"Frame column values:\n{data['Frame'].head()}")
    data['Timestamp'] = data['Frame'].apply(extract_timestamp)
    data = data.dropna(subset=['Timestamp'])
    data.sort_values(by='Timestamp', inplace=True)
    
    logging.debug(data[['Frame', 'Timestamp']].to_string())
    cls = os.path.basename(csv_file).split('_')[2].replace('.csv', '')

    plt.figure(figsize=(10, 6))
    plt.scatter(data['Timestamp'], data['Particle_Count'], color='blue', label='Data Points')
    plt.plot(data['Timestamp'], data['Particle_Count'], color='red', linestyle='-', label='Trend Line')

    plt.xlabel('Time')
    plt.ylabel('Particle Count')
    plt.title(f'Particle Count Over Time for Class {cls}')
    plt.xticks(rotation=45)

    # show time values only
    plt.gca().xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%H:%M:%S'))
    plt.gca().xaxis.set_major_locator(plt.matplotlib.dates.AutoDateLocator())

    plt.legend()
    plt.tight_layout()
    output_path = os.path.join(output_dir, f'counts_over_time_class_{cls}.png')
    plt.savefig(output_path)
    plt.close()
    logging.info(f"Counts over time plot for class {cls} has been saved at {output_path}.")

csv_file = 'csv/counts_energy_5_5.csv'  
output_dir = 'time-series-energies' 
plot_counts_over_time_for_class(csv_file, output_dir)
