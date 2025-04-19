import os
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


""" 
    Energy dependency analysis: Pixel intensity and Activated pixel plots against Energy.
"""

def process_images_in_directory(directory, threshold):
    total_activated_pixels = 0
    total_intensity = 0
    total_coverage = 0
    frame_count = 0
    
    for filename in os.listdir(directory):
        if filename.endswith(".png"):
            image_path = os.path.join(directory, filename)
            image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            if image is not None:
                activated_pixels = np.sum(image > threshold)
                total_activated_pixels += activated_pixels
                total_intensity += np.sum(image)
                total_coverage += activated_pixels / image.size
                frame_count += 1
                
    average_activated_pixels = total_activated_pixels / frame_count if frame_count else 0
    average_intensity = total_intensity / frame_count if frame_count else 0
    average_coverage_ratio = total_coverage / frame_count if frame_count else 0

    # convert 5_5 
    energy_value = directory.replace('_', '.').split('/')[-1]
    energy_value = float(energy_value)
    
    return {
        'Energy': energy_value,
        'Average Activated Pixels': average_activated_pixels,
        'Average Total Intensity': average_intensity,
        'Average Coverage Ratio': average_coverage_ratio
    }

threshold = 20

root_directory = 'runs'
energy_directories = [d for d in os.listdir(root_directory) if os.path.isdir(os.path.join(root_directory, d))]
energy_directories.sort(key=lambda x: float(x.replace('_', '.')))

all_data_list = []

for energy_dir in energy_directories:
    dir_path = os.path.join(root_directory, energy_dir)
    energy_data = process_images_in_directory(dir_path, threshold)
    all_data_list.append(energy_data)


all_data = pd.DataFrame(all_data_list)
all_data['Energy'] = all_data['Energy'].astype(str)

# plot
fig, axs = plt.subplots(1, 3, figsize=(21, 7))  # 1 row, 3 columns


axs[0].scatter(all_data['Energy'], all_data['Average Activated Pixels'], color='blue')
axs[0].set_title('Avg Activated Pixels by Energy')
axs[0].set_xlabel('Energy (MeV)')
axs[0].set_ylabel('Average Activated Pixels')
# axs[0].set_xscale('linear')
axs[0].set_yscale('linear')
axs[0].set_xticks(all_data['Energy'])
axs[0].tick_params(axis='x', rotation=90)


axs[1].scatter(all_data['Energy'], all_data['Average Total Intensity'], color='red')
axs[1].set_title('Avg Total Intensity by Energy')
axs[1].set_xlabel('Energy (MeV)')
axs[1].set_ylabel('Average Total Intensity')
# axs[1].set_xscale('linear')
axs[1].set_yscale('linear')
axs[1].set_xticks(all_data['Energy'])
axs[1].tick_params(axis='x', rotation=90)


axs[2].scatter(all_data['Energy'], all_data['Average Coverage Ratio'], color='green')
axs[2].set_title('Avg Coverage Ratio by Energy')
axs[2].set_xlabel('Energy (MeV)')
axs[2].set_ylabel('Average Coverage Ratio')
# axs[2].set_xscale('linear')
axs[2].set_yscale('linear')
axs[2].set_xticks(all_data['Energy'])
axs[2].tick_params(axis='x', rotation=90)

plt.tight_layout()
plt.show()