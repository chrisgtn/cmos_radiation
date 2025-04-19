import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

""" 
    Box and Swarm Plot of track lengths for all radiation sources.
"""

directories = {
    'Uranium-238': 'data/1-uranium/',
    'Bismuth-207': 'data/3-bismuth/',
    'Cobalt-60': 'data/4-cobalt/',
    'Sodium-22': 'data/2-sodium/'
}

track_features = {
    'Uranium-238': {'lengths': [], 'widths': []},
    'Bismuth-207': {'lengths': [], 'widths': []},
    'Cobalt-60': {'lengths': [], 'widths': []},
    'Sodium-22': {'lengths': [], 'widths': []}
}

def extract_track_features(image):
    features = {
        'lengths': [],
        'widths': []
    }
    _, thresh = cv2.threshold(image, 30, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    for contour in contours:
        length = cv2.arcLength(contour, closed=False)
        features['lengths'].append(length)
        _, _, w, h = cv2.boundingRect(contour)
        width = min(w, h) 
        features['widths'].append(width)
        
    return features


for radiation_type, directory in directories.items():
    for filename in os.listdir(directory):
        if filename.endswith(".png"):
            
            image_path = os.path.join(directory, filename)
            image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

            features = extract_track_features(image)
            track_features[radiation_type]['lengths'].extend(features['lengths'])
            track_features[radiation_type]['widths'].extend(features['widths'])

# tracks
lengths_data = []
widths_data = []
for radiation_type, features in track_features.items():
    lengths_data.extend([(length, radiation_type) for length in features['lengths']])
    widths_data.extend([(width, radiation_type) for width in features['widths']])

lengths_df = pd.DataFrame(lengths_data, columns=['Length', 'Radiation Type'])
widths_df = pd.DataFrame(widths_data, columns=['Width', 'Radiation Type'])


def plot_track_feature_distribution(df, feature_name):
    if not df.empty:
        plt.figure(figsize=(14, 8))

        
        plt.subplot(1, 2, 1)
        sns.boxplot(x='Radiation Type', y=feature_name, data=df)
        plt.title(f'Box Plot of {feature_name}')

        
        plt.subplot(1, 2, 2)
        sns.violinplot(x='Radiation Type', y=feature_name, data=df)
        plt.title(f'Violin Plot of {feature_name}')

        plt.show()
    else:
        print(f"No data to plot for {feature_name}.")

# box and violin plots 
#plot_track_feature_distribution(lengths_df, 'Length')
#plot_track_feature_distribution(widths_df, 'Width')

plt.figure(figsize=(16, 10))

# box and swarm 
sns.boxplot(x='Radiation Type', y='Length', data=lengths_df, showfliers=False, boxprops={'facecolor':'None'})
sns.swarmplot(x='Radiation Type', y='Length', data=lengths_df, size=4)

plt.title('Box and Swarm Plot of Track Lengths')
plt.xlabel('Radiation Type')
plt.ylabel('Track Length')
plt.show()