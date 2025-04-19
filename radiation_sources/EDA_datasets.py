import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns


"""
    Exploratory Data Analysis
    This script is used to analyze the datasets of different radiation types.
"""


directories = {
    'Uranium-238': 'data/1-uranium/',
    'Bismuth-207': 'data/3-bismuth/',
    'Cobalt-60': 'data/4-cobalt/',
    'Sodium-22': 'data/2-sodium/'
}


features = {
    'Uranium-238': {'activated_pixels': [], 'average_intensity': [], 'max_intensity': [], 'total_intensity': []},
    'Bismuth-207': {'activated_pixels': [], 'average_intensity': [], 'max_intensity': [], 'total_intensity': []},
    'Cobalt-60': {'activated_pixels': [], 'average_intensity': [], 'max_intensity': [], 'total_intensity': []},
    'Sodium-22': {'activated_pixels': [], 'average_intensity': [], 'max_intensity': [], 'total_intensity': []}
}

def extract_features(image):
    # threshold to highlight activated pixels
    _, thresh = cv2.threshold(image, 30, 255, cv2.THRESH_BINARY)

    # contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Initialize feature variables
    activated_pixels = np.sum(thresh > 0)
    max_intensity = np.max(image)
    total_intensity = np.sum(image)
    average_intensity = np.mean(image[thresh > 0]) if activated_pixels > 0 else 0

    return activated_pixels, average_intensity, max_intensity, total_intensity

# each dataset
for radiation_type, directory in directories.items():
    for filename in os.listdir(directory):
        if filename.endswith(".png"):
            
            image_path = os.path.join(directory, filename)
            
            image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

            # Extract features
            a_p, a_i, m_i, t_i = extract_features(image)

            # Append features lists
            features[radiation_type]['activated_pixels'].append(a_p)
            features[radiation_type]['average_intensity'].append(a_i)
            features[radiation_type]['max_intensity'].append(m_i)
            features[radiation_type]['total_intensity'].append(t_i)


# def plot_histograms(features, feature_name):
#     plt.figure(figsize=(10, 8))

#     for radiation_type, feature_dict in features.items():
#         plt.hist(feature_dict[feature_name], bins=50, alpha=0.5, label=f"{radiation_type} {feature_name}")

#     plt.title(f'Histogram of {feature_name}')
#     plt.xlabel(feature_name)
#     plt.ylabel('Frequency')
#     plt.legend()
#     plt.show()

#    histograms for each feature
# for feature_name in ['activated_pixels', 'average_intensity', 'max_intensity', 'total_intensity']:
#     plot_histograms(features, feature_name)




def plot_density(features, feature_name):
    plt.figure(figsize=(12, 8))

    for radiation_type, feature_dict in features.items():
        sns.kdeplot(feature_dict[feature_name], label=f"{radiation_type} {feature_name}")

    plt.title(f'Density Plot of {feature_name}')
    plt.xlabel(feature_name)
    plt.ylabel('Density')
    plt.legend()
    plt.show()


# for feature_name in ['activated_pixels', 'average_intensity', 'max_intensity', 'total_intensity']:
#     plot_density(features, feature_name)


def plot_cdf(features, feature_name):
    plt.figure(figsize=(12, 8))

    for radiation_type, feature_dict in features.items():
        # Calculate the CDF
        sorted_data = np.sort(feature_dict[feature_name])
        yvals = np.arange(len(sorted_data))/float(len(sorted_data)-1)
        plt.plot(sorted_data, yvals, label=f"{radiation_type} {feature_name}")

    plt.title(f'Cumulative Distribution Function (CDF) of {feature_name}')
    plt.xlabel(feature_name)
    plt.ylabel('Probability')
    plt.legend()
    plt.show()

# CDF for each feature
for feature_name in ['activated_pixels', 'average_intensity', 'max_intensity', 'total_intensity']:
    plot_cdf(features, feature_name)