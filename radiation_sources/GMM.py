import os
import cv2
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA

""" 
    Gaussian Mixture Model, including Principal Analysis Component for visualisation.
"""


def load_images_from_folder(folder):
    image_paths = []
    for filename in os.listdir(folder):
        if filename.endswith(".png"):  
            image_paths.append(os.path.join(folder, filename))
    return image_paths

def extract_track_features(image):
    features = {
        'lengths': [],
        'widths': [],
        'total_intensity': [],
        'activated_pixels': []
    }
    _, thresh = cv2.threshold(image, 30, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    for contour in contours:

        length = cv2.arcLength(contour, closed=False)
        features['lengths'].append(length)


        _, _, w, h = cv2.boundingRect(contour)
        width = min(w, h) 
        features['widths'].append(width)
        
        mask = np.zeros_like(image)
        cv2.drawContours(mask, [contour], -1, 255, -1)  
        activated_pixels = np.sum(mask == 255)
        total_intensity = np.sum(image[mask == 255])
        
        features['activated_pixels'].append(activated_pixels)
        features['total_intensity'].append(total_intensity)
        
    return features

def extract_features(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    track_features = extract_track_features(image)
    
    lengths = np.mean(track_features['lengths'])
    widths = np.mean(track_features['widths'])
    
    
    total_intensity = np.sum(track_features['total_intensity'])
    activated_pixels = np.sum(track_features['activated_pixels'])

    return [lengths, widths, total_intensity, activated_pixels]

def plot_feature_pairs(features, labels):
    df = pd.DataFrame(features, columns=['Length', 'Width', 'Total Intensity', 'Activated Pixels'])
    df['Cluster'] = labels
    pairplot = sns.pairplot(df, hue='Cluster', vars=['Length', 'Width', 'Total Intensity', 'Activated Pixels'])
    pairplot.fig.suptitle("Pairwise Feature Relationships", y=1.08)  
    plt.show()
    
def plot_clusters_2d(features, labels, title):
    pca = PCA(n_components=2)
    reduced_features = pca.fit_transform(features)
    
    df = pd.DataFrame(reduced_features, columns=['Component 1', 'Component 2'])
    df['Cluster'] = labels
    
    plt.figure(figsize=(10, 8))
    sns.scatterplot(data=df, x='Component 1', y='Component 2', hue='Cluster', palette='viridis', s=100, alpha=0.7, edgecolor='k')
    plt.title(title)
    plt.xlabel('Component 1')
    plt.ylabel('Component 2')
    plt.legend(title='Cluster')
    plt.grid(True)
    plt.show()


image_directory = '2-sodium'  
image_paths = load_images_from_folder(image_directory)

all_features = []
for image_path in image_paths:
    features = extract_features(image_path)
    all_features.append(features)
all_features_array = np.array(all_features)

# normalization
scaler = StandardScaler()
all_features_normalized = scaler.fit_transform(all_features_array)

# gausian mixture model
gmm = GaussianMixture(n_components=3, random_state=42)
labels = gmm.fit_predict(all_features_normalized)

plot_feature_pairs(all_features_normalized, labels)
#plot_clusters_2d(all_features_normalized, labels, "GMM Clustering with PCA")
