import cv2
import glob
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt

""" 
    Principal Component Analysis on feature extraction and K-means for clustering.
"""

def load_images_grayscale(image_path):
    images = [cv2.imread(file, cv2.IMREAD_GRAYSCALE) for file in glob.glob(image_path)]
    return [img for img in images if img is not None]

def preprocess_image(image):
    # noise reduction
    image = cv2.GaussianBlur(image, (5, 5), 0)
    # contrast enhancement
    image = cv2.equalizeHist(image)
    # binarization
    _, binary_image = cv2.threshold(image, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

    return binary_image

# def analyze_spatial_patterns(image):
#     edges = cv2.Canny(image, 100, 200)
#     return edges


def extract_features(images):
    features = []
    for img in images:
        max_intensity = np.max(img)
        mean_intensity = np.mean(img)
        activated_pixels = np.sum(img > 0) 
        features.append([max_intensity, mean_intensity, activated_pixels])

    return np.array(features)

def apply_pca(features, num_components=3):
    pca = PCA(n_components=num_components)
    principal_components = pca.fit_transform(features)
    return principal_components

def apply_kmeans(features, n_clusters=5):
    kmeans = KMeans(n_clusters=n_clusters)
    labels = kmeans.fit_predict(features)
    return labels, kmeans.cluster_centers_


def calculate_cluster_properties(image, labels, cluster_centers):
    properties = []
    for i in range(len(cluster_centers)):
        cluster_pixels = np.column_stack(np.where(labels == i))
        size = len(cluster_pixels)

        # elongation and compactness
        if size > 1:
            x_coords, y_coords = zip(*cluster_pixels)
            x_range, y_range = max(x_coords) - min(x_coords), max(y_coords) - min(y_coords)
            elongation = max(x_range, y_range) / max(min(x_range, y_range), 1)
            compactness = size / (x_range * y_range)
        else:
            elongation, compactness = 1, 1

        properties.append({
            'size': size,
            'elongation': elongation,
            'compactness': compactness
        })

    return properties

# load images
alpha_images = load_images_grayscale('alpha/alphas_*.png')
beta_images = load_images_grayscale('/beta/beta_*.png')

if len(alpha_images) == 0:
    print("No alpha images were loaded. Check the file paths.")


print("Number of alpha images loaded:", len(alpha_images))
print("Number of beta images loaded:", len(beta_images))

preprocessed_alpha_images = [preprocess_image(image) for image in alpha_images]
preprocessed_beta_images = [preprocess_image(image) for image in beta_images]


# if len(preprocessed_alpha_images) > 0:
#     print("Shape of the first preprocessed alpha image:", preprocessed_alpha_images[0].shape)
# else:
#     print("No alpha images were preprocessed.")

# extract features
alpha_features = extract_features(alpha_images)

print("Shape of alpha_features:", alpha_features.shape)

pca_alpha_features = apply_pca(alpha_features)

if alpha_features.size > 0 and len(alpha_features.shape) == 2:
    pca_alpha_features = apply_pca(alpha_features)
else:
    print("PCA cannot be applied. Check the feature extraction step.")


# k-means clustering on pca reduced features
labels, cluster_centers = apply_kmeans(pca_alpha_features)

# plot
plt.scatter(pca_alpha_features[:, 0], pca_alpha_features[:, 1], c=labels)
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.title('PCA of Alpha Dataset')
plt.show()

def visualize_kmeans_clusters(pca_features, labels, cluster_centers):
    plt.figure(figsize=(10, 10))

    # scatter plot 
    plt.scatter(pca_features[:, 0], pca_features[:, 1], c=labels, cmap='viridis', s=50, alpha=0.6)
    plt.scatter(cluster_centers[:, 0], cluster_centers[:, 1], c='red', s=100, marker='X')

    # Labelling 
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.title('K-means Clustering on PCA-reduced Features')
    plt.colorbar(label='Cluster Label')

    plt.show()

visualize_kmeans_clusters(pca_alpha_features, labels, cluster_centers)
