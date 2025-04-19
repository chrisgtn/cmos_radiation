import cv2
import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN

""" 
Image Processing Algorithm for particle counting using K-means Colour quantization based on  https://stackoverflow.com/questions/72118665/particle-detection-with-python-opencv
K-means color quantization is simplifying the representation of the image by reducing the numbers of colours. This process involves clustering pixel colors into k groups 
and replacing each pixel with its cluster’s centroid color. In this application, k-means is performed with two clusters to create a grayscale image and Otsu’s thresholding is for binarization. 
After filtering out tiny noise, using contour area filtering and a mask is created to erase it. Bitwise-operation applies this mask to the original image, resulting in significant particle clusters.
The number of particle counts and average particle size computed from this algorithm are saved in a csv file for further processing.
"""


def kmeans_color_quantization(image, clusters=8, rounds=1):
    h, w = image.shape[:2]
    samples = np.zeros([h*w,3], dtype=np.float32)
    count = 0

    for x in range(h):
        for y in range(w):
            samples[count] = image[x][y]
            count += 1

    compactness, labels, centers = cv2.kmeans(samples,
            clusters, 
            None,
            (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10000, 0.0001), 
            rounds, 
            cv2.KMEANS_RANDOM_CENTERS)

    centers = np.uint8(centers)
    res = centers[labels.flatten()]
    return res.reshape((image.shape))

def detect_edges_and_tracks(image, low_threshold=50, high_threshold=150, min_line_length=50, max_line_gap=10):
    # grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # gaussian blur
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    # edges using canny algorithm
    edges = cv2.Canny(blurred, low_threshold, high_threshold)
    # hough line transform
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 100, minLineLength=min_line_length, maxLineGap=max_line_gap)
    

    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(image, (x1, y1), (x2, y2), (255, 0, 0), 2)
    
    return edges, lines, image


def process_image(image_path, threshold=0, AREA_THRESHOLD=1):

    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Image at path {image_path} could not be loaded.")
    original = image.copy()
    
    
    # kmeans color segmentation, grayscale and otsu's threshold
    kmeans = kmeans_color_quantization(image, clusters=2)
    gray = cv2.cvtColor(kmeans, cv2.COLOR_BGR2GRAY)
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    
    #edges, lines, processed_image = detect_edges_and_tracks(original)

    # contour area filtering + gather points
    points_list = []
    size_list = []
    cnts, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2:]
    for c in cnts:
        area = cv2.contourArea(c)
        if area < AREA_THRESHOLD:
            cv2.drawContours(thresh, [c], -1, 0, -1)
        else:
            (x, y), radius = cv2.minEnclosingCircle(c)
            points_list.append((int(x), int(y)))
            size_list.append(area)

    # mask
    result = cv2.bitwise_and(original, original, mask=thresh)
    result[thresh == 255] = (36, 255, 12)

    # overlay on original
    original[thresh == 255] = (36, 255, 12)

    return len(points_list), size_list, original, kmeans, thresh, result

# directories
runs_directory = "runs_mu"
classes_dirs = [d for d in os.listdir(runs_directory) if os.path.isdir(os.path.join(runs_directory, d))]

# csv headers
particle_data = pd.DataFrame(columns=['Class', 'Frame', 'Particle_Count', 'Avg_Particle_Size'])

# for each class
for class_dir in classes_dirs:
    class_path = os.path.join(runs_directory, class_dir)
    frame_files = [f for f in os.listdir(class_path) if f.endswith('.png')]
    class_particle_counts = []
    class_particle_sizes = []
    temp_data = []
    total_particles = 0
    all_particle_sizes = []
    last_images = {} 
    
    last_kmeans = None
    last_thresh = None
    last_result = None
    last_frame_file = None

    
    for frame_file in frame_files:
        frame_path = os.path.join(class_path, frame_file)

        particle_count, size_list, original, kmeans, thresh, result = process_image(frame_path)
        total_particles += particle_count
        all_particle_sizes.extend(size_list)
        class_particle_counts.append(particle_count)
        class_particle_sizes.extend(size_list)
        # particle_data = particle_data.append({
        new_row = pd.DataFrame([{
            'Class': class_dir,
            'Frame': frame_file,
            'Particle_Count': particle_count,
            'Avg_Particle_Size': np.mean(size_list) if size_list else 0
        }])
        temp_data.append(new_row)
    
        # Concatenate all new rows for this class to the main DataFrame
        particle_data = pd.concat([particle_data, pd.concat(temp_data, ignore_index=True)], ignore_index=True)
        
        # # Update last_images to hold the latest images processed
        # last_kmeans = kmeans
        # last_thresh = thresh
        # last_result = result
        # last_frame_file = frame_file
        
    # Create a histogram of particle counts for this class
    class_particle_counts = particle_data[particle_data['Class'] == class_dir]['Particle_Count']
    plt.figure(figsize=(10, 6))
    plt.hist(class_particle_counts, bins=50, alpha=0.5, label=class_dir)
    plt.xlabel('Particle Count')
    plt.ylabel('Frequency')
    plt.title(f'Particle Count Distribution for Class {class_dir}')
    plt.legend()
    plt.tight_layout()
    plt.show()
    
    # if last_frame_file:
    #     cv2.imwrite(f'{class_path}/last_frame_kmeans_{last_frame_file}.png', last_kmeans)
    #     cv2.imwrite(f'{class_path}/last_frame_thresh_{last_frame_file}.png', last_thresh)
    #     cv2.imwrite(f'{class_path}/last_frame_result_{last_frame_file}.png', last_result)
        
    # average particle size
    average_particle_size = np.mean(all_particle_sizes) if all_particle_sizes else 0

    #Print class summary
    total_particles = sum(class_particle_counts)
    avg_particle_size = np.mean(class_particle_sizes) if class_particle_sizes else 0
    
    # summary 
    summary_row = pd.DataFrame([{
        'Class': class_dir,
        'Frame': 'Total',
        'Particle_Count': total_particles,
        'Avg_Particle_Size': average_particle_size
    }])
    particle_data = pd.concat([particle_data, summary_row], ignore_index=True)
    print(f"Class: {class_dir}, Total Particles: {total_particles}, Average Particle Size: {average_particle_size:.2f}")


# save csv 
particle_data.to_csv('counts_mu.csv', index=False)

print("CSV file with particle counts has been saved.")