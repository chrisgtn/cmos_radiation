
import cv2
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Load the image
image = cv2.imread('data/1-uranium/uranium_33_18.12_12.07.09.png', cv2.IMREAD_GRAYSCALE)

# coordinates
x = np.arange(image.shape[1])
y = np.arange(image.shape[0])
X, Y = np.meshgrid(x, y)

# figure and 3D axis
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')

# surface plot
surf = ax.plot_surface(X, Y, image, cmap='viridis', edgecolor='none')
fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5)

# plot
ax.set_xlabel('X Coordinate')
ax.set_ylabel('Y Coordinate')
ax.set_zlabel('Pixel Intensity')
plt.savefig('mesh_plots/Uranium.png')
plt.show()