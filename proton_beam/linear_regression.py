import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

""" 
    Linear Regression Analysis: Total particle counts vs MU with Regression and Confidence Bands.
"""

# data
processed_csv = 'total_per_mu.csv'
total_counts_df = pd.read_csv(processed_csv)
mu_classes = total_counts_df['MU_Class'].values.reshape(-1, 1)
particle_counts = total_counts_df['Total_Particle_Count'].values

# regression
model = LinearRegression()
model.fit(mu_classes, particle_counts)
predicted_counts = model.predict(mu_classes)

# coefficients
slope = model.coef_[0]
intercept = model.intercept_

# MSE
mse = mean_squared_error(particle_counts, predicted_counts)
print(f'Equation: Total Particle Count = {intercept:.2f} + {slope:.2f} * MU Class')
print(f'Mean Squared Error: {mse:.2f}')

# confidence intervals
pred_std = np.sqrt(mse)
confidence_interval = 1.96 * pred_std

# plot
plt.figure(figsize=(10, 10)) 
plt.scatter(mu_classes, particle_counts, color='blue', label='Data Points')
plt.plot(mu_classes, predicted_counts, color='red', linestyle='-', label=f'Regression Line\nMSE={mse:.2f}')
plt.fill_between(
    mu_classes.flatten(),
    (predicted_counts - confidence_interval).flatten(),
    (predicted_counts + confidence_interval).flatten(),
    color='red', alpha=0.2, label='95% Confidence Band'
)

# equation on the plot
plt.text(0.05, 0.85, f'Equation: y = {intercept:.2f} + {slope:.2f}x', 
         transform=plt.gca().transAxes, fontsize=12, verticalalignment='top', bbox=dict(facecolor='white', alpha=0.8))

plt.xlabel('MU Class')
plt.ylabel('Total Particle Count')
plt.title('Total Particle Count vs MU Class with Regression and Confidence Bands')
plt.legend(loc='upper left') 
plt.tight_layout()
plt.savefig('counts_vs_mu_regression.png')
plt.show()
