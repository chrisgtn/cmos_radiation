import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

""" 
    Kernel Density Functions: Smooth KDE for Particle Count and Average Particle Size Distributions for each Energy class.
"""

df = pd.read_csv('particle_counts.csv')
print(df.head())

sns.set(style="whitegrid")


fig, axs = plt.subplots(nrows=2, ncols=1, figsize=(12, 10))

for class_label in df['Class'].unique():
    subset = df[df['Class'] == class_label]
    sns.kdeplot(subset['Avg_Particle_Size'], ax=axs[0], label=str(class_label))

axs[0].set_title('Density Estimation of Average Particle Size by Class')
axs[0].set_xlabel('Average Particle Size')
axs[0].set_ylabel('Density')
axs[0].legend(title='Class')

for class_label in df['Class'].unique():
    subset = df[df['Class'] == class_label]
    sns.kdeplot(subset['Particle_Count'], ax=axs[1], label=str(class_label))

axs[1].set_title('Density Estimation of Particle Count by Class')
axs[1].set_xlabel('Particle Count')
axs[1].set_ylabel('Density')
axs[1].legend(title='Class')
plt.tight_layout()
plt.show()

plt.figure(figsize=(12, 8))
for class_label in df['Class'].unique():
    subset = df[df['Class'] == class_label]
    bandwidth = 0.1 if class_label == 244 else 3.0
    #sns.kdeplot(subset['Particle_Count'], label=f'Class {class_label}', bw_adjust=bandwidth)
    sns.kdeplot(subset['Avg_Particle_Size'], label=f'Class {class_label}', bw_adjust=3)


plt.title('Smooth KDE of Average Particle Size Distribution by Class')
plt.xlabel('Average Particle Size')
plt.ylabel('Density')
plt.xlim(0, 10)  
# plt.xscale('log') 
plt.legend(title='Class')
plt.show()

print(df['Avg_Particle_Size'].describe())
df['Energy'] = df['Class'].str.extract('(\d+)').astype(float)  

