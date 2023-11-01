import pandas as pd
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # Import 3D plotting tools

# Load the dataset
data = pd.read_csv('Aflatoxin level detection Reflectance Dark Current Reducted Bilateral Filtered Data Set (Super Pixel) Dataset.csv')

# Extract features (X1 to X13) and labels
features = data.iloc[:, 0:13]
labels = data['Labels']

# Perform PCA
pca = PCA(n_components=4)  # You can change n_components as needed
pca_result = pca.fit_transform(features)

# Create a new DataFrame for PCA results
pca_columns = [f'PC{i+1}' for i in range(pca_result.shape[1])]
pca_df = pd.DataFrame(data=pca_result, columns=pca_columns)

# Add labels to the PCA DataFrame
pca_df['Labels'] = labels

# Save the PCA results to a CSV file
pca_df.to_csv('Aflatoxin level detection Reflectance Dark Current Reducted Bilateral Filtered Data Set (Super Pixel) Dataset with PCA.csv', index=False)

# Create two subplots
plt.figure(figsize=(12, 6))

# Subplot 1: PCA1 vs PCA2
plt.subplot(1, 2, 1)
plt.scatter(pca_df['PC1'], pca_df['PC2'], c=labels, cmap='viridis')
plt.xlabel('PCA1')
plt.ylabel('PCA2')
plt.title('Projected Aflatoxin dataset onto first two PCs')

# Subplot 2: 3D Plot with PCA1, PCA2, and PCA3
ax = plt.subplot(1, 2, 2, projection='3d')
scatter = ax.scatter(pca_df['PC1'], pca_df['PC2'], pca_df['PC3'], c=labels, cmap='viridis')
ax.set_xlabel('PCA1')
ax.set_ylabel('PCA2')
ax.set_zlabel('PCA3')
ax.set_title('Projected Aflatoxin dataset onto first three PCs')

# Add a colorbar
colorbar = plt.colorbar(scatter, ax=ax)
colorbar.set_label('Labels')

# Show the plots
plt.tight_layout()
plt.show()
