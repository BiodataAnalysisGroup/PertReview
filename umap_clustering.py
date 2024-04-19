import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import umap
from matplotlib.patches import Ellipse
from matplotlib.colors import to_rgba

# Load CSV file into a DataFrame
df = pd.read_csv('Pert_methods12.csv')

# Replace NaN or blank values in the DataFrame with 0
df.fillna(0, inplace=True)

# Extract tool names (methods) from the first column of the DataFrame
tool_names = df.iloc[:, 0].values

# Extract feature columns from the DataFrame (excluding the first column)
feature_columns = df.columns[1:]
X = df[feature_columns].values  # Convert feature columns to numpy array

# Standardize the feature matrix (important for UMAP)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Apply UMAP for dimensionality reduction
umap_reducer = umap.UMAP(n_components=2, random_state=42)
embedding = umap_reducer.fit_transform(X_scaled)

# Apply K-means clustering to the UMAP embeddings
num_clusters = 7  # Number of clusters for K-means
kmeans = KMeans(n_clusters=num_clusters, random_state=42)
cluster_labels = kmeans.fit_predict(embedding)

# Manually assign specific tools to a different cluster
tools_to_reassign = ["GrouNdGAN", "graphVCI", "PRESCIENT", "OntoVAE", "UNAGI"]
desired_cluster = 1  # Assign to a different cluster (e.g., Cluster 5)

# Find the indices of the tools to reassign in the tool_names array
tool_indices_to_reassign = [np.where(tool_names == tool)[0][0] for tool in tools_to_reassign]

# Modify the cluster labels for the selected tools
cluster_labels[tool_indices_to_reassign] = desired_cluster

# Plot the UMAP visualization with updated clustering and transparent ellipses
plt.figure(figsize=(12, 10))
for cluster in np.unique(cluster_labels):
    cluster_points = embedding[cluster_labels == cluster]
    cluster_color = plt.cm.tab10(cluster / num_clusters)  # Access color directly from the colormap
    cluster_color_transparent = to_rgba(cluster_color, alpha=0.2)  # Make the color transparent
    plt.scatter(cluster_points[:, 0], cluster_points[:, 1] - 0.1,
                label=f'Cluster {cluster}', color=cluster_color, s=50, alpha=0.5)
    # Draw transparent ellipses containing all points in the cluster
    cov_matrix = np.cov(cluster_points.T)
    eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)
    major_axis = 2 * np.sqrt(5.991 * eigenvalues[0])  # 95% confidence interval
    minor_axis = 2 * np.sqrt(5.991 * eigenvalues[1])  # 95% confidence interval
    angle = np.degrees(np.arctan2(*eigenvectors[:, 0][::-1]))
    ellipse = Ellipse(xy=np.mean(cluster_points, axis=0), width=major_axis, height=minor_axis, angle=angle,
                      alpha=0.2, color=cluster_color_transparent)
    plt.gca().add_patch(ellipse)
    # Annotate tool names for points in the current cluster
    for i in np.where(cluster_labels == cluster)[0]:
        plt.text(embedding[i, 0], embedding[i, 1], tool_names[i], fontsize=8, ha='center', va='center')

plt.title('UMAP Visualization with Updated Clustering and Transparent Ellipses')
plt.xlabel('UMAP Component 1')
plt.ylabel('UMAP Component 2')
plt.show()
