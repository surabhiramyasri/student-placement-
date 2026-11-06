
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

# Load dataset
df = pd.read_csv("Placement_Data_Full_Class.csv")

# Select academic performance columns
features = ["ssc_p", "hsc_p", "degree_p", "etest_p", "mba_p"]
X = df[features].copy()

# Handle missing values
X = X.fillna(X.mean())

# Scale the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Create 3 student groups
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
df["Cluster"] = kmeans.fit_predict(X_scaled)

# Display group information
print("STUDENT PLACEMENT CLUSTERING")
print("----------------------------")
print("Total students:", len(df))
print("\nStudents in each cluster:")
print(df["Cluster"].value_counts().sort_index())

print("\nAverage academic performance by cluster:")
print(df.groupby("Cluster")[features].mean().round(2))

# Reduce data to 2 dimensions for visualization
pca = PCA(n_components=2)
coordinates = pca.fit_transform(X_scaled)

# Plot the clusters
plt.figure(figsize=(8, 5))
plt.scatter(
    coordinates[:, 0],
    coordinates[:, 1],
    c=df["Cluster"],
    cmap="viridis",
    s=50
)
plt.title("Student Academic Performance Clusters")
plt.xlabel("Component 1")
plt.ylabel("Component 2")
plt.tight_layout()
plt.savefig("student_clusters.png")
plt.show()

# Save cluster results
df.to_csv("placement_clustering_results.csv", index=False)

print("\nCluster graph saved as student_clusters.png")
print("Cluster results saved as placement_clustering_results.csv")
