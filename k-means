import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns

# -----------------------------
# Load dataset (SAFE)
# -----------------------------
try:
    df = pd.read_csv('iris(2)(1).csv')
except FileNotFoundError:
    print("Error: File not found!")
    exit()   # ✅ stop execution

print(df.head())
print(df.columns)

# -----------------------------
# Select features
# -----------------------------
feature_columns = ['sepal length (cm)', 'sepal width (cm)', 
                   'petal length (cm)', 'petal width (cm)']

if all(col in df.columns for col in feature_columns):
    X = df[feature_columns].copy()
else:
    # fallback option
    X = df.select_dtypes(include=['number']).copy()

# -----------------------------
# Elbow Method
# -----------------------------
wcss = []

for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, init='k-means++',
                    max_iter=300, n_init=10, random_state=42)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)

# Plot elbow graph
plt.figure(figsize=(8, 5))
plt.plot(range(1, 11), wcss, marker='o')
plt.title('Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.grid()
plt.show()

# -----------------------------
# Apply KMeans (k=3)
# -----------------------------
kmeans = KMeans(n_clusters=3, random_state=42)
clusters = kmeans.fit_predict(X)

# Add cluster labels to original dataset
df['cluster'] = clusters

# -----------------------------
# Visualization
# -----------------------------
plt.figure(figsize=(8, 6))
sns.scatterplot(x=X.iloc[:, 0], y=X.iloc[:, 1],
                hue=clusters, palette='viridis', s=100)

# Centroids
centroids = kmeans.cluster_centers_
plt.scatter(centroids[:, 0], centroids[:, 1],
            s=300, c='red', marker='X', label='Centroids')

plt.title('K-Means Clustering (k=3)')
plt.xlabel(X.columns[0])
plt.ylabel(X.columns[1])
plt.legend()
plt.grid()
plt.show()

# -----------------------------
# Output
# -----------------------------
print("Cluster count:")
print(df['cluster'].value_counts())
