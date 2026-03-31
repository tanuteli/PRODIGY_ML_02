# Import libraries
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# Step 1: Load dataset
data = pd.read_csv("Mall_Customers.csv")

print("Dataset Preview:")
print(data.head())

# Step 2: Select features for clustering
# Annual Income and Spending Score
X = data[['Annual Income (k$)', 'Spending Score (1-100)']]

# Step 3: Apply K-Means
kmeans = KMeans(n_clusters=5, random_state=42)

# Train model
kmeans.fit(X)

# Step 4: Assign cluster labels
data['Cluster'] = kmeans.labels_

print("\nCustomer Groups:")
print(data.head())

# Step 5: Visualize clusters
plt.scatter(
    X['Annual Income (k$)'],
    X['Spending Score (1-100)'],
    c=kmeans.labels_
)

plt.xlabel("Annual Income")
plt.ylabel("Spending Score")
plt.title("Customer Segmentation using K-Means")

plt.show()
