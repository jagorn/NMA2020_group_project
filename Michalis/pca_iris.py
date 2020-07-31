import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import datasets
from sklearn.preprocessing import StandardScaler


# Load and preprocess the iris dataset
iris = datasets.load_iris()
X = iris.data # rows --> a single flower, columns --> properties (petal lentgh etc.) 
targets = iris.target # matching data rows to flower species
target_names = iris.target_names # names of flower species

# SCale the data
""" Standardize the dataset’s features onto unit scale (mean = 0 and
    variance = 1). We don’t want some feature to be voted as “more important”
    due to scale differences. """
X_scaled = StandardScaler().fit_transform(X)

# Calculate the covariance matrix
features = X_scaled.T
cov_matrix = np.cov(features)

#Eigendecomposition: square matrix --> eigenvalues and eigenvectors
values, vectors = np.linalg.eig(cov_matrix)

# Calculate the proportion of explained variance per principal component
exp_var = [value / np.sum(values) for value in values]
print(f"\nExplained variance per principal component: \n{exp_var} \n")

# Project the data to calculate principal components
pc1 = X_scaled.dot(vectors.T[0])
pc2 = X_scaled.dot(vectors.T[1])

# Visualize results
sns.set()
for target_id, name in enumerate(target_names):
    indices = np.where(targets == target_id)
    plt.plot(pc1[indices], pc2[indices], 'o', label=f"Iris {name}", alpha=0.9)

plt.legend()
plt.xlabel(f'PC1 \n({exp_var[0] * 100: .2f} %)')
plt.ylabel(f'PC2 \n({exp_var[1] * 100: .2f} %)')
plt.tight_layout()
plt.show()
