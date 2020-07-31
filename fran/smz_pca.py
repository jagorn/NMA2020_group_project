import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

def reduce_dimensionality(dataset, desired_variance):

    x = StandardScaler().fit_transform(dataset)
    n_components=2

    pca = PCA(n_components=n_components)
    principalComponents = pca.fit_transform(x)
    explained_variance = np.sum(pca.explained_variance_ratio_)

    while explained_variance < desired_variance:
        n_components += 1
        pca = PCA(n_components=n_components)
        principalComponents = pca.fit_transform(x)
        explained_variance = np.sum(pca.explained_variance_ratio_)
        print("components = " + str(n_components) + ", total variance = " + str(explained_variance))

    print("Number of components = " + str(n_components) + ". Explained variance = " + str(explained_variance))
    return principalComponents
