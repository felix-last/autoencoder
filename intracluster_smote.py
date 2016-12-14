import numpy as np
from sklearn.metrics.pairwise import euclidean_distances

class IntraclusterSmote:
    def __init__(self, n_intra, imbalance_ratio_threshold=1.0):
        """ 
        Initialize the model
        Args:
            n_intra (int): The total (intracluster) number of synthetic samples to be generated
            imbalance_ratio_threshold (float): Ratio of (majority cases + 1 / minority cases + 1) below which a cluster is considered a minority cluster (default 1, corresponds to 50-50)
        """
        self.n_intra = n_intra
        self.imbalance_ratio_threshold = imbalance_ratio_threshold
        
    def fit(self, X, y, minority, cluster_labels):
        """
        Perform SMOTE in each cluster
        Args:
            X (np.array): Input data
            y (np.array): Mask vector indicating whether an observation belongs to the minority class.
            c (np.array): Vector of cluster assignment. Must match first dimension of data.
        """
        if minority.dtype != 'bool':
            minority = (minority == 1)

        minority_label = y[minority][0]

        filtered_clusters, density_sum = self._filter_clusters(X, y, minority, cluster_labels)
        
        oversampled_X = [X]
        oversampled_y = [y]
        for i, (cluster, density_factor) in enumerate(filtered_clusters):
            weight = (1/density_factor) / (density_sum)
            generate_count = int(np.floor(self.n_intra * weight))
            oversampled_X.append(self._smote(cluster, generate_count))
            oversampled_y.append(np.full((generate_count,), minority_label, dtype=y.dtype))

        oversampled_X = np.random.permutation(np.concatenate(oversampled_X))
        oversampled_y = np.random.permutation(np.concatenate(oversampled_y))

        return oversampled_X, oversampled_y

    def _filter_clusters(self, X, y, minority, cluster_labels):
        filtered_clusters = list()
        density_sum = 0
        for i in np.unique(cluster_labels):
            cluster = X[cluster_labels == i]
            mask = minority[cluster_labels == i]
            minority_count = cluster[mask].shape[0]
            majority_count = cluster[-mask].shape[0]
            imbalance_ratio = (majority_count + 1) / (minority_count + 1)
            if imbalance_ratio < self.imbalance_ratio_threshold:
                average_minority_distance = np.mean(euclidean_distances(cluster))
                if average_minority_distance is 0: average_minority_distance = 1e-10 # to avoid division by 0
                density_factor = minority_count / average_minority_distance**2
                density_sum += (1 / density_factor)
                filtered_clusters.append((cluster, density_factor))
        return filtered_clusters, density_sum

    def _smote(self, X, n):
        generated = np.empty((n,X.shape[1]))
        for i in range(0,n):
            a, b = X[np.random.choice(X.shape[0], size=(2), replace=False)]
            generated[i] = a + ((b-a) * np.random.rand())
        return generated

    ### CLASS METHODS
    def compute_synthetic_count(n, ratio):
        return int(np.floor(n - n * ratio))