import numpy as np
from sklearn.metrics.pairwise import euclidean_distances
import warnings

def compute_synthetic_count(n, ratio):
    return int(np.floor(n - n * ratio))

class IntraclusterSmote:
    def __init__(self, n_intra, k, imbalance_ratio_threshold=1.0, decoder=None, save_creation_examples=0):
        """ 
        Initialize the model
        Args:
            n_intra (int): The total (intracluster) number of synthetic samples to be generated
            k (int): number of nearest neighbors
            imbalance_ratio_threshold (float): Ratio of (majority cases + 1 / minority cases + 1) below which a cluster is considered a minority cluster (default 1, corresponds to 50-50)
            decoder (function): If a decoder is given, X will be assumed to be in a transformed space and all generated samples will be decoded before returning.
        """
        self.n_intra = n_intra
        self.imbalance_ratio_threshold = imbalance_ratio_threshold
        self.default_decoder = lambda X: X
        if decoder is not None:
            self.decoder = decoder
        else:
            self.decoder = self.default_decoder
        self.k = k

        self.metadata = {}
        self.save_creation_examples = save_creation_examples
        if save_creation_examples:
            self.metadata['creation_examples'] = list()
            self.metadata['encoded_creation_examples'] = list()
        
        
    def fit_transform(self, X, y, minority_label, cluster_labels, X_unenc=None):
        """
        Perform SMOTE in each cluster
        Args:
            X (np.array): Input data (encoded)
            y (np.array): Vector assigning classes to X.
            minority_label (y.dtype): The minority class label in y.
            cluster_labels (np.array): Vector of cluster assignment. Must match first dimension of data.
            X_unenc: Unencoded X. Used if creation examples are kept.
        """
        X = np.asarray(X)
        y = np.asarray(y)

        minority_mask = (y == minority_label)

        if X_unenc is None:
            print('X_unenc not given.')
            X_unenc = X

        filtered_clusters, density_sum = self._filter_clusters(X, y, minority_mask, cluster_labels, X_unenc)

        # save number of minority clusters for later analysis
        self.metadata['minority_cluster_count'] = len(filtered_clusters)
        if len(filtered_clusters) < 1:
            # if no minority clusters can be identified, warn and perform regular smote
            minority_count, majority_count = X[minority_mask].shape[0], X[-minority_mask].shape[0]
            warning_msg = 'No minority clusters found. Performing regular SMOTE. Try changing the number of clusters. Recommended number of clusters: ' + str(minority_count) + ' to ' + str(majority_count) + '.'
            warnings.warn(warning_msg)
            # regular smote is achieved by pretending the entire dataset is a minority cluster
            filtered_clusters = [(X,1,minority_mask,X_unenc)]
            density_sum = 1

        oversampled_X = X
        oversampled_y = y
        synthetic_X = []
        synthetic_y = []
        for i, (cluster, density_factor, minority_mask, cluster_unenc) in enumerate(filtered_clusters):
            weight = (1/density_factor) / (density_sum)
            generate_count = int(np.floor(self.n_intra * weight))
            synthetic_X.append(self._smote(cluster, generate_count, minority_mask, cluster_unenc))
            synthetic_y.append(np.full((generate_count,), minority_label, dtype=y.dtype))

        # save the encoded generated samples
        self.metadata['generated_samples_encoded'] = np.concatenate(synthetic_X)

        d = self.decoder
        synthetic_X = d(np.random.permutation(np.concatenate(synthetic_X)))
        synthetic_y = np.concatenate(synthetic_y) # don't permute because y is all minority class
        oversampled_X = np.concatenate([d(oversampled_X), synthetic_X])
        oversampled_y = np.concatenate([oversampled_y, synthetic_y])
        oversampled_y = oversampled_y.reshape(oversampled_y.shape[0], 1)
        oversampled = np.random.permutation(np.hstack((oversampled_X, oversampled_y)))
        oversampled_X, oversampled_y = np.hsplit(oversampled, [-1])
        oversampled_y = oversampled_y.reshape(oversampled_y.shape[0])

        return (oversampled_X, oversampled_y), (synthetic_X, synthetic_y), self.metadata

    def _filter_clusters(self, X, y, minority_mask, cluster_labels, X_unenc):
        filtered_clusters = list()
        density_sum = 0
        for i in np.unique(cluster_labels):
            cluster = X[cluster_labels == i]
            cluster_unenc = X_unenc[cluster_labels == i]
            mask = minority_mask[cluster_labels == i]
            minority_count = cluster[mask].shape[0]
            majority_count = cluster[-mask].shape[0]
            imbalance_ratio = (majority_count + 1) / (minority_count + 1)
            if imbalance_ratio < self.imbalance_ratio_threshold and minority_count > 1:
                average_minority_distance = np.mean(euclidean_distances(cluster))
                if average_minority_distance is 0: average_minority_distance = 1e-1 # to avoid division by 0
                # issue here! density factor often becomes inf!
                density_factor = minority_count / average_minority_distance**2
                density_sum += (1 / density_factor)
                filtered_clusters.append((cluster, density_factor, mask, cluster_unenc))
        return filtered_clusters, density_sum

    def _smote(self, X, n, minority_mask, X_unenc):
        X, X_unenc = X[minority_mask], X_unenc[minority_mask]
        generated = np.empty((n,X.shape[1]))
        d = self.decoder
        for i in range(0,n):
            a_index = np.random.choice(X.shape[0], size=(1))[0]
            
            # find a random nearest neighbor
            k = self.k if X.shape[0] >= self.k else X.shape[0]
            distances = np.sqrt(((X[a_index] - X)**2).sum(axis=1))
            indices_k_nearest_neighbors = distances.argsort()[1:k+1]
            b_index = np.random.choice(indices_k_nearest_neighbors, size=(1))[0]

            a, b = X[a_index], X[b_index]
            generated[i] = a + ((b-a) * np.random.rand())
            if self.save_creation_examples > 0 and self.decoder != self.default_decoder:
                self.metadata['encoded_creation_examples'].append((a,generated[i],b))
            if self.save_creation_examples > np.random.rand():
                decode_single_instance = lambda x: d(np.asarray([x]))[0] # use to turn a single instance into multi-row dataset, so the decoder can work with it, then reshape back
                self.metadata['creation_examples'].append((X_unenc[a_index], decode_single_instance(generated[i]), X_unenc[b_index]))
        return generated
