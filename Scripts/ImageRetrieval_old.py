import os
import numpy as np
import scipy as sp

from tqdm import tqdm
from statistics import mode
from abc import ABC, abstractmethod
from sklearn.neighbors import NearestNeighbors


class ImageRetrieval(ABC):
    def __init__(self, query_features, query_labels):
        self.query_features = query_features
        self.query_labels = query_labels

    @abstractmethod
    def retrieve_images(self, images, query_features, query_labels):
        pass

    def save_results(self, images, labels):
        """Save the results to a csv file"""
        path = os.path.join(os.getcwd(), "results.csv")
        with open(path, "w") as f:
            for image, label in zip(images, labels):
                f.write(f"{image},{label}\n")

class BaseRetrieval(ImageRetrieval):
    def __init__(self, query_features, query_labels, k=5, metric="cosine"):
        super().__init__(query_features, query_labels)
        self.k = k
        self.metric = metric
    
    def retrieve_images(self, images):
        """Retrieve the k nearest images to the query images"""
        neigh = NearestNeighbors(n_neighbors=self.k, n_jobs=-1, metric=self.metric)
        neigh.fit(images)
        
        distances, indices = neigh.kneighbors(self.query_features)
        
        return distances, indices
    
    

class KNNRetrieval(ImageRetrieval):
    def __init__(self, query_features, query_labels, k=5):
        super().__init__(query_features, query_labels)
        self.k = k
        self.min_class = min(query_labels)
        self.max_class = max(query_labels)

    def retrieve_images(self, images, metric="cosine"):
        """Retrieve the k nearest images to the query images"""

        neigh = NearestNeighbors(n_neighbors=self.k, n_jobs=-1, metric=metric)
        neigh.fit(self.query_features)

        distances, indices = neigh.kneighbors(images)

        return self._augment_dataset2(distances, indices, metric)

    def _augment_dataset(self, distances, indices, metric="cosine"):
        """Augment the dataset with the query images"""
        new_labels = np.zeros(indices.shape[0], dtype=np.int32) - 1
        seen = set()
        for i in tqdm(
            range(
                self.min_class,
                self.max_class + 1,
            )
        ):
            # Calcola le features e la distanza mediana della classe una sola volta
            idxs = np.where(self.query_labels == i)[0]
            class_feat = self.query_features[idxs]
            median_d_class = np.median(
                sp.spatial.distance.pdist(class_feat, metric=metric)
            )

            # Converti idxs in set una sola volta
            idxs_set = set(idxs)

            # Vettorizza il controllo delle intersezioni
            intersections = np.array(
                [len(idxs_set.intersection(set(idx))) for idx in indices]
            )
            valid_indices = np.where(intersections >= 3)[0]

            # Processa solo gli indici validi non ancora visti
            valid_indices = [j for j in valid_indices if j not in seen]
            for j in valid_indices:
                intersection = idxs_set.intersection(set(indices[j]))
                positions = [
                    list(indices[j]).index(idx) for idx in sorted(intersection)
                ]
                d = distances[j][positions]

                # Vettorizza il controllo delle distanze
                if np.any(d <= median_d_class):
                    new_labels[j] = i
                    seen.add(j)

        return new_labels

    def _augment_dataset2(self, distances, indices, metric="cosine"):
        """Augment the dataset with the query images"""
        new_labels = np.full(indices.shape[0], -1, dtype=np.int32)

        labels = []
        for idx in indices:
            labels.append(self.query_labels[idx])

        medians = np.full(self.max_class - self.min_class + 1, -1, dtype=np.float32)

        seen = set()
        for i, lab in tqdm(enumerate(labels)):
            _class = mode(lab)

            idxs = np.where(self.query_labels == _class)[0]
            class_feat = self.query_features[idxs]

            if medians[_class] == -1:
                medians[_class] = np.median(
                    sp.spatial.distance.pdist(class_feat, metric=metric)
                )
            if np.count_nonzero(lab == _class) >= 3:
                positions = [i for i, x in enumerate(lab) if x == _class]
                d = distances[positions]

                if np.any(d <= medians[_class]):
                    new_labels[i] = _class
                    seen.add(i)
        return new_labels

    def _expand_less_frequent_class():
        pass


class FAISSRetrieval(ImageRetrieval):
    def __init__(self, query_paths, query_labels, k=5):
        super().__init__(query_paths, query_labels)
        self.k = k

    def retrieve_images(self, images, query_features, query_labels):
        """Retrieve the k nearest images to the query images"""
        pass
