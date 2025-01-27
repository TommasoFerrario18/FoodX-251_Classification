import os
import numpy as np

from abc import ABC, abstractmethod
from sklearn.neighbors import NearestNeighbors


class ImageRetrieval(ABC):
    def __init__(self, dataset_paths, query_paths, query_labels):
        self.dataset_paths = dataset_paths
        self.query_paths = query_paths
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


class KNNRetrieval(ImageRetrieval):
    def __init__(self, dataset_paths, query_paths, query_labels, k=5):
        super().__init__(dataset_paths, query_paths, query_labels)
        self.k = k

    def retrieve_images(self, images, query_features, query_labels):
        """Retrieve the k nearest images to the query images"""
        labels = np.zeros(len(images))

        neigh = NearestNeighbors(n_neighbors=self.k, n_jobs=-1, metric="cosine")
        neigh.fit(query_features)

        distances, indices = neigh.kneighbors(images)

        # # Get the labels of the k nearest images
        # for i, idx in enumerate(indices):
        #     labels[i] = query_labels[idx]

        return indices, distances


class FAISSRetrieval(ImageRetrieval):
    def __init__(self, dataset_paths, query_paths, query_labels, k=5):
        super().__init__(dataset_paths, query_paths, query_labels)
        self.k = k

    def retrieve_images(self, images, query_features, query_labels):
        """Retrieve the k nearest images to the query images"""
        pass
