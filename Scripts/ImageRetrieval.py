import os
import numpy as np
import scipy as sp
import pandas as pd

from tqdm import tqdm
from statistics import mode
from abc import ABC, abstractmethod
from sklearn.neighbors import NearestNeighbors


class ImageRetrieval(ABC):
    def __init__(self, df_small, feat_small, df_unlabeled, feat_unlabeled):
        self.df_small = df_small
        self.feat_small = feat_small
        self.df_unlabeled = df_unlabeled
        self.feat_unlabeled = feat_unlabeled

    @abstractmethod
    def retrieve_images(self):
        pass

    def save_results(self):
        """Save the results to a csv file"""
        path = os.path.join(os.getcwd(), "results.csv")
        images = self.df_unlabeled.loc[self.df_unlabeled["Label"] != -1, "Image"]
        labels = self.df_unlabeled.loc[self.df_unlabeled["Label"] != -1, "Label"]
        with open(path, "w") as f:
            for image, label in zip(self.df_small["Image"], self.df_small["Label"]):
                f.write(f"{image},{label}\n")
            for image, label in zip(images, labels):
                f.write(f"{image},{label}\n")

    def augment_dataset_single_class(
        self, image_features, label, metric="cosine", k=10
    ):
        indici_m = self.df_unlabeled.index[self.df_unlabeled["Label"] == -1].tolist()
        feat_unlabelled_current = self.feat_unlabeled[indici_m]
        indices_top_k_current = self.get_top_k_neighbours(
            feat_unlabelled_current, image_features, k=k, metric=metric
        )
        indices_unlabelled = [indici_m[i] for i in indices_top_k_current]
        self.df_unlabeled.loc[indices_unlabelled, "Label"] = label

    def get_top_k_neighbours(unlabeled_features, image_features, k=10, metric="cosine"):
        neigh = NearestNeighbors(n_neighbors=k, metric=metric)
        neigh.fit(unlabeled_features)
        _, indices = neigh.kneighbors([image_features])
        return indices[0]


class NearestImageRetrieval(ImageRetrieval):
    def __init__(
        self, df_small, feat_small, df_unlabeled, feat_unlabeled, k=5, metric="cosine"
    ):
        super().__init__(df_small, feat_small, df_unlabeled, feat_unlabeled)

        if k < 1 or k > len(df_small):
            raise ValueError(f"k must be between 1 and {len(df_small)}")

        if feat_small.shape[1] != feat_unlabeled.shape[1]:
            raise ValueError("Feature dimensions must match between datasets")

        self.k = k
        self.metric = metric

    def retrieve_images(self):
        """Retrieve the k nearest images to the query images"""
        neigh = NearestNeighbors(n_neighbors=self.k, n_jobs=-1, metric=self.metric)
        neigh.fit(self.feat_unlabeled)

        distances, indices = neigh.kneighbors(self.feat_small)

        self._post_process(distances, indices)

    def _post_process(self, distances, indices):
        """Post-process the retrieved images"""
        info = dict()

        for i in range(len(self.df_unlabeled)):
            info[i] = {"idxs": [], "dist": []}

        for i in range(distances.shape[0]):
            idx = indices[i]
            ds = distances[i]

            for j in range(len(idx)):
                info[idx[j]]["idxs"].append(i)
                info[idx[j]]["dist"].append(ds[j])

        new_df = pd.DataFrame(info).T
        new_df["Label"] = -1

        for i, row in new_df.iterrows():
            if len(row["dist"]) == 0:
                new_df.at[i, "dist"] = -1
                new_df.at[i, "idxs"] = -1
            else:
                _lab = np.argmin(row["dist"])
                new_df.at[i, "dist"] = row["dist"][_lab]
                new_df.at[i, "Label"] = self.df_small.iloc[row["idxs"][_lab]]["Label"]
                new_df.at[i, "idxs"] = row["idxs"][_lab]

        self.df_unlabeled["Label"] = new_df["Label"]


class CentroidRetrieval(ImageRetrieval):
    def __init__(
        self, df_small, feat_small, df_unlabeled, feat_unlabeled, k=5, metric="cosine"
    ):
        super().__init__(df_small, feat_small, df_unlabeled, feat_unlabeled)
        self.k = k
        self.metric = metric

    def retrieve_images(self):
        """Retrieve the k nearest images to the query images"""
        local_df = self.df_unlabeled.copy()

        local_df["Candidates"] = [[] for _ in range(len(self.df_unlabeled))]
        local_df["Distances"] = [[] for _ in range(len(self.df_unlabeled))]

        neigh = NearestNeighbors(n_neighbors=self.k, n_jobs=-1, metric=self.metric)
        neigh.fit(self.feat_unlabeled)
        range_labels = self.df_small['Label'].nunique()

        for i in tqdm(range(range_labels)):
            indices = self.df_small[self.df_small['Label'] == i].index

            if len(self.feat_small) == 1:
                current_features = self.feat_small
            else:
                current_features = self.feat_small[indices]
            current_label = i
            centroid = np.mean(current_features, axis=0)

            distances, indices = neigh.kneighbors([centroid])

            for j in range(self.k):
                idx = indices[0][j]
                local_df.loc[idx, "Candidates"].append(current_label)
                local_df.loc[idx, "Distances"].append(distances[0][j])

        for index, row in local_df.iterrows():
            if row["Candidates"]:
                idx = np.argmin(row["Distances"])
                local_df.at[index, "Label"] = row["Candidates"][idx]

        self.df_unlabeled = local_df[["Image", "Label"]]


class ClosestToLabelSetRetrieval(ImageRetrieval):
    def __init__(
        self, df_small, feat_small, df_unlabeled, feat_unlabeled, k=5, metric="cosine"
    ):
        super().__init__(df_small, feat_small, df_unlabeled, feat_unlabeled)
        self.k = k
        self.metric = metric

    def retrieve_images(self):
        """Retrieve the k nearest images to the query images"""

        neigh = NearestNeighbors(n_neighbors=self.k, n_jobs=-1, metric=self.metric)
        neigh.fit(self.feat_small)

        distances, indices = neigh.kneighbors(self.feat_unlabeled)

        self.df_unlabeled["Label"] = self._augment_dataset(distances, indices)

    def _augment_dataset(self, distances, indices):
        """Augment the dataset with the query images"""
        new_labels = np.full(indices.shape[0], -1, dtype=np.int32)

        labels = []
        query_labels = self.df_small["Label"].values
        for idx in indices:
            labels.append(query_labels[idx])

        medians = np.full(251, -1, dtype=np.float32)

        seen = set()
        for i, lab in tqdm(enumerate(labels)):
            _class = mode(lab)

            idxs = np.where(query_labels == _class)[0]
            class_feat = self.feat_small[idxs]

            if medians[_class] == -1:
                medians[_class] = np.median(
                    sp.spatial.distance.pdist(class_feat, metric=self.metric)
                )
            if np.count_nonzero(lab == _class) >= 3:
                positions = [i for i, x in enumerate(lab) if x == _class]
                d = distances[positions]

                if np.any(d <= medians[_class]):
                    new_labels[i] = _class
                    seen.add(i)
        return new_labels
