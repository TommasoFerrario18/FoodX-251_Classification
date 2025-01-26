import cv2
import numpy as np

from tqdm import tqdm
from sklearn.cluster import KMeans


class BaseFeatureExtractor:
    def __init__(self, method, feat_step, imsize, k):
        self.method = method
        self.feat_step = feat_step
        self.imsize = imsize
        self.k = k

    def compute_features(self, image_paths, labels):
        point_positions = self._generate_grid(self.feat_step, self.imsize)

        features, feature_labels = self.extract_features_from_images(
            image_paths, labels, point_positions
        )

        kmeans = KMeans(n_clusters=self.k, max_iter=100, n_init=10)
        kmeans.fit(features)
        centroids = kmeans.cluster_centers_

        bow_features, bow_labels = self._create_bow(
            kmeans.labels_, feature_labels, image_paths, labels
        )

        return bow_features, bow_labels, centroids

    def _generate_grid(self, feat_step, imsize):
        height, width = imsize
        ii, jj = np.meshgrid(
            np.arange(feat_step, width - feat_step + 1, feat_step),
            np.arange(feat_step, height - feat_step + 1, feat_step),
        )
        return np.column_stack([ii.ravel(), jj.ravel()]).astype(np.float32)

    def extract_features_from_images(self, image_paths, labels, point_positions):
        all_features = []
        all_feature_labels = []

        # Select appropriate feature detector
        if self.method.lower() == "sift":
            detector = cv2.SIFT_create()
        elif self.method.lower() == "surf":
            detector = cv2.xfeatures2d.SURF_create()
        else:
            raise ValueError("Method must be 'sift' or 'surf'")

        for idx, (image_path, label) in tqdm(
            enumerate(zip(image_paths, labels)), total=len(image_paths)
        ):
            # Read and preprocess image
            img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            img = cv2.resize(img, self.imsize)

            # Extract keypoints at predefined grid points
            keypoints = [
                cv2.KeyPoint(float(x), float(y), 1.0) for x, y in point_positions
            ]
            _, features = detector.compute(img, keypoints)

            if features is not None:
                all_features.append(features)
                # Create labels: image label and image index
                feature_labels = np.column_stack(
                    [np.full(features.shape[0], label), np.full(features.shape[0], idx)]
                )
                all_feature_labels.append(feature_labels)

        return np.vstack(all_features), np.vstack(all_feature_labels)

    def _create_bow(self, cluster_assignments, feature_labels, image_paths, labels):
        bow_features = np.zeros((len(image_paths), self.k))
        bow_labels = np.array(labels)

        for i in range(len(image_paths)):
            # Find features for this specific image
            image_feature_mask = feature_labels[:, 1] == i
            image_features_clusters = cluster_assignments[image_feature_mask]

            # Compute histogram
            unique, counts = np.unique(image_features_clusters, return_counts=True)
            bow_features[i, unique] = counts / len(image_features_clusters)

        return bow_features, bow_labels

    def compute_feature_sift_or_surf_test(self, image_paths, labels, centroids):
        point_positions = self._generate_grid(self.feat_step, self.imsize)

        print("Feature extraction")
        features, labels = self._extract_features_for_test(
            image_paths, labels, point_positions, centroids
        )

        return features, labels

    def _extract_features_for_test(
        self, image_paths, labels, point_positions, centroids
    ):
        # Select appropriate feature detector
        if self.method.lower() == "sift":
            detector = cv2.SIFT_create()
        elif self.method.lower() == "surf":
            detector = cv2.xfeatures2d.SURF_create()
        else:
            raise ValueError("Method must be 'sift' or 'surf'")

        bow_features = []
        bow_labels = []

        for idx, (image_path, label) in tqdm(
            enumerate(zip(image_paths, labels)), total=len(image_paths)
        ):
            # Read and preprocess image
            img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            img = cv2.resize(img, self.imsize)

            # Extract keypoints at predefined grid points
            keypoints = [cv2.KeyPoint(x, y, 1) for x, y in point_positions]
            _, features = detector.compute(img, keypoints)

            if features is not None:
                # Assign features to nearest cluster centroids
                distances = np.sum(
                    (features[:, np.newaxis, :] - centroids[np.newaxis, :, :]) ** 2,
                    axis=2,
                )
                words = np.argmin(distances, axis=1)

                # Create histogram of visual words
                H = np.bincount(words, minlength=k) / len(words)

                bow_features.append(H)
                bow_labels.append(label)

        return np.array(bow_features), np.array(bow_labels)
