import cv2
import numpy as np
import concurrent.futures

from tqdm import tqdm
from typing import List, Tuple
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from skimage.feature import local_binary_pattern


class BaseFeatureExtractor:
    def __init__(self, method: str, feat_step: int, imsize: Tuple[int, int], k: int):
        self.method = method.lower()
        self.feat_step = feat_step
        self.imsize = imsize
        self.k = k
        self.detector = self._create_detector()
        self.point_positions = self._generate_grid()
        self.scaler = StandardScaler()

    def _create_detector(self):
        """Create and return the appropriate feature detector."""
        if self.method == "sift":
            return cv2.SIFT_create()
        elif self.method == "surf":
            return cv2.xfeatures2d.SURF_create()
        else:
            raise ValueError("Method must be 'sift' or 'surf'")

    def _generate_grid(self):
        height, width = self.imsize
        ii, jj = np.meshgrid(
            np.arange(self.feat_step, width - self.feat_step + 1, self.feat_step),
            np.arange(self.feat_step, height - self.feat_step + 1, self.feat_step),
        )
        return np.column_stack([ii.ravel(), jj.ravel()]).astype(np.float32)

    def _process_single_image(self, image_data: Tuple[str, int]):
        """Process a single image and extract features."""
        image_path, label = image_data
        img = cv2.imread(image_path)
        if img is None:
            return None, None

        img = cv2.resize(img, self.imsize[::-1])

        keypoints = [
            cv2.KeyPoint(float(x), float(y), 1.0) for x, y in self.point_positions
        ]
        _, features = self.detector.compute(img, keypoints)

        if features is None:
            return None, None

        feature_labels = np.column_stack(
            [
                np.full(features.shape[0], label),
                np.full(features.shape[0], hash(image_path)),
            ]
        )
        return features, feature_labels

    def compute_features(
        self, image_paths: List[str], labels: List[int]
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Compute BoW features for all images using parallel processing.
        """
        with concurrent.futures.ThreadPoolExecutor() as executor:
            results = list(
                tqdm(
                    executor.map(self._process_single_image, zip(image_paths, labels)),
                    total=len(image_paths),
                    desc="Extracting features",
                )
            )

        valid_results = [(f, l) for f, l in results if f is not None]
        if not valid_results:
            raise ValueError("No valid features extracted from images")

        features, feature_labels = zip(*valid_results)
        features = np.vstack(features)
        feature_labels = np.vstack(feature_labels)

        features_scaled = self.scaler.fit_transform(features)

        kmeans = KMeans(
            n_clusters=self.k,
            max_iter=100,
            init="k-means++",
            n_init=10,
            algorithm="elkan",
            random_state=42
        )

        print("Clustering features...")
        kmeans.fit(features_scaled)

        bow_features, bow_labels = self._create_bow(
            kmeans.labels_, feature_labels, image_paths, labels
        )

        return bow_features, bow_labels, kmeans.cluster_centers_

    def _create_bow(
        self,
        cluster_assignments: np.ndarray,
        feature_labels: np.ndarray,
        image_paths: List[str],
        labels: List[int],
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Create Bag of Words features using vectorized operations."""
        print("Creating Bag of Words features...")
        bow_features = np.zeros((len(image_paths), self.k))

        # Create a mapping from image hash to index
        unique_images = np.unique(feature_labels[:, 1])
        image_to_idx = {hash(path): idx for idx, path in enumerate(image_paths)}

        # Vectorized operation to create BoW features
        for img_hash in unique_images:
            mask = feature_labels[:, 1] == img_hash
            if not np.any(mask):
                continue

            clusters = cluster_assignments[mask]
            hist, _ = np.histogram(clusters, bins=range(self.k + 1))
            idx = image_to_idx[int(img_hash)]
            bow_features[idx] = hist / len(clusters)

        return bow_features, np.array(labels)

    def extract_features_from_images(self, image_paths, labels):
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
                cv2.KeyPoint(float(x), float(y), 1.0) for x, y in self.point_positions
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

    def compute_feature_sift_or_surf_test(self, image_paths, labels, centroids):
        print("Feature extraction")
        features, labels = self._extract_features_for_test(
            image_paths, labels, centroids
        )

        return features, labels

    def _extract_features_for_test(self, image_paths, labels, centroids):
        bow_features = []
        bow_labels = []

        for idx, (image_path, label) in tqdm(
            enumerate(zip(image_paths, labels)), total=len(image_paths)
        ):
            # Read and preprocess image
            img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            img = cv2.resize(img, self.imsize)

            # Extract keypoints at predefined grid points
            keypoints = [cv2.KeyPoint(x, y, 1) for x, y in self.point_positions]
            _, features = self.detector.compute(img, keypoints)

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

    def compute_color_lbp_features(self, datastore) -> Tuple[np.ndarray, np.ndarray]:
        """Compute color and LBP features for images."""
        features, labels = [], []

        def process_single_item(item):
            image, label = item
            image = image.permute(1, 2, 0).numpy() * 255
            image = image.astype(np.uint8)

            Y = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)[:, :, 0]
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            points = 8
            lbp = local_binary_pattern(Y, points, 1, "uniform")
            hist_lbp, _ = np.histogram(lbp, bins=np.arange(points + 3), density=True)

            mean, stddev = cv2.meanStdDev(image_rgb)
            mean_rgb = mean.flatten()[:3]
            std_rgb = stddev.flatten()[:3]

            return np.concatenate([hist_lbp, mean_rgb, std_rgb]), label

        with concurrent.futures.ThreadPoolExecutor() as executor:
            results = list(
                tqdm(
                    executor.map(process_single_item, datastore),
                    total=len(datastore),
                    desc="Computing color/LBP features",
                )
            )

        features, labels = zip(*results)
        return np.array(features), np.array(labels)
