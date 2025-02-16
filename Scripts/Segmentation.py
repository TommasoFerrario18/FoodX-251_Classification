import cv2
import numpy as np
import matplotlib.pyplot as plt

from scipy import ndimage
from sklearn.cluster import KMeans
from skimage.measure import regionprops
from sklearn.preprocessing import StandardScaler
from skimage.feature import local_binary_pattern
from concurrent.futures import ThreadPoolExecutor


class ImageSegmentation:
    def __init__(self):
        self.num_classes = 2

    def compute_superpixels(self, image, region_size=20, ruler=15, num_iterations=100):
        """
        Compute superpixels for the given image
        """
        slic = cv2.ximgproc.createSuperpixelSLIC(
            image, algorithm=cv2.ximgproc.SLICO, region_size=region_size, ruler=ruler
        )
        slic.iterate(num_iterations)

        return slic.getLabels(), slic.getNumberOfSuperpixels()

    def process_superpixel(self, args):
        """
        Process a single superpixel - for parallel processing
        """
        i, image_rgb, Y, lbp_image, mask, points = args

        # Compute LBP histogram
        lbp_values = lbp_image[mask]
        hist_lbp, _ = np.histogram(
            lbp_values, bins=np.arange(0, points + 3), density=True
        )

        # Compute color features using OpenCV's optimized functions
        mean, stddev = cv2.meanStdDev(image_rgb, mask=mask.astype(np.uint8))
        mean_rgb = mean.flatten()[:3]
        std_rgb = stddev.flatten()[:3]

        return np.concatenate([hist_lbp, mean_rgb, std_rgb])

    def extract_features(self, image, superpixel_labels, num_superpixel):
        """
        Extract features from the image
        """
        Y = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)[:, :, 0]
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        radius = 1
        points = 8 * radius
        lbp_image = local_binary_pattern(Y, points, radius, method="uniform")

        args_list = [
            (i, image_rgb, Y, lbp_image, superpixel_labels == i, points)
            for i in range(num_superpixel)
        ]

        with ThreadPoolExecutor() as executor:
            features = list(executor.map(self.process_superpixel, args_list))

        features = np.array(features)

        scaler = StandardScaler()
        return scaler.fit_transform(features)

    def segment_image(self, image):
        """
        Segment the given image
        """
        if isinstance(image, str):
            image = cv2.imread(image)

        if image is None:
            raise ValueError("Could not read the image")

        superpixel_labels, num_superpixel = self.compute_superpixels(image)
        features = self.extract_features(image, superpixel_labels, num_superpixel)

        kmeans = KMeans(n_clusters=self.num_classes, random_state=42)
        cluster_labels = kmeans.fit_predict(features)

        # Create segmented image
        segmented = np.zeros_like(image)
        labeled_image = np.zeros(image.shape[:2], dtype=np.uint8)

        # Vectorized assignment for foreground
        foreground_mask = cluster_labels == 1
        background_mask = ~foreground_mask

        for i in range(num_superpixel):
            mask = superpixel_labels == i
            if cluster_labels[i] == 1:
                segmented[mask] = image[mask]
                labeled_image[mask] = 255
            else:
                segmented[mask] = [0, 0, 0]

        foreground_mask = self.get_foreground_mask(image, labeled_image)

        self.visualize_segmentation(image, segmented, foreground_mask)

        return segmented, foreground_mask

    def visualize_segmentation(self, image, segmented, labeled_image):
        """
        Visualize the segmented image
        """
        plt.figure(figsize=(10, 10))
        plt.subplot(131)
        plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        plt.title("Original Image")
        plt.axis("off")

        plt.subplot(132)
        plt.imshow(cv2.cvtColor(segmented, cv2.COLOR_BGR2RGB))
        plt.title("Segmented Image")
        plt.axis("off")

        plt.subplot(133)
        plt.imshow(labeled_image, cmap="gray")
        plt.title("Labeled Image")
        plt.axis("off")

        plt.tight_layout()
        plt.show()

    def calculate_centroid(self, mask):
        """
        Calculate the centroid of a binary mask.
        """
        props = regionprops(mask.astype(int))[0]
        return props.centroid

    def check_border_pixels(self, mask, border_width=1):
        """
        Check if the mask touches the image border.
        """
        h, w = mask.shape
        border_mask = np.zeros_like(mask)
        border_mask[:border_width, :] = 1  # Top
        border_mask[-border_width:, :] = 1  # Bottom
        border_mask[:, :border_width] = 1  # Left
        border_mask[:, -border_width:] = 1  # Right

        return np.any(mask & border_mask)

    def calculate_texture_variance(self, image, mask):
        """
        Calculate texture variance using LBP for the masked region.
        """
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = np.mean(image, axis=2)
        else:
            gray = image

        if image.dtype != np.uint8:
            gray = (gray * 255).astype(np.uint8)

        radius = 3
        n_points = 8 * radius
        lbp = local_binary_pattern(gray, n_points, radius, method="uniform")

        # Calculate variance of LBP values in masked region
        masked_lbp = lbp[mask]
        if len(masked_lbp) == 0:
            return 0
        return np.var(masked_lbp)

    def compute_weighted_score(
        self, distance_from_center, touches_border, relative_area, texture_variance
    ):
        """
        Compute weighted score for foreground probability.
        """
        max_distance = np.sqrt(2)
        normalized_distance = 1 - (distance_from_center / max_distance)

        weights = {"distance": 0.3, "border": 0.2, "area": 0.2, "texture": 0.3}

        distance_score = normalized_distance * weights["distance"]
        border_score = (1 - float(touches_border)) * weights["border"]

        area_score = (1 - abs(relative_area - 0.3)) * weights["area"]

        normalized_texture = min(texture_variance / 1000, 1)
        texture_score = normalized_texture * weights["texture"]

        return distance_score + border_score + area_score + texture_score

    def classify_clusters(self, image, labels):
        """
        Classify image clusters into foreground and background.

        """
        h, w = labels.shape
        image_center = (h / 2, w / 2)
        scores = []

        for label in np.unique(labels):
            mask = labels == label

            # Calculate centroid and its distance from center
            centroid = self.calculate_centroid(mask)
            distance_from_center = np.sqrt(
                (centroid[0] - image_center[0]) ** 2
                + (centroid[1] - image_center[1]) ** 2
            )

            touches_border = self.check_border_pixels(mask)
            relative_area = np.sum(mask) / mask.size
            texture_variance = self.calculate_texture_variance(image, mask)

            score = self.compute_weighted_score(
                distance_from_center, touches_border, relative_area, texture_variance
            )

            scores.append((label, score))

        return max(scores, key=lambda x: x[1])[0]

    def get_foreground_mask(self, image, labels):
        """
        Get foreground mask from image and cluster labels.
        """
        foreground_label = self.classify_clusters(image, labels)
        return labels == foreground_label
