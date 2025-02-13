import cv2
import numpy as np
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
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
                segmented[mask] = [128, 128, 128]

        self.visualize_segmentation(image, segmented, labeled_image)

        return segmented, labeled_image

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
