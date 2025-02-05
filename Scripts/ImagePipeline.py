import cv2
import torch
import numpy as np


class ImagePipeline:
    def __init__(self, model, feature_extractor, preprocessing=True):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model
        self.feature_extractor = feature_extractor
        self.preprocessing = preprocessing

    def predict(self, image, brisque_threshold=(20, 70)):
        """Predict image label"""
        if self.preprocessing:
            image = self.preprocess(image, brisque_threshold)

        if image == -1:
            return -1

        features = self.extract_features(image)

        if isinstance(self.model, torch.nn.Module):
            return self.model(features.to(self.device)).argmax().item()

        return self.model.predict(features)

    def preprocess(self, image, brisque_threshold):
        """Image preprocessing"""
        # BRISQUE evaluation
        score = cv2.quality.QualityBRISQUE_compute(
            image,
            "./../Scripts/Quality/brisque_model_live.yml",
            "./../Scripts/Quality/brisque_range_live.yml",
        )[0]

        if score < brisque_threshold[0]:
            return image

        if score > brisque_threshold[1]:
            return -1

        # Image to clean
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blur_score = cv2.Laplacian(gray, cv2.CV_64F).var()

        # Initialize sharpened as the original image
        sharpened = image.copy()

        if blur_score < 50:
            sharpen_kernel = np.array(
                [[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]]
            )  # Modified kernel
            # Apply the kernel to the image
            sharpened = cv2.filter2D(image, -1, sharpen_kernel)
            # Ensure pixel values stay in valid range
            sharpened = np.clip(sharpened, 0, 255).astype(np.uint8)
        else:
            sharpened = image.copy()

        x = cv2.quality.QualityBRISQUE_compute(
            sharpened,
            "./../Scripts/Quality/brisque_model_live.yml",
            "./../Scripts/Quality/brisque_range_live.yml",
        )[0]
        print(f"BRISQUE: {score}, Laplacian: {blur_score}")
        print(f"Image sharpened {x}")

        # Convert color space
        processed_image = cv2.cvtColor(sharpened, cv2.COLOR_BGR2RGB)
        return processed_image

    def extract_features(self, image):
        """Feature extraction from image"""
        if image is None:
            ValueError("Image is None")

        if isinstance(image, np.ndarray):
            image = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0

        return self.feature_extractor.compute_features_single_image(image)
