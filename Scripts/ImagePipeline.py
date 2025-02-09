import os
import cv2
import torch
import numpy as np

from ImageEnhancer import ImageEnhancer


class ImagePipeline:
    def __init__(self, model, feature_extractor, preprocessing=True):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model
        self.feature_extractor = feature_extractor
        self.preprocessing = preprocessing
        self.brisque_path = os.path.join(
            "..", "Scripts", "Quality", "brisque_model_live.yml"
        )
        self.brisque_range_path = os.path.join(
            "..", "Scripts", "Quality", "brisque_range_live.yml"
        )
        self.brisque_model = cv2.quality.QualityBRISQUE_create(
            self.brisque_path, self.brisque_range_path
        )

        # Image enhancement
        self.enhancer = ImageEnhancer()

    def predict(self, image, brisque_threshold=(20, 70)):
        """Predict image label"""
        if self.preprocessing:
            image = self.preprocess(image, brisque_threshold)
            if image is None:
                return None, -1

        features = self.extract_features(image)

        with torch.no_grad():
            if isinstance(self.model, torch.nn.Module):
                return features, self.model(features.to(self.device)).argmax().item()

        return features, self.model.predict(features)

    def preprocess(self, image, brisque_threshold):
        """Image preprocessing"""
        # BRISQUE evaluation
        score = self.brisque_model.compute(image)[0]

        if score < brisque_threshold[0]:
            return image

        if score > brisque_threshold[1]:
            return None

        # Image to clean
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blur_score = cv2.Laplacian(gray, cv2.CV_64F).var()

        if blur_score < 50:
            out_image = self.enhancer.enhance_image(image)
        elif blur_score > 1000:
            out_image = image
            for _ in range(4):
                sigma_spatial, sigma_range = self.enhancer.estimate_bilateral_params(
                    out_image
                )
                out_image = cv2.bilateralFilter(
                    out_image, 9, sigmaColor=sigma_range, sigmaSpace=sigma_spatial
                )

                blur_score = cv2.Laplacian(
                    cv2.cvtColor(out_image, cv2.COLOR_BGR2GRAY), cv2.CV_64F
                ).var()
                if blur_score < 1300:
                    break
        else:
            out_image = self.enhancer.neural_enhance(image)

        # Final check and conversion
        if out_image is not None and out_image.dtype != np.uint8:
            out_image = np.clip(out_image * 255, 0, 255).astype(np.uint8)

        return out_image

    def extract_features(self, image):
        """Feature extraction from image"""
        if image is None:
            ValueError("Image is None")

        if isinstance(image, np.ndarray):
            image = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0

        return self.feature_extractor.compute_features_single_image(image)
