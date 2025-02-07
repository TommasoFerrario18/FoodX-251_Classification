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

        # Image enhancement
        enhancer = ImageEnhancer()

        if blur_score < 50:
            out_image = enhancer.enhance_image(image)
        elif blur_score > 1000:
            # out_image = cv2.medianBlur(image, 3)
            out_image = image
            for i in range(4):
                sigma_spatial, sigma_range = enhancer.estimate_bilateral_params(
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
            out_image = enhancer.neural_enhance(image)

        # Convert color space
        # cv2.cvtColor(out_image, cv2.COLOR_BGR2RGB)
        return out_image

    def extract_features(self, image):
        """Feature extraction from image"""
        if image is None:
            ValueError("Image is None")

        if isinstance(image, np.ndarray):
            image = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0

        return self.feature_extractor.compute_features_single_image(image)
