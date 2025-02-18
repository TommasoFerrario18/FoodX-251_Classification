import os
import cv2
import torch
import numpy as np
import pandas as pd
from torchvision import transforms

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
            for i in range(2):
                out_image = self.enhancer.enhance_image(out_image)
        elif blur_score > 1000:
            out_image = image
            for _ in range(6):
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

    def format_dashboard_output(self, proba):
        """Format prediction for dashboard"""
        class_name = pd.read_csv("class_list.txt", header=None)
        top_5 = proba[0].argsort()[-5:][::-1]

        results = []
        for j in top_5:
            class_j = class_name.iloc[j, 0].split(" ")
            results.append((class_j[0], class_j[1], proba[0][j]))

        return results

    def predict_for_dashboard(
        self, image_path, preprocessing=False, brisque_threshold=(20, 70)
    ):
        """Predict image label for dashboard"""
        if not os.path.exists(image_path):
            return None, -1

        image = cv2.imread(image_path)

        if preprocessing:
            print("Preprocessing")
            image = self.preprocess(image, brisque_threshold)
            if image is None:
                return None, -1
            
        transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Resize(232, interpolation=transforms.InterpolationMode.BILINEAR),
                transforms.Pad(padding=(0, 0, 0, 0), fill=0),
                transforms.CenterCrop((232, 232)),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )

        image = transform(image)

        features = self.extract_features(image)
        # reshape numpyarray to 1,length
        features = features.reshape(1, -1)

        with torch.no_grad():
            if isinstance(self.model, torch.nn.Module):
                return self.format_dashboard_output(
                    self.model(features.to(self.device))
                )

        return self.format_dashboard_output(self.model.predict_proba(features))
