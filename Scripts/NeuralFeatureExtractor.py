import os
import torch
import numpy as np

from tqdm import tqdm
from typing import Any, Tuple
from torchvision.models import mobilenet_v3_large, alexnet, resnet50


class NeuralFeatureExtractor:
    def __init__(
        self,
        model: torch.nn.Module,
        target_layer: str,
        input_size: Tuple[int, int],
        result_file: str = None,
    ):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.target_layer = target_layer
        self.features = None  # Stores the extracted features

        # Result Directory Setup
        self.result_file = result_file
        self.result_dir = "../Features"
        os.makedirs(os.path.join(self.result_dir, "features"), exist_ok=True)
        os.makedirs(os.path.join(self.result_dir, "labels"), exist_ok=True)

        self.input_size = input_size
        self.model = self._prepare_model(model)  # Prepare and hook model

    def _get_layer(self, model: torch.nn.Module, layer_name: str) -> torch.nn.Module:
        """Retrieve a layer from the model using its name"""
        try:
            for layer in layer_name.split("."):
                model = getattr(model, layer)
            return model
        except AttributeError:
            raise ValueError(f"Layer {layer_name} not found in model")

    def _hook_fn(self, module: torch.nn.Module, input: Any, output: Any) -> None:
        """Capture intermediate features from the target layer"""
        self.features = output.detach()

    def _prepare_model(self, model: torch.nn.Module) -> torch.nn.Module:
        """Register hook to capture features from the target layer"""
        model.to(self.device).eval()

        # Register forward hook once
        target_layer = self._get_layer(model, self.target_layer)
        target_layer.register_forward_hook(self._hook_fn)

        return model

    def compute_features(
        self, dataloader: torch.utils.data.DataLoader
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Compute features for all images in a DataLoader"""
        features_list, labels_list = [], []

        with torch.no_grad():
            for batch, (x, y) in tqdm(enumerate(dataloader), total=len(dataloader)):
                x = x.to(self.device, non_blocking=True)  # Use non-blocking transfer
                _ = self.model(x)  # Forward pass

                if self.features is None:
                    raise RuntimeError("No features captured. Check target layer name.")

                features_list.append(
                    self.features.view(self.features.size(0), -1).cpu()
                )
                labels_list.append(y.cpu())

        final_feat = torch.cat(features_list).numpy()  # Convert tensors at the end
        final_lab = torch.cat(labels_list).numpy()

        if self.result_file:
            self._save_features(final_feat, final_lab)

        return final_feat, final_lab

    def compute_features_single_image(self, image: torch.Tensor) -> np.ndarray:
        """Extract features from a single image"""
        with torch.no_grad():
            image = image.unsqueeze(0).to(self.device, non_blocking=True)
            _ = self.model(image)

            if self.features is None:
                raise RuntimeError("No features captured. Check target layer name.")

            return self.features.view(-1).cpu().numpy()

    def _save_features(self, features: np.ndarray, labels: np.ndarray) -> None:
        """Save extracted features and labels"""
        np.save(os.path.join(self.result_dir, "features", self.result_file), features)
        np.save(os.path.join(self.result_dir, "labels", self.result_file), labels)


class MobileNetFeatureExtractor(NeuralFeatureExtractor):
    def __init__(self, target_layer: str = "classifier.0", result_file: str = None):
        model = mobilenet_v3_large(weights="IMAGENET1K_V2")
        super().__init__(
            model,
            target_layer,
            (232, 232),
            result_file,
        )


class AlexNetFeatureExtractor(NeuralFeatureExtractor):
    def __init__(self, target_layer: str = "classifier.2", result_file: str = None):
        model = alexnet(weights="IMAGENET1K_V1")
        super().__init__(
            model,
            target_layer,
            (256, 256),
            result_file,
        )


class ResNetFeatureExtractor(NeuralFeatureExtractor):
    def __init__(self, target_layer: str = "avgpool", result_file: str = None):
        model = resnet50(weights="IMAGENET1K_V2")
        super().__init__(
            model,
            target_layer,
            (232, 232),
            result_file,
        )
