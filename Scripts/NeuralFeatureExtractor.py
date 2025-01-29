import os
import torch
import numpy as np
from typing import Any, List, Tuple

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
        self.features = None
        self.hooks = []

        # Result Information
        self.result_file = result_file
        self.result_dir = "../Features"

        if not os.path.exists(self.result_dir):
            os.makedirs(self.result_dir)
            os.makedirs(os.path.join(self.result_dir, "features"))
            os.makedirs(os.path.join(self.result_dir, "labels"))

        # Input Size
        self.input_size = input_size

        # Prepare model
        self.model = self._prepare_model(model)

    def _get_layer(self, model: torch.nn.Module, layer_name: str) -> torch.nn.Module:
        """Get a layer from model given its name"""
        if "." in layer_name:
            layers = layer_name.split(".")
            current = model
            for layer in layers:
                if hasattr(current, layer):
                    current = getattr(current, layer)
                else:
                    raise ValueError(f"Layer {layer_name} not found in model")
            return current
        else:
            if hasattr(model, layer_name):
                return getattr(model, layer_name)
            raise ValueError(f"Layer {layer_name} not found in model")

    def _hook_fn(self, module: torch.nn.Module, input: Any, output: Any) -> None:
        """Hook function to capture intermediate features"""
        self.features = output.detach()

    def _prepare_model(self, model: torch.nn.Module) -> torch.nn.Module:
        """Prepare the model by adding hooks to the target layer"""
        model = model.to(self.device)
        model.eval()

        # Remove any existing hooks
        for hook in self.hooks:
            hook.remove()
        self.hooks = []

        # Add new hook
        target = self._get_layer(model, self.target_layer)
        hook = target.register_forward_hook(self._hook_fn)
        self.hooks.append(hook)

        return model

    def compute_features(
        self, imageDataloader: torch.utils.data.DataLoader
    ) -> Tuple[np.ndarray, List]:
        """Compute the features for the images in the imageDataloader"""
        features, labels = [], []

        with torch.no_grad():
            for batch, (x, y) in enumerate(imageDataloader):
                print(f"Batch {batch + 1} / {len(imageDataloader)}", end="\r")
                x = x.to(self.device)
                _ = self.model(x)  # Forward pass

                if self.features is None:
                    raise ValueError(
                        "No features were captured. Check if the target layer name is correct."
                    )

                # Reshape features to (batch_size, -1)
                batch_features = self.features.reshape(self.features.size(0), -1)
                features.append(batch_features.cpu().numpy())
                labels.append(y.cpu().numpy())

                self.features = None  # Reset features for next batch

        final_feat = np.vstack(features)
        final_lab = np.concatenate(labels)

        if self.result_file is not None:
            self._save_features(final_feat, final_lab)

        return final_feat, final_lab

    def _save_features(self, features: np.ndarray, labels: List) -> None:
        """Save the features and labels to the result directory"""
        np.save(
            os.path.join(self.result_dir, "features", self.result_file),
            features,
            allow_pickle=False,
        )
        np.save(
            os.path.join(self.result_dir, "labels", self.result_file),
            labels,
            allow_pickle=False,
        )

    def __del__(self):
        """Clean up hooks when the object is deleted"""
        for hook in self.hooks:
            hook.remove()


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
