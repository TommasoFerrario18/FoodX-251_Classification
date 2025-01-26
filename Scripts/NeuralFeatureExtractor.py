import os
import torch
import numpy as np

from torchvision.models import mobilenet_v3_large, alexnet, resnet50

class NeuralFeatureExtractor:
    def __init__(self, model, layer, input_size, result_file=None):
        # Model Information
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self._prepare_model(model, layer)

        # Result Information
        self.result_file = result_file
        self.result_dir = "../Features"

        if not os.path.exists(self.result_dir):
            os.makedirs(self.result_dir)
            os.makedirs(os.path.join(self.result_dir, "features"))
            os.makedirs(os.path.join(self.result_dir, "labels"))

        # Input Size
        self.input_size = input_size

    def _prepare_model(self, model, layer):
        """Prepare the model by removing the layers after the specified layer"""
        new_model_layers = []
        for name, module in model.named_children():
            if name == layer:
                break
            new_model_layers.append(module)

        new_model = torch.nn.Sequential(*new_model_layers)
        return new_model.to(self.device)

    def compute_features(self, imageDataloder):
        """Compute the features for the images in the imageDataloder"""
        self.model.eval()

        features, labels = [], []
        with torch.no_grad():
            for batch, (x, y) in enumerate(imageDataloder):
                print(f"Batch {batch}")
                x = x.to(self.device)
                outputs = self.model(x)
                features.append(outputs.cpu().numpy())
                labels.append(y.cpu().numpy())

        # Flatten the features and labels
        final_feat = np.vstack(
            [x.reshape(1, -1) for feat_group in features for x in feat_group]
        )
        final_lab = [
            label for _, lab_group in zip(features, labels) for label in lab_group
        ]

        if self.result_file is not None:
            self._save_features(final_feat, final_lab)

        return final_feat, final_lab

    def _save_features(self, features, labels):
        """Save the features and labels to the result directory"""
        np.savetxt(
            os.path.join(self.result_dir, "features", self.result_file), features
        )
        np.savetxt(os.path.join(self.result_dir, "labels", self.result_file), labels)


class MobileNetFeatureExtractor(NeuralFeatureExtractor):
    def __init__(self, result_file=None):
        super().__init__(
            mobilenet_v3_large(weights="IMAGENET1K_V2"),
            "avgpool",
            (232, 232),
            result_file,
        )

class AlexNetFeatureExtractor(NeuralFeatureExtractor):
    def __init__(self, result_file=None):
        super().__init__(
            alexnet(weights="IMAGENET1K_V1"),
            "avgpool",
            (256, 256),
            result_file,
        )

class ResNetFeatureExtractor(NeuralFeatureExtractor):
    def __init__(self, result_file=None):
        super().__init__(
            resnet50(weights="IMAGENET1K_V2:"),
            "avgpool",
            (232, 232),
            result_file,
        )