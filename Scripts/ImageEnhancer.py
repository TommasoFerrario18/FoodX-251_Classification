import os
import cv2
import torch
import numpy as np

from network_rrdbnet import RRDBNet as net


class ImageEnhancer:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_path = os.path.join("..", "Scripts", "Models", "BSRGANx2.pth")

        # Load the model
        self.model = net(in_nc=3, out_nc=3, nf=64, nb=23, gc=32, sf=2).to(self.device)
        self.model.load_state_dict(
            torch.load(self.model_path, weights_only=True), strict=True
        )
        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad = False

    def estimate_bilateral_params(self, image):
        """Estimate bilateral filter parameters using a faster approach."""
        height, width = image.shape[:2]
        diagonal = np.sqrt(height**2 + width**2)

        # Spatial sigma: 3% of the image diagonal
        sigma_spatial = diagonal * 0.03

        # Range sigma: estimate based on image statistics
        gray = (
            cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        )

        mean = cv2.boxFilter(gray.astype(np.float32), ddepth=-1, ksize=(3, 3))
        sq_mean = cv2.boxFilter(gray.astype(np.float32) ** 2, ddepth=-1, ksize=(3, 3))
        local_std = np.sqrt(np.abs(sq_mean - mean**2))

        # Use median of local standard deviations * 2 for range sigma
        sigma_range = np.median(local_std) * 2.0

        return sigma_spatial, sigma_range

    def bilateral_sharpen(self, image):
        """
        Sharpens image using bilateral filter to preserve edges.
        """
        image_float = image.astype(np.float32) / 255.0

        sigma_spatial, sigma_range = self.estimate_bilateral_params(image)

        smooth = cv2.bilateralFilter(
            image, d=5, sigmaColor=sigma_range, sigmaSpace=sigma_spatial
        )
        smooth = smooth.astype(np.float32) / 255.0

        # Vectorized unsharp mask
        sharpened = np.clip(image_float + (image_float - smooth) * 2.5, 0, 1)

        return (sharpened * 255).astype(np.uint8)

    def adaptive_gamma_correction(self, image):
        """Adjusts gamma dynamically based on brightness."""
        image_float = image.astype(np.float32) / 255.0
        brightness = np.mean(image_float)
        gamma = max(0.5, 2.0 - brightness)
        corrected = np.power(image_float, gamma)
        return (corrected * 255).astype(np.uint8)

    def enhance_image(self, image):
        image = self.adaptive_gamma_correction(image)
        return self.bilateral_sharpen(image)

    def neural_enhance(self, image):
        """Super-resolution using a neural network model."""
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        input_image = self._uint2tensor4(image_rgb).to(self.device)

        with torch.no_grad():
            output_image = self.model(input_image).squeeze().cpu().clamp_(0, 1).numpy()

        output_image = (np.transpose(output_image, (1, 2, 0)) * 255).astype(np.uint8)
        return cv2.cvtColor(output_image, cv2.COLOR_RGB2BGR)

    def _uint2tensor4(self, img):
        """Converts a NumPy image to a 4D PyTorch tensor."""
        return torch.from_numpy(img.transpose(2, 0, 1)).float().div(255.0).unsqueeze(0)
