import os
import cv2
import torch
import numpy as np
import torch_directml

from network_rrdbnet import RRDBNet as net


class ImageEnhancer:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # self.device = torch_directml.device()
        self.model_path = os.path.join("..", "Scripts", "Models", "BSRGANx2.pth")
        self.model = net(in_nc=3, out_nc=3, nf=64, nb=23, gc=32, sf=2).to(self.device)
        self.model.load_state_dict(
            torch.load(self.model_path, weights_only=True), strict=True
        )

    def estimate_bilateral_params(self, image):
        height, width = image.shape[:2]
        diagonal = np.sqrt(height**2 + width**2)

        # Spatial sigma: 3% of the image diagonal
        sigma_spatial = diagonal * 0.03

        # Range sigma: estimate based on image statistics
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image

        # Estimate noise using local standard deviation
        kernel_size = 3
        pad_size = kernel_size // 2
        padded = cv2.copyMakeBorder(
            gray, pad_size, pad_size, pad_size, pad_size, cv2.BORDER_REFLECT
        )
        local_std = np.zeros_like(gray, dtype=np.float32)

        # Calculate local standard deviation
        for i in range(pad_size, padded.shape[0] - pad_size):
            for j in range(pad_size, padded.shape[1] - pad_size):
                window = padded[
                    i - pad_size : i + pad_size + 1, j - pad_size : j + pad_size + 1
                ]
                local_std[i - pad_size, j - pad_size] = np.std(window)

        # Use median of local standard deviations * 2 for range sigma
        sigma_range = np.median(local_std) * 2.0

        return sigma_spatial, sigma_range

    def bilateral_sharpen(self, image):
        """
        Sharpens image using bilateral filter to preserve edges.
        """
        image_float = image.astype(np.float32) / 255.0

        sigma_spatial, sigma_range = self.estimate_bilateral_params(image)

        smooth = (
            cv2.bilateralFilter(
                image,
                d=5,
                sigmaColor=sigma_range,
                sigmaSpace=sigma_spatial,
            ).astype(np.float32)
            / 255.0
        )

        unsharp_mask = image_float - smooth

        amount = 2.5
        sharpened = image_float + (unsharp_mask * amount)

        return np.clip(sharpened * 255, 0, 255).astype(np.uint8)

    def adaptive_gamma_correction(self, image):
        image_float = image.astype(float) / 255.0
        brightness = np.mean(image_float)
        gamma = 1.0 + (1.0 - brightness)
        corrected = np.power(image_float, gamma)
        return (corrected * 255).astype(np.uint8)

    def enhance_image(self, image):
        if image.dtype != np.uint8:
            image = (image * 255).astype(np.uint8)

        gamma_corrected = self.adaptive_gamma_correction(image)
        sharpened = self.bilateral_sharpen(gamma_corrected)
        return sharpened

    def neural_enhance(self, image):
        self.model.eval()
        for k, v in self.model.named_parameters():
            v.requires_grad = False
        self.model.to(self.device)
        torch.cuda.empty_cache()

        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        input_image = self._uint2tensor4(image_rgb)

        with torch.no_grad():
            output_image = (
                self.model(input_image.to(self.device))
                .data.squeeze()
                .float()
                .cpu()
                .clamp_(0, 1)
                .numpy()
            )

        output_image = np.transpose(output_image, (1, 2, 0))
        output_image = cv2.cvtColor(
            (output_image * 255).astype(np.uint8), cv2.COLOR_RGB2BGR
        )
        return output_image

    def _uint2tensor4(self, img):
        if img.ndim == 2:
            img = np.expand_dims(img, axis=2)
        return (
            torch.from_numpy(np.ascontiguousarray(img))
            .permute(2, 0, 1)
            .float()
            .div(255.0)
            .unsqueeze(0)
        )
