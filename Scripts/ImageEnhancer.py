import os
import cv2
import torch
import numpy as np

from network_rrdbnet import RRDBNet as net


class ImageEnhancer:
    def denoise_image(self, image):
        # Non-local means denoising
        denoised = cv2.fastNlMeansDenoisingColored(image, None, 10, 10, 7, 21)
        return denoised

    def estimate_bilateral_params(self, image):
        height, width = image.shape[:2]
        diagonal = np.sqrt(height**2 + width**2)

        # Spatial sigma: 2% of the image diagonal
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
        # Convert to float32 for calculations
        image_float = image.astype(np.float32) / 255.0

        # Get estimated parameters
        sigma_spatial, sigma_range = self.estimate_bilateral_params(image)

        # Apply bilateral filter with estimated parameters
        smooth = (
            cv2.bilateralFilter(
                image,
                d=5,  # Small window size for fine detail preservation
                sigmaColor=sigma_range,
                sigmaSpace=sigma_spatial,
            ).astype(np.float32)
            / 255.0
        )

        # Calculate unsharp mask (original - smoothed)
        unsharp_mask = image_float - smooth

        # Add scaled mask back to original
        amount = 1.5  # Sharpening strength (adjust as needed)
        sharpened = image_float + (unsharp_mask * amount)

        # Ensure results stay in valid range
        return np.clip(sharpened * 255, 0, 255).astype(np.uint8)

    def adaptive_gamma_correction(self, image):
        image_float = image.astype(float) / 255.0
        brightness = np.mean(image_float)
        gamma = 1.0 + (1.0 - brightness)
        corrected = np.power(image_float, gamma)
        return (corrected * 255).astype(np.uint8)

    def enhance_image(self, image):
        # Optional denoising step
        # denoised = self.denoise_image(image)

        # Apply gamma correction first
        gamma_corrected = self.adaptive_gamma_correction(image)

        # Apply bilateral sharpening
        sharpened = self.bilateral_sharpen(gamma_corrected)
        return sharpened

    def neural_enhance(self, image):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model_path = os.path.join("..", "Scripts", "Models", "BSRGANx2.pth")
        model = net(in_nc=3, out_nc=3, nf=64, nb=23, gc=32, sf=2).to(device)

        model.load_state_dict(torch.load(model_path, weights_only=True), strict=True)
        model.eval()
        for k, v in model.named_parameters():
            v.requires_grad = False
        model = model.to(device)
        torch.cuda.empty_cache()

        input_image = self._uint2tensor4(image)

        with torch.no_grad():
            output_image = (
                model(input_image.to(device))
                .data.squeeze()
                .float()
                .cpu()
                .clamp_(0, 1)
                .numpy()
            )

        return np.transpose(output_image, (1, 2, 0))

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
