import sys
import torch
import cv2 as cv
import numpy as np
import pandas as pd

from tqdm import tqdm
from torchvision import transforms
from ImageDatastore import ImageDatastore


def tensor_to_cv2(tensor):
    """
    Convert a PyTorch tensor to an OpenCV image
    Args:
        tensor: PyTorch tensor of shape (C, H, W) or (H, W, C)
    Returns:
        OpenCV image of shape (H, W, C) for RGB/BGR or (H, W) for grayscale
    """
    # If tensor is on GPU, move it to CPU and detach from computation graph
    if tensor.is_cuda:
        tensor = tensor.cpu()
    if tensor.requires_grad:
        tensor = tensor.detach()

    # Convert to numpy
    numpy_img = tensor.numpy()

    # Handle different tensor formats
    if len(numpy_img.shape) == 3:
        if numpy_img.shape[0] in [3, 4]:  # If image is in format [C, H, W]
            numpy_img = np.transpose(numpy_img, (1, 2, 0))

        # Ensure we have 3 channels for color images
        if numpy_img.shape[2] == 3:  # RGB/BGR image
            pass
        elif numpy_img.shape[2] == 1:  # Single channel image
            numpy_img = cv.cvtColor(numpy_img, cv.COLOR_GRAY2BGR)
        elif numpy_img.shape[2] == 4:  # RGBA image
            numpy_img = numpy_img[:, :, :3]

    # If values are float in [0,1] range, convert to [0,255] range
    if numpy_img.dtype == np.float32 or numpy_img.dtype == np.float64:
        numpy_img = (numpy_img * 255).astype(np.uint8)

    return numpy_img


def calculate_brisque_scores(dataset):
    """
    Calculate BRISQUE scores for all images in the dataset
    """
    blurScores = []
    brisqueScores = []

    for idx in tqdm(range(len(dataset))):
        image, _ = dataset[idx]
        # Add batch dimension and ensure image is on CPU
        image = tensor_to_cv2(image)
        grey = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

        blurScores.append(cv.Laplacian(grey, cv.CV_64F).var())
        brisqueScores.append(
            cv.quality.QualityBRISQUE_compute(
                image,
                "./Quality/brisque_model_live.yml",
                "./Quality/brisque_range_live.yml",
            )
        )

    return brisqueScores, blurScores


if __name__ == "__main__":
    batch_size = 512

    val_small = ImageDatastore("val_set", transform=None)
    val_deg = ImageDatastore("val_degradate", transform=None)

    # Initialize lists for NIQE scores
    val_brisque, val_blur = calculate_brisque_scores(val_small)
    deg_brisque, deg_blur = calculate_brisque_scores(val_deg)

    score = pd.DataFrame(
        {
            "val_brisque": val_brisque,
            "deg_brisque": deg_brisque,
            "val_blur": val_blur,
            "deg_blur": deg_blur,
        }
    )
    print(score.describe())
    score.to_csv("./Quality/brisque_scores.csv", index=True)
