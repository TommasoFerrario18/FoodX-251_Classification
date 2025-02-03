import sys
import torch
import pandas as pd

from pyiqa import create_metric
from torchvision import transforms
from ImageDatastore import ImageDatastore


def calculate_brisque_scores(dataset):
    """
    Calculate BRISQUE scores for all images in the dataset
    """
    brisque_metric = create_metric("brisque")
    brisque_scores = []

    for idx in range(len(dataset)):
        image, _ = dataset[idx]
        # Add batch dimension and ensure image is on CPU
        image = image.unsqueeze(0).cpu()

        # Ensure correct channel ordering (B, C, H, W)
        if image.shape[1] != 3:
            image = image.permute(0, 3, 1, 2)

        # Ensure values are exactly in [0, 1] range
        image = torch.clamp(image, 0.0, 1.0)

        # Print shape for debugging
        print(f"Image shape: {image.shape}")

        try:
            brisque_score = float(brisque_metric(image))
            brisque_scores.append(brisque_score)
        except Exception as e:
            print(f"Error processing image {idx}: {str(e)}")
            print(
                f"Image stats - min: {image.min()}, max: {image.max()}, shape: {image.shape}"
            )
            raise

    return brisque_scores


if __name__ == "__main__":
    batch_size = 512
    transform = transforms.Compose(
        [
            transforms.Resize(
                (256, 256), interpolation=transforms.InterpolationMode.BILINEAR
            ),
            transforms.Grayscale(num_output_channels=1),
        ]
    )

    val_small = ImageDatastore("val_set", transform=transform)
    val_deg = ImageDatastore("val_degradate", transform=transform)

    # Initialize lists for NIQE scores
    val_brisque = calculate_brisque_scores(val_small)
    deg_brisque = calculate_brisque_scores(val_deg)

    score = pd.DataFrame({"val_brisque": val_brisque, "deg_brisque": deg_brisque})
    print(score.describe())
