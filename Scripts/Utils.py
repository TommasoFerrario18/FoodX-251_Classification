import os
import shutil
from torchvision import transforms

def create_or_clear_directory(output_folder):

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    else:
        for filename in os.listdir(output_folder):
            file_path = os.path.join(output_folder, filename)
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)

def transform(input_size):
    transform = transforms.Compose(
        [
            transforms.Resize(
                (input_size, input_size), interpolation=transforms.InterpolationMode.BILINEAR
            ),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    return transform