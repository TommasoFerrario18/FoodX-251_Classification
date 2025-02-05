import os
import shutil
from torchvision import transforms
import pandas as pd
import numpy as np
import base64

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

def get_datasets():

    base_feature_path = '../Features/features/'

    df_small = pd.read_csv('../Dataset/train_small.csv')
    df_unlabeled = pd.read_csv('../Dataset/train_unlabeled.csv')

    feat_small = np.load(base_feature_path + 'mobilenet_v3_classifier.npy')
    feat_unlabeled = np.load(base_feature_path + 'mobilenet_v3_classifier_unlabelled.npy')

    return df_small, feat_small, df_unlabeled, feat_unlabeled

def get_validation_datasets():
    df_degraded = pd.read_csv('../Dataset/val_info.csv')
    return df_degraded

def encode_image(image_file):
    with open(image_file, 'rb') as f:
        return 'data:image/png;base64,' + base64.b64encode(f.read()).decode('ascii')