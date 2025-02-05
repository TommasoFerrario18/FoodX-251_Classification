import random
from torchvision import transforms
from PIL import Image, ImageDraw
import numpy as np
import cv2

class DataAugmentation:

    def __init__(self):
        self.params = {}

    def random_rotation(self, image):
        angle = random.uniform(-30, 30)
        self.params["rotation"] = angle
        return transforms.functional.rotate(image, angle)

    def random_horizontal_flip(self, image):
        if random.random() > 0.5:
            self.params["flip"] = True
            return transforms.functional.hflip(image)
        self.params["flip"] = False
        return image
    
    def random_saturation(self, image):
        saturation_factor = random.uniform(0.5, 1.5)
        self.params["saturation"] = saturation_factor
        color_jitter = transforms.ColorJitter(saturation=saturation_factor)
        
        return color_jitter(image)
    
    def random_gaussian_blur(self, im, max_kernel_size=5):

        image = np.array(im)

        kernel_size = random.randint(1, max_kernel_size)

        if kernel_size % 2 == 0:  
            kernel_size += 1
        
        blurred_image = cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)
    
        return Image.fromarray(blurred_image)
    
    def random_brightness(self, image, brightness_factor=0.3):

        enhancer = transforms.ColorJitter(brightness=brightness_factor)
        bright_image = enhancer(image)

        return bright_image
    
    def random_cutout(self, im, max_hole_size=0.2, min_hole_size=0.05):

        image = im.copy()

        width, height = image.size
        
        hole_width = random.randint(int(min_hole_size * width), int(max_hole_size * width))
        hole_height = random.randint(int(min_hole_size * height), int(max_hole_size * height))
        
        x1 = random.randint(0, width - hole_width)
        y1 = random.randint(0, height - hole_height)
        x2 = x1 + hole_width
        y2 = y1 + hole_height
        
        draw = ImageDraw.Draw(image)
        
        draw.rectangle([x1, y1, x2, y2], fill=(0, 0, 0))
        
        return image

    def random_resized_crop(self, image):
        i, j, h, w = transforms.RandomResizedCrop(size=(512, 512)).get_params(
            image, scale=(0.8, 1.0), ratio=(3 / 4, 4 / 3)
        )
        self.params["crop"] = (i, j, h, w)
        return transforms.functional.resized_crop(
            image,
            i,
            j,
            h,
            w,
            size=(512, 512),
            interpolation=transforms.InterpolationMode.NEAREST,
        )

    def apply_transformations(self, image):
        image = self.random_rotation(image)
        image = self.random_horizontal_flip(image)
        image = self.random_saturation(image)
        image = self.random_cutout(image)
        image = self.random_brightness(image)
        image = self.random_gaussian_blur(image)
        image = self.random_resized_crop(image)
        return image

    def generate_augmentations(self, image, num_augmentations=5):
        images = []
        for _ in range(num_augmentations):
            self.params = {}
            augmented_image = self.apply_transformations(image)
            images.append(augmented_image)
        return images