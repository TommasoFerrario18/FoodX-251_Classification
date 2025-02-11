import random
from torchvision import transforms
from PIL import Image, ImageDraw
import numpy as np
import cv2

class DataAugmentation2:

    def __init__(self):
        self.params = {}

    def random_rgb_value(self):
        return tuple(np.random.randint(0, 256, 3))
    
    def add_salt_and_pepper_RGB(self, img, prob=0.01):
        output = np.zeros(img.shape, np.uint8)
        for i in range(img.shape[0]):
            for j in range(img.shape[1]):
                rdn = random.random()
                if rdn < prob:
                    output[i][j] = self.random_rgb_value()
                else:
                    output[i][j] = img[i][j]
        return output
    
    def add_gaussian_noise(self, img, mean=0, std=50):
        noise = np.random.normal(mean, std, img.shape)
        output = img + noise
        output = np.clip(output, 0, 255)
        output = np.uint8(output)
        return output
    
    def add_jpeg_compression(self, img):
        quality = np.random.randint(50, 100)
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
        _, encimg = cv2.imencode(".jpg", img, encode_param)
        decimg = cv2.imdecode(encimg, 1)
        return decimg
    
    def add_blur(self, img, kernel_size=5):
        strength = np.random.randint(1, 10)
        for i in range(strength):
            img = cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)
        return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)

    def apply_transformations(self, image):
        image = np.array(image)

        n = np.random.randint(1, 5)
        for _ in range(n):
            r = np.random.randint(1, 6)
            if r == 1:
                image = self.add_salt_and_pepper_RGB(image)
            elif r == 2:
                image = self.add_gaussian_noise(image)
            elif r == 3:
                image = self.add_jpeg_compression(image)
            elif r == 4:
                image = self.add_blur(image)
        
        return image

    def generate_augmentations(self, image, num_augmentations=5):
        images = []
        for _ in range(num_augmentations):
            self.params = {}
            augmented_image = self.apply_transformations(image)
            images.append(augmented_image)
        return images