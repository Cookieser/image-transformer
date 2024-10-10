

from PIL import Image, ImageEnhance, ImageFilter
import numpy as np
import random
import os
import io
from skimage.filters import gaussian
import skimage as sk
import cv2
from scipy.ndimage import map_coordinates
from pkg_resources import resource_filename
from tqdm import tqdm

from base import ImageTransform  # 引入基类


class ContrastTransform(ImageTransform):
    def __init__(self, range_min=0.5, range_max=2.0):
        super().__init__(range_min, range_max)

    def apply(self, image: Image.Image, degree: float) -> np.ndarray:
        self.validate_degree(degree)
        x = np.array(image) / 255.
        means = np.mean(x, axis=(0, 1), keepdims=True)
        return np.clip((x - means) * degree + means, 0, 1) * 255


class JPEGCompressionTransform(ImageTransform):
    def __init__(self, range_min=10, range_max=95):
        super().__init__(range_min, range_max)

    def apply(self, image: Image.Image, degree: float) -> np.ndarray:
        self.validate_degree(degree)
        output = io.BytesIO()
        image.save(output, 'JPEG', quality=degree)
        compressed_image = Image.open(output)
        return np.array(compressed_image)
