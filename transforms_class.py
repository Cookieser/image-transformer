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


def plasma_fractal(mapsize=256, wibbledecay=3):
    """
    Generate a heightmap using diamond-square algorithm.
    Return square 2d array, side length 'mapsize', of floats in range 0-255.
    'mapsize' must be a power of two.
    """
    assert (mapsize & (mapsize - 1) == 0)
    maparray = np.empty((mapsize, mapsize), dtype=np.float_)
    maparray[0, 0] = 0
    stepsize = mapsize
    wibble = 100


def disk(radius, alias_blur=0.1, dtype=np.float32):
    if radius <= 8:
        L = np.arange(-8, 8 + 1)
        ksize = (3, 3)
    else:
        L = np.arange(-radius, radius + 1)
        ksize = (5, 5)
    X, Y = np.meshgrid(L, L)
    aliased_disk = np.array((X ** 2 + Y ** 2) <= radius ** 2, dtype=dtype)
    aliased_disk /= np.sum(aliased_disk)

    # supersample disk to antialias
    return cv2.GaussianBlur(aliased_disk, ksize=ksize, sigmaX=alias_blur)


class ContrastTransform(ImageTransform):
    def __init__(self, range_min=0.3, range_max=0.7):
        super().__init__(range_min, range_max)

    def apply(self, image: Image.Image, degree: float) -> np.ndarray:
        self.validate_degree(degree)
        image = np.array(image) / 255.
        means = np.mean(image, axis=(0, 1), keepdims=True)
        return np.clip((image - means) * degree + means, 0, 1) * 255


class JPEGCompressionTransform(ImageTransform):
    def __init__(self, range_min=30, range_max=70):
        super().__init__(range_min, range_max)

    def apply(self, image: Image.Image, degree: float) -> np.ndarray:
        self.validate_degree(degree)
        output = io.BytesIO()
        image.save(output, 'JPEG', quality=degree)
        compressed_image = Image.open(output)
        return np.array(compressed_image)


class BrightnessTransform(ImageTransform):
    def __init__(self, range_min=0.1, range_max=0.5):
        super().__init__(range_min, range_max)

    def apply(self, image: Image.Image, degree: float) -> np.ndarray:
        self.validate_degree(degree)
        image = np.array(image) / 255.
        image = sk.color.rgb2hsv(image)
        image[:, :, 2] = np.clip(image[:, :, 2] + degree, 0, 1)
        image = sk.color.hsv2rgb(image)
        return np.clip(image, 0, 1) * 255


class DefocusBlurTransform(ImageTransform):
    def __init__(self, range_min=1, range_max=5):
        super().__init__(range_min, range_max)

    def apply(self, image: Image.Image, degree: float) -> np.ndarray:
        self.validate_degree(degree)
        image = np.array(image) / 255.
        kernel = disk(radius=degree, alias_blur=0.5)

        channels = []
        for d in range(3):
            channels.append(cv2.filter2D(image[:, :, d], -1, kernel))
        channels = np.array(channels).transpose((1, 2, 0))  # 3x224x224 -> 224x224x3

        return np.clip(channels, 0, 1) * 255


class FogTransform(ImageTransform):
    def __init__(self, range_min=0.5, range_max=2.5):
        super().__init__(range_min, range_max)

    def apply(self, image: Image.Image, degree: float) -> np.ndarray:
        image = np.array(image) / 255.
        max_val = image.max()
        image += degree * plasma_fractal(mapsize=image.shape[0], wibbledecay=2.0)[:image.shape[0], :image.shape[0]][
            ..., np.newaxis]
        return np.clip(image * max_val / (max_val + degree), 0, 1) * 255


class GlassBlurTransform(ImageTransform):
    def __init__(self, range_min=0.2, range_max=1):
        super().__init__(range_min, range_max)

    def apply(self, image: Image.Image, degree: float) -> np.ndarray:
        image = np.uint8(gaussian(np.array(image) / 255., sigma=degree, channel_axis=2) * 255)
        # locally shuffle pixels
        for i in range(2):
            for h in range(image.shape[0] - 1, 1, -1):
                for w in range(image.shape[1] - 1, 1, -1):
                    dx, dy = np.random.randint(-1, 1, size=(2,))
                    h_prime, w_prime = h + dy, w + dx
                    # swap
                    image[h, w], image[h_prime, w_prime] = image[h_prime, w_prime], image[h, w]

        return np.clip(gaussian(image / 255., sigma=degree, channel_axis=2), 0, 1) * 255


class GaussianNoiseTransform(ImageTransform):
    def __init__(self, range_min=0.02, range_max=0.1):
        super().__init__(range_min, range_max)

    def apply(self, image: Image.Image, degree: float) -> np.ndarray:
        image = np.array(image) / 255.
        return np.clip(image + np.random.normal(size=image.shape, scale=degree), 0, 1) * 255
