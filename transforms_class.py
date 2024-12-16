from PIL import Image, ImageEnhance, ImageFilter, ImageDraw
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
from base import ImageTransform
from scipy.signal import wiener


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

    def wibbledmean(array):
        return array / 4 + wibble * np.random.uniform(-wibble, wibble, array.shape)

    def fillsquares():
        """For each square of points stepsize apart,
           calculate middle value as mean of points + wibble"""
        cornerref = maparray[0:mapsize:stepsize, 0:mapsize:stepsize]
        squareaccum = cornerref + np.roll(cornerref, shift=-1, axis=0)
        squareaccum += np.roll(squareaccum, shift=-1, axis=1)
        maparray[stepsize // 2:mapsize:stepsize,
        stepsize // 2:mapsize:stepsize] = wibbledmean(squareaccum)

    def filldiamonds():
        """For each diamond of points stepsize apart,
           calculate middle value as mean of points + wibble"""
        mapsize = maparray.shape[0]
        drgrid = maparray[stepsize // 2:mapsize:stepsize, stepsize // 2:mapsize:stepsize]
        ulgrid = maparray[0:mapsize:stepsize, 0:mapsize:stepsize]
        ldrsum = drgrid + np.roll(drgrid, 1, axis=0)
        lulsum = ulgrid + np.roll(ulgrid, -1, axis=1)
        ltsum = ldrsum + lulsum
        maparray[0:mapsize:stepsize, stepsize // 2:mapsize:stepsize] = wibbledmean(ltsum)
        tdrsum = drgrid + np.roll(drgrid, 1, axis=1)
        tulsum = ulgrid + np.roll(ulgrid, -1, axis=0)
        ttsum = tdrsum + tulsum
        maparray[stepsize // 2:mapsize:stepsize, 0:mapsize:stepsize] = wibbledmean(ttsum)

    while stepsize >= 2:
        fillsquares()
        filldiamonds()
        stepsize //= 2
        wibble /= wibbledecay

    maparray -= maparray.min()
    return maparray / maparray.max()


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
        # self.validate_degree(degree)
        image = np.array(image) / 255.
        means = np.mean(image, axis=(0, 1), keepdims=True)
        return np.clip((image - means) * degree + means, 0, 1) * 255


class JPEGCompressionTransform(ImageTransform):
    def __init__(self, range_min=30, range_max=70):
        super().__init__(range_min, range_max)

    def apply(self, image: Image.Image, degree: float) -> np.ndarray:
        output = io.BytesIO()
        degree = int(degree)
        image.save(output, 'JPEG', quality=degree)
        compressed_image = Image.open(output)
        return np.array(compressed_image)


class BrightnessTransform(ImageTransform):
    def __init__(self, range_min=0.2, range_max=0.6):
        super().__init__(range_min, range_max)

    def apply(self, image: Image.Image, degree: float) -> np.ndarray:
        image = np.array(image) / 255.
        image = sk.color.rgb2hsv(image)
        image[:, :, 2] = np.clip(image[:, :, 2] + degree* 0.1, 0, 1)
        image = sk.color.hsv2rgb(image)
        return np.clip(image, 0, 1) * 255


class DefocusBlurTransform(ImageTransform):
    def __init__(self, range_min=1, range_max=5):
        super().__init__(range_min, range_max)

    def apply(self, image: Image.Image, degree: float) -> np.ndarray:
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
        # Adjust the map_size to the nearest power of two
        map_size = 1 << (image.shape[0] - 1).bit_length()
        # print(f"Adjusted map_size from {image.shape[0]} to nearest power of two: {map_size}")
        image += degree * plasma_fractal(mapsize=map_size, wibbledecay=2.0)[:image.shape[0], :image.shape[0]][
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
    def __init__(self, range_min=0.2, range_max=1):
        super().__init__(range_min, range_max)

    def apply(self, image: Image.Image, degree: float) -> np.ndarray:
        image = np.array(image) / 255.
        return np.clip(image + np.random.normal(size=image.shape, scale=degree*0.1), 0, 1) * 255


class ElasticTransform(ImageTransform):
    def __init__(self, range_min=0.1, range_max=0.5):
        super().__init__(range_min, range_max)

    def apply(self, image: Image.Image, degree: float) -> np.ndarray:
        params = (np.array(image).shape[0] * degree * 0.1, np.array(image).shape[0] * 0.01, np.array(image).shape[0] * 0.02)

        image = np.array(image, dtype=np.float32) / 255.
        shape = image.shape
        shape_size = shape[:2]

        # random affine
        center_square = np.float32(shape_size) // 2
        square_size = min(shape_size) // 3
        pts1 = np.float32([center_square + square_size,
                           [center_square[0] + square_size, center_square[1] - square_size],
                           center_square - square_size])
        pts2 = pts1 + np.random.uniform(-params[2], params[2], size=pts1.shape).astype(np.float32)
        M = cv2.getAffineTransform(pts1, pts2)
        image = cv2.warpAffine(image, M, shape_size[::-1], borderMode=cv2.BORDER_REFLECT_101)

        dx = (gaussian(np.random.uniform(-1, 1, size=shape[:2]),
                       params[1], mode='reflect', truncate=3) * params[0]).astype(np.float32)
        dy = (gaussian(np.random.uniform(-1, 1, size=shape[:2]),
                       params[1], mode='reflect', truncate=3) * params[0]).astype(np.float32)
        dx, dy = dx[..., np.newaxis], dy[..., np.newaxis]

        x, y, z = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]), np.arange(shape[2]))
        indices = np.reshape(y + dy, (-1, 1)), np.reshape(x + dx, (-1, 1)), np.reshape(z, (-1, 1))
        return np.clip(map_coordinates(image, indices, order=1, mode='reflect').reshape(shape), 0, 1) * 255


class FrostTransform(ImageTransform):
    def __init__(self, range_min=0.5, range_max=2.5):
        super().__init__(range_min, range_max)

    def apply(self, image: Image.Image, degree: float) -> np.ndarray:
        idx = np.random.randint(5)
        filename = [resource_filename(__name__, 'frost/frost1.png'),
                    resource_filename(__name__, 'frost/frost2.png'),
                    resource_filename(__name__, 'frost/frost3.png'),
                    resource_filename(__name__, 'frost/frost4.jpg'),
                    resource_filename(__name__, 'frost/frost5.jpg'),
                    resource_filename(__name__, 'frost/frost6.jpg')][idx]
        frost = cv2.imread(filename)
        frost = cv2.resize(frost, (int(1.5 * np.array(image).shape[0]), int(1.5 * np.array(image).shape[1])))
        # randomly crop and convert to rgb
        x_start, y_start = (np.random.randint(0, frost.shape[0] - np.array(image).shape[0]), np.random.randint
        (0, frost.shape[1] - np.array(image).shape[1]))
        frost = frost[x_start:x_start + np.array(image).shape[0], y_start:y_start + np.array(image).shape[1]][
            ..., [2, 1, 0]]
        return np.clip(np.array(image) + degree * frost, 0, 255)


class SaltAndPepperNoiseTransform(ImageTransform):
    def __init__(self, range_min=0.01, range_max=0.1, salt_vs_pepper=0.5):
        super().__init__(range_min, range_max)
        self.salt_vs_pepper = salt_vs_pepper

    def apply(self, image: Image.Image, degree: float) -> np.ndarray:
        # self.validate_degree(degree)
        image = np.array(image) / 255.  # Normalize the image to [0, 1]
        noisy_image = np.copy(image)

        total_pixels = image.size
        num_salt = np.ceil(degree * total_pixels * self.salt_vs_pepper)
        num_pepper = np.ceil(degree * total_pixels * (1.0 - self.salt_vs_pepper))

        coords_salt = [np.random.randint(0, i - 1, int(num_salt)) for i in image.shape[:2]]
        noisy_image[coords_salt[0], coords_salt[1], :] = 1

        coords_pepper = [np.random.randint(0, i - 1, int(num_pepper)) for i in image.shape[:2]]
        noisy_image[coords_pepper[0], coords_pepper[1], :] = 0

        return np.clip(noisy_image, 0, 1) * 255


class PoissonNoiseTransform(ImageTransform):
    def __init__(self, range_min=0.05, range_max=0.3):
        super().__init__(range_min, range_max)

    def apply(self, image: Image.Image, degree: float) -> np.ndarray:
        # Normalize the image to [0, 1]
        image = np.array(image) / 255.

        # Scale the image slightly based on the degree to generate subtle Poisson noise
        scaled_image = image * (degree * 255)

        # Generate Poisson noise
        noisy_image = np.random.poisson(scaled_image) / (degree * 255)

        return np.clip(noisy_image, 0, 1) * 255  # Clip to ensure values within [0, 255]


class UniformNoiseTransform(ImageTransform):
    def __init__(self, range_min=0.01, range_max=0.1):
        super().__init__(range_min, range_max)

    def apply(self, image: Image.Image, degree: float) -> np.ndarray:
        # Normalize the image to [0, 1]
        image = np.array(image) / 255.

        # Generate uniform noise in the range [-degree, degree]
        noise = np.random.uniform(-degree*0.1, degree*0.1, image.shape)

        # Add the uniform noise to the image
        noisy_image = image + noise

        # Clip to ensure the values are within [0, 1] range
        return np.clip(noisy_image, 0, 1) * 255


class RotationTransform(ImageTransform):
    def __init__(self, range_min=-30, range_max=30):
        # Allows rotation within the specified degree range
        super().__init__(range_min, range_max)

    def apply(self, image: Image.Image, degree: float) -> np.ndarray:
        # Rotate the image by the specified degree
        rotated_image = image.rotate(degree*0.1, resample=Image.BICUBIC, expand=True)
        return np.array(rotated_image)


class MedianFilterTransform(ImageTransform):
    def __init__(self, range_min=3, range_max=7):
        # range_min and range_max define the range of kernel sizes
        super().__init__(range_min, range_max)

    def apply(self, image: Image.Image, degree: float) -> np.ndarray:
        # Convert the degree to a valid kernel size, which should be odd
        kernel_size = int(self.range_min + degree * (self.range_max - self.range_min))
        kernel_size = max(3, kernel_size | 1)  # Ensure it's an odd number, minimum 3

        # Convert the image to a numpy array for OpenCV
        image_array = np.array(image)

        # Apply median filtering using OpenCV
        filtered_image = cv2.medianBlur(image_array, kernel_size)

        return np.array(Image.fromarray(filtered_image))


class GammaCorrectionTransform(ImageTransform):
    def __init__(self, range_min=0.5, range_max=2.0):
        # range_min and range_max define the range for the gamma value
        super().__init__(range_min, range_max)

    def apply(self, image: Image.Image, degree: float) -> np.ndarray:
        # Calculate the gamma value based on the degree (range_min to range_max)
        gamma = self.range_min + degree * (self.range_max - self.range_min)

        # Convert image to a numpy array and normalize to [0, 1]
        image_array = np.array(image) / 255.0

        # Apply gamma correction: output = input^(1/gamma)
        gamma_corrected = np.power(image_array, 1.0 / gamma)

        # Scale back to [0, 255] and convert to uint8
        gamma_corrected = np.clip(gamma_corrected * 255.0, 0, 255).astype(np.uint8)

        return np.array(Image.fromarray(gamma_corrected))


class LogarithmicTransform(ImageTransform):
    def __init__(self, range_min=1, range_max=255):
        # range_min and range_max control the scaling factor for the log transform
        super().__init__(range_min, range_max)

    def apply(self, image: Image.Image, degree: float) -> np.ndarray:
        # Calculate the scaling constant based on the degree
        c = self.range_min + degree * (self.range_max - self.range_min)

        # Convert the image to a numpy array and normalize to [0, 1]
        image_array = np.array(image) / 255.0

        # Apply logarithmic transformation: output = c * log(1 + input)
        log_transformed = c * np.log1p(image_array)  # log1p computes log(1 + x)

        # Scale back to [0, 255], and clip values to ensure they fall within the valid range
        log_transformed = np.clip(log_transformed * 255.0 / np.log1p(1), 0, 255).astype(np.uint8)

        return np.array(Image.fromarray(log_transformed))
