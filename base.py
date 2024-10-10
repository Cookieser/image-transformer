from abc import ABC, abstractmethod
import numpy as np


class ImageTransform(ABC):
    def __init__(self, range_min: float, range_max: float):
        self.range_min = range_min
        self.range_max = range_max

    @abstractmethod
    def apply(self, image: np.ndarray, degree: float) -> np.ndarray:
        """Apply the transformation to the image with a given degree."""
        pass

    def validate_degree(self, degree: float):
        """Ensure the degree is within the valid range."""
        if not (self.range_min <= degree <= self.range_max):
            raise ValueError(f"Degree must be between {self.range_min} and {self.range_max}")


def gaussian_noise(x, c):
    x = np.array(x) / 255.
    return np.clip(x + np.random.normal(size=x.shape, scale=c), 0, 1) * 255


def jpeg_compression(x, c):
    output = io.BytesIO()
    x.save(output, 'JPEG', quality=c)
    x = Image.open(output)
    return x


def brightness(x, c):
    x = np.array(x) / 255.
    x = sk.color.rgb2hsv(x)
    x[:, :, 2] = np.clip(x[:, :, 2] + c, 0, 1)
    x = sk.color.hsv2rgb(x)
    return np.clip(x, 0, 1) * 255


def contrast(x, c):
    x = np.array(x) / 255.
    means = np.mean(x, axis=(0, 1), keepdims=True)
    return np.clip((x - means) * c + means, 0, 1) * 255
