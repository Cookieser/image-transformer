from abc import ABC, abstractmethod
from PIL import Image
import numpy as np


# Uniform output format -》np.ndarray
class ImageTransform(ABC):
    def __init__(self, range_min: float, range_max: float):
        self.range_min = range_min
        self.range_max = range_max

    @abstractmethod
    def apply(self, image: Image.Image, degree: float) -> np.ndarray:
        """Apply the transformation to the image with a given degree."""
        pass

    def validate_degree(self, degree: float):
        """Ensure the degree is within the valid range."""
        if not (self.range_min <= degree <= self.range_max):
            raise ValueError(f"Degree must be between {self.range_min} and {self.range_max}")

