import numpy as np
from base import ImageTransform  # 引入基类


class ContrastTransform(ImageTransform):
    def __init__(self, range_min=0.5, range_max=2.0):
        super().__init__(range_min, range_max)

    def apply(self, image: np.ndarray, degree: float) -> np.ndarray:
        self.validate_degree(degree)
        x = np.array(image) / 255.
        means = np.mean(x, axis=(0, 1), keepdims=True)
        return np.clip((x - means) * degree + means, 0, 1) * 255
