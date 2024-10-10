import numpy as np
from PIL import Image
from transforms_class import *


def apply_transform_and_show(image_path, transform_class, degree):
    img = Image.open(image_path)
    original_image = img.convert('RGB')

    original_image.show(title="Original Image")

    transform = transform_class()

    transformed_img = transform.apply(original_image, degree)

    transformed_image = Image.fromarray(np.uint8(transformed_img))
    transformed_image.show(title="Transformed Image")


image_path = '/Users/yupuwang/Documents/Code/image-transformer/Images/mmvp/4.jpg'

# apply_transform_and_show(image_path, ContrastTransform, degree=0.7)

# apply_transform_and_show(image_path, JPEGCompressionTransform, degree=75)

# apply_transform_and_show(image_path, BrightnessTransform, degree=0.1)

# apply_transform_and_show(image_path, DefocusBlurTransform, degree=3)

# apply_transform_and_show(image_path, GlassBlurTransform, degree=0.5)

# apply_transform_and_show(image_path, GaussianNoiseTransform, degree=0.05)

# Todo: There is something wrong with this  FogTransform
# apply_transform_and_show(image_path, FogTransform, degree=1)



