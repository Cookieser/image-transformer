from transforms_class import *
from result_analysis import *
from save import *

image_directory = 'Images'  # Base directory containing image subfolders
image_file = '1.png'  # The image name, adjust this based on your specific naming convention
dataset_name = 'mmvp'
save_dir = 'result'

# apply_transform_and_show(image_path, SaltAndPepperNoiseTransform, degree=0.0001)

# apply_transform_and_show(image_path, PoissonNoiseTransform, degree=0.5)

# apply_transform_and_show(image_path, UniformNoiseTransform, degree=0.1)

# apply_transform_and_show(image_path, RotationTransform, degree=1)


# apply_transform_and_show(image_path, MedianFilterTransform, degree=0.1)

# apply_transform_and_show(image_path, GammaCorrectionTransform, degree=1)

# apply_transform_and_show(image_path, LogarithmicTransform, degree=0.0001)


transforms_dict = {
    # Transform:(Min, Max, Step)
    PoissonNoiseTransform: (0.5, 1.0, 0.1),
    UniformNoiseTransform: (0.05, 0.2, 0.01),
    RotationTransform: (0.5, 1.5, 0.1),
    MedianFilterTransform: (0.05, 0.15, 0.01),
    GammaCorrectionTransform: (0.5, 0.7, 0.1),
    # ContrastTransform: (0.9, 1.9, 0.1),
    # JPEGCompressionTransform: (50, 90, 10),
    # BrightnessTransform: (0.01, 0.06, 0.01),
    # DefocusBlurTransform: (0.1, 0.8, 0.1),
    # FogTransform: (0.1, 0.4, 0.1),
    # GlassBlurTransform: (0.01, 0.05, 0.01),
    # GaussianNoiseTransform: (0.02, 0.05, 0.01),
    # ElasticTransform: (0.01, 0.02, 0.01),

}

save_image(dataset_name, image_directory, transforms_dict)

for transform_class, degree_range in transforms_dict.items():
    transformer_name = transform_class.__name__
    plot_degrees_in_same_transformer(image_directory, image_file, dataset_name, transformer_name, save_dir)
