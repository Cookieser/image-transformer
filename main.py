from transforms_class import *
from result_analysis import *
from save import *

image_directory = 'Images'  # Base directory containing image subfolders
image_file = '1.png'  # The image name, adjust this based on your specific naming convention
dataset_name = 'mmvp'
save_dir = 'result'


transforms_dict = {
    # Transform:(Min, Max, Step)
    ContrastTransform: (0.1, 2.0, 0.1),
    JPEGCompressionTransform: (10, 90, 10),
    BrightnessTransform: (0.1, 2.0, 0.1),
    DefocusBlurTransform: (1, 10, 1),
    FogTransform: (0.5, 4.0, 0.5),
    GlassBlurTransform: (0.1, 1, 0.1),
    GaussianNoiseTransform: (0.02, 0.15, 0.01),
    ElasticTransform: (0.01, 0.1, 0.01),

}

save_image(dataset_name, image_directory, transforms_dict)

for transform_class, degree_range in transforms_dict.items():
    transformer_name = transform_class.__name__
    plot_degrees_in_same_transformer(image_directory, image_file, dataset_name, transformer_name, save_dir)
