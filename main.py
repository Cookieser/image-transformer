from transforms_class import *
from result_analysis import *
from save import *

image_directory = 'Images'  # Base directory containing image subfolders
image_file = '23.png'  # The image name, adjust this based on your specific naming convention
dataset_name = 'mmvp'
save_dir = 'result'


transforms_dict = {
    # Transform:(Min, Max, Step)
    ContrastTransform: (0.9, 2.0, 0.1),
    JPEGCompressionTransform: (50, 90, 10),
    BrightnessTransform: (0.01, 0.06, 0.01),
    DefocusBlurTransform: (0.1, 0.8, 0.1),
    FogTransform: (0.1, 0.4, 0.1),
    GlassBlurTransform: (0.01, 0.05, 0.01),
    GaussianNoiseTransform: (0.02, 0.05, 0.01),
    ElasticTransform: (0.01, 0.02, 0.01),

}

save_image(dataset_name, image_directory, transforms_dict)

for transform_class, degree_range in transforms_dict.items():
    transformer_name = transform_class.__name__
    plot_degrees_in_same_transformer(image_directory, image_file, dataset_name, transformer_name, save_dir)
