import argparse
from transforms_class import *
from result_analysis import *
from save import *
# ls | grep '^mmvp_P' | wc -l

if __name__ == "__main__":
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Apply image transformations and save the results.")
    parser.add_argument(
        "--image_directory",
        type=str,
        required=True,
        help="Base directory containing some image dataset folders like mmvp/."
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        required=True,
        help="Specific dataset name and we will read from image_directory/dataset_name and save in image_directory/dataset_name_transformation_xx."
    )

    args = parser.parse_args()
    
    transforms_dict = {
        #Transform:(Min, Max, Step)
        
        BrightnessTransform: (0.2, 0.58, 0.02),
        ContrastTransform: (0.9, 1.85, 0.05),
        DefocusBlurTransform: (0.2, 0.39, 0.01),
        ElasticTransform: (0.1, 0.29, 0.01),
        FogTransform: (0.1, 0.29, 0.01),
        GammaCorrectionTransform: (0.5, 0.69, 0.01),
        GaussianNoiseTransform: (0.2, 0.39, 0.01),
        JPEGCompressionTransform: (50, 88, 2),
        

        MedianFilterTransform: (0.01, 0.2, 0.01),
        PoissonNoiseTransform: (0.5, 0.88, 0.02),
        RotationTransform: (0.1, 2.0, 0.1),
        UniformNoiseTransform: (0.05, 0.24, 0.01),
        
        # LogarithmicTransform
        # ScalingTransform_13
        # TranslationTransform
        # GlassBlurTransform
        

    }

    save_image(args.dataset_name, args.image_directory, transforms_dict)



    # for transform_class, degree_range in transforms_dict.items():
    #     transformer_name = transform_class.__name__
    #     plot_degrees_in_same_transformer(image_directory, image_file, dataset_name, transformer_name, save_dir)