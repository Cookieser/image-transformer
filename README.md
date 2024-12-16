## Overview of the Program Structure

```
├── utils
│   └── test.py
├── Images
│   ├── {dataset}
│   ├── {dataset}_{XXXTransform}_{degree}
│   ├── {dataset}_{XXXTransform}_{degree}
│   ├── ...
│   ├── {dataset}_{XXXTransform}_{degree}
├── result
│   ├── {dataset}_{XXXTransform}_{photo_index}
├── base.py
├── transforms_class.py
├── main.py
├── result_analysis.py
├── save.py
├── README.md
├── Image Transformation.md
```

### Directories and Files

- **utils**: Contains utility scripts.
  - `test.py`: Used for testing the implemented image transformations (`Transform`) to check for potential errors.
- **Images**: Stores the dataset and the output images after applying different transformations and degrees.
  - `{dataset}`: Original dataset.
  - `{dataset}_{XXXTransform}_{degree}`: Transformed images for each specific transformation and degree.
- **result**: Contains the results of transformations applied to images from the dataset, indexed by `photo_index`.
  - `{dataset}_{XXXTransform}_{photo_index}`: Transformed images from the dataset for each photo and transformation.
- **base.py**: Defines the abstract class for `Transform`.
- **transforms_class.py**: Implements specific transformations derived from the abstract class defined in `base.py`.
- **main.py**: The main entry point for running the transformations on the dataset.
- **result_analysis.py**: Contains analysis functions to evaluate the results of the transformations.
- **save.py**: Handles saving of transformed images.
- **Image Transformation.md**: A comprehensive documentation of all the possible transformations.



## Running the Program

To run the program, you need to modify the parameters in `main.py` as follows:


Adjust the transformations and their respective ranges in `transforms_dict` as you like:

```
transforms_dict = {
        # Transform:(Min, Max, Step)
        BrightnessTransform: (0.2, 0.58, 0.02),
        ContrastTransform: (0.9, 1.85, 0.05),
        DefocusBlurTransform: (0.2, 0.39, 0.01),
        ElasticTransform: (0.1, 0.29, 0.01),
        FogTransform: (0.1, 0.29, 0.01),
    }
```

```
python main.py --image_directory <image_directory> --dataset_name <dataset_name>
```
We will read these original images from `<image_directory>/<dataset_name>` and save new images in `<image_directory>/<dataset_name_XXXTransform_degree>`."

For example,
```
python main.py --image_directory Images --dataset_name mmvp
```

### Key Parameters

- **`image_directory`**: Directory that contains the dataset and images.
- **`image_file`**: Name of the specific image to process.
- **`dataset_name`**: Name of the dataset used for image transformations.
- **`save_dir`**: Directory where the results will be saved.
- **`transforms_dict`**: Specifies the transformations and their ranges (minimum, maximum, and step values).















