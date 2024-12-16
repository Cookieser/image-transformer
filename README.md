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

```
image_directory = 'Images'  # Base directory containing image subfolders
image_file = '1.png'  # The image name, test different degrees(output -> result)
dataset_name = 'mmvp'  # The name of the dataset
save_dir = 'result'  # Directory to save the transformed images
```

Next, adjust the transformations and their respective ranges in `transforms_dict`:

```
transforms_dict = {
    # Transform: (Min, Max, Step)
    ContrastTransform: (0.1, 2.0, 0.1),
    JPEGCompressionTransform: (10, 90, 10),
    BrightnessTransform: (0.1, 2.0, 0.1),
    DefocusBlurTransform: (1, 10, 1),
    FogTransform: (0.5, 4.0, 0.5),
    GlassBlurTransform: (0.1, 1, 0.1),
    GaussianNoiseTransform: (0.02, 0.15, 0.01),
    ElasticTransform: (0.01, 0.1, 0.01),
}
```

### Key Parameters

- **`image_directory`**: Directory that contains the dataset and images.
- **`image_file`**: Name of the specific image to process.
- **`dataset_name`**: Name of the dataset used for image transformations.
- **`save_dir`**: Directory where the results will be saved.
- **`transforms_dict`**: Specifies the transformations and their ranges (minimum, maximum, and step values).





## The range of parameters 

We use these ranges to test, and the results are shown on `result`

```
transforms_dict = {
    # Transform: (Min, Max, Step)
    ContrastTransform: (0.1, 2.0, 0.1),
    JPEGCompressionTransform: (10, 90, 10),
    BrightnessTransform: (0.1, 2.0, 0.1),
    DefocusBlurTransform: (1, 10, 1),
    FogTransform: (0.5, 4.0, 0.5),
    GlassBlurTransform: (0.1, 1, 0.1),
    GaussianNoiseTransform: (0.02, 0.15, 0.01),
    ElasticTransform: (0.01, 0.1, 0.01),
}
```

We change these values according to these results

```
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
```



# Log Record

## 10.11

#### 1. Path Issue Modification

Previously, the path was embedded in the program. It has now been extracted to make it easier to modify, and the concatenation method has been adjusted.

#### 2. Overall Code Structure Refactoring

The original code contained a large number of if-else statements and nested loops, which required modifying many scattered places every time a new method was added. This was not scalable. The code has now been refactored using abstract classes.

```
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
```

```
class ContrastTransform(ImageTransform):
    def __init__(self, range_min=0.3, range_max=0.7):
        super().__init__(range_min, range_max)

    def apply(self, image: Image.Image, degree: float) -> np.ndarray:
        self.validate_degree(degree)
        image = np.array(image) / 255.
        means = np.mean(image, axis=(0, 1), keepdims=True)
        return np.clip((image - means) * degree + means, 0, 1) * 255
```

#### 3. Adding a test function

A temporary test function has been added to facilitate testing of the methods.



## 10.16

Summarize the possible transformations that may be used, as documented in `Image Transformation.md`.

## 10.17-10.18

1. Write the `save` function and complete the entire code refactoring.

2. Write the `result_analysis` function to aggregate the impact of the same transformation at different degrees on a single image, facilitating future adjustments of the degree range.

3. Adjust the degree ranges based on the existing transformations.

