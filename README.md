# Log Record

## 1. Path Issue Modification

Previously, the path was embedded in the program. It has now been extracted to make it easier to modify, and the concatenation method has been adjusted.

## 2. Overall Code Structure Refactoring

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

## 3. Adding a test function

A temporary test function has been added to facilitate testing of the methods.

## 4. Identified an issue with the method

The initial codes just test these five method:

```
for name in ['jpeg', 'brightness', 'gaussian noise', 'defocus blur','contrast']:
```

the method `FogTransform` is wrong

```
Traceback (most recent call last):
  File "/Users/yupuwang/Documents/Code/image-transformer/test.py", line 45, in <module>
    apply_transform_and_show(image_path, FogTransform, degree=1)
  File "/Users/yupuwang/Documents/Code/image-transformer/test.py", line 18, in apply_transform_and_show
    transformed_img = transform.apply(original_image, degree)
                      ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/yupuwang/Documents/Code/image-transformer/transforms_class.py", line 103, in apply
    image += degree * plasma_fractal(mapsize=image.shape[0], wibbledecay=2.0)[:image.shape[0], :image.shape[0]][
                      ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/yupuwang/Documents/Code/image-transformer/transforms_class.py", line 21, in plasma_fractal
    assert (mapsize & (mapsize - 1) == 0)
            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^
AssertionError
```





# Next

- Debug the aforementioned issue

- Complete the refactoring of the remaining two methods

- Find new methods and list them

- Modify the parameters of the previous methods

