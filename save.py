import numpy as np
import os
from PIL import Image
from tqdm import tqdm
from PIL import ImageDraw, ImageFont


# 读取base path下的以“某具体dataset”命名的作为图片原文件
def save_image(dataset, base_path, transforms):
    for transform_class, degree_range in transforms.items():
        name = transform_class.__name__
        min_degree, max_degree, step = degree_range
        # Check if the transformation is for JPEG
        if name == 'JPEGCompressionTransform':
            # For JPEG, ensure degrees are integers
            degrees = np.arange(min_degree, max_degree + step, step).astype(int)
        else:
            # For other transformations, round the degrees to two decimal places
            degrees = np.round(np.arange(min_degree, max_degree + step, step), 2)

        for degree in tqdm(degrees, desc=f"Processing {name}", unit="degree", dynamic_ncols=False):
            # new these folder paths to save
            new_folder_path = os.path.join(base_path, f"{dataset}_{name}_{degree}/")
            os.makedirs(new_folder_path, exist_ok=True)

            # open these src_images
            folder = os.path.join(base_path, dataset)

            if not os.path.exists(folder):
                raise ValueError(f"Folder does not exist: {folder}")
            for root, dirs, files in os.walk(folder):
                # for file in tqdm(files, unit="file", dynamic_ncols=False):
                for file in files:
                    if file.endswith(".jpg"):
                        # Open the image
                        img_path = os.path.join(root, file)
                        img = Image.open(img_path)
                        original_image = img.convert('RGB')

                        transform = transform_class()

                        transformed_img = transform.apply(original_image, degree)

                        transformed_image = Image.fromarray(np.uint8(transformed_img))

                        save_path = img_path.replace(dataset, dataset + '_' + name + '_' +
                                                     str(degree))
                        file_name = file.replace('.jpg', '.png')
                        transformed_img_path = os.path.join(new_folder_path, file_name)

                        transformed_image.save(transformed_img_path)


def add_title_to_image(image, title):
    draw = ImageDraw.Draw(image)

    font = ImageFont.load_default()
    text_position = (10, 10)

    draw.text(text_position, title, font=font, fill=(255, 255, 255))  # 白色文字

    return image


def apply_transformations_and_show(image_path, transforms_dict):
    img = Image.open(image_path)
    original_image = img.convert('RGB')

    original_image.show(title="Original Image")

    for transform_class, degree_range in transforms_dict.items():
        min_degree, max_degree, step = degree_range
        degrees = np.arange(min_degree, max_degree + step, step)
        for degree in degrees:
            transform = transform_class()

            transformed_img = transform.apply(original_image, degree)

            transformed_image = Image.fromarray(np.uint8(transformed_img))
            transformed_image_with_text = add_title_to_image(transformed_image,
                                                             f"{transform.__class__.__name__} with degree {degree}")
            # not useful in macbook
            # transformed_image.show(title=f"{transform.__class__.__name__} with degree {degree}")
            transformed_image_with_text.show()

# apply_transformations_and_show(image_path, transforms_dict)
