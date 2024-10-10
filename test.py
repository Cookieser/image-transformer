import numpy as np
from PIL import Image
from transforms_class import ContrastTransform

# 指定电脑里图像的路径
image_path = '/Users/yupuwang/Documents/Code/image-transformer/Images/mmvp/4.jpg'

# 加载图像
img = Image.open(image_path)

# 显示原始图像
original_image = img.convert('RGB')
original_image.show(title="Original Image")

# 初始化 ContrastTransform 变换类
contrast_transform = ContrastTransform()

# 应用对比度变换
contrast_img = contrast_transform.apply(img, degree=0.7)

# 显示变换后的图像
contrast_image = Image.fromarray(np.uint8(contrast_img))
contrast_image.show(title="Contrast Transformed Image")

