from PIL import Image, ImageEnhance, ImageFilter
import numpy as np
import random
import os
import io
from skimage.filters import gaussian
import skimage as sk
import cv2
from scipy.ndimage import map_coordinates
from pkg_resources import resource_filename
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# The base path of the project includes the original images directory and transformed images
# In the beginning, given these original images fold in the base_path by their dataset name e.g. mmvp
base_path = '/Users/yupuwang/Documents/Code/image-transformer/Images'


def to_float32(image):
    image = np.array(image, dtype=np.float32) / 255.0
    return image


def to_uint8(image):
    return (np.clip(image, 0, 1) * 255).astype(np.uint8)


def disk(radius, alias_blur=0.1, dtype=np.float32):
    if radius <= 8:
        L = np.arange(-8, 8 + 1)
        ksize = (3, 3)
    else:
        L = np.arange(-radius, radius + 1)
        ksize = (5, 5)
    X, Y = np.meshgrid(L, L)
    aliased_disk = np.array((X ** 2 + Y ** 2) <= radius ** 2, dtype=dtype)
    aliased_disk /= np.sum(aliased_disk)

    # supersample disk to antialias
    return cv2.GaussianBlur(aliased_disk, ksize=ksize, sigmaX=alias_blur)


def plasma_fractal(mapsize=256, wibbledecay=3):
    """
    Generate a heightmap using diamond-square algorithm.
    Return square 2d array, side length 'mapsize', of floats in range 0-255.
    'mapsize' must be a power of two.
    """
    assert (mapsize & (mapsize - 1) == 0)
    maparray = np.empty((mapsize, mapsize), dtype=np.float_)
    maparray[0, 0] = 0
    stepsize = mapsize
    wibble = 100

    def wibbledmean(array):
        return array / 4 + wibble * np.random.uniform(-wibble, wibble, array.shape)

    def fillsquares():
        """For each square of points stepsize apart,
           calculate middle value as mean of points + wibble"""
        cornerref = maparray[0:mapsize:stepsize, 0:mapsize:stepsize]
        squareaccum = cornerref + np.roll(cornerref, shift=-1, axis=0)
        squareaccum += np.roll(squareaccum, shift=-1, axis=1)
        maparray[stepsize // 2:mapsize:stepsize,
        stepsize // 2:mapsize:stepsize] = wibbledmean(squareaccum)

    def filldiamonds():
        """For each diamond of points stepsize apart,
           calculate middle value as mean of points + wibble"""
        mapsize = maparray.shape[0]
        drgrid = maparray[stepsize // 2:mapsize:stepsize, stepsize // 2:mapsize:stepsize]
        ulgrid = maparray[0:mapsize:stepsize, 0:mapsize:stepsize]
        ldrsum = drgrid + np.roll(drgrid, 1, axis=0)
        lulsum = ulgrid + np.roll(ulgrid, -1, axis=1)
        ltsum = ldrsum + lulsum
        maparray[0:mapsize:stepsize, stepsize // 2:mapsize:stepsize] = wibbledmean(ltsum)
        tdrsum = drgrid + np.roll(drgrid, 1, axis=1)
        tulsum = ulgrid + np.roll(ulgrid, -1, axis=0)
        ttsum = tdrsum + tulsum
        maparray[stepsize // 2:mapsize:stepsize, 0:mapsize:stepsize] = wibbledmean(ttsum)

    while stepsize >= 2:
        fillsquares()
        filldiamonds()
        stepsize //= 2
        wibble /= wibbledecay

    maparray -= maparray.min()
    return maparray / maparray.max()


def gaussian_noise(x, c):
    x = np.array(x) / 255.
    return np.clip(x + np.random.normal(size=x.shape, scale=c), 0, 1) * 255


def jpeg_compression(x, c):
    output = io.BytesIO()
    x.save(output, 'JPEG', quality=c)
    x = Image.open(output)
    return x


def brightness(x, c):
    x = np.array(x) / 255.
    x = sk.color.rgb2hsv(x)
    x[:, :, 2] = np.clip(x[:, :, 2] + c, 0, 1)
    x = sk.color.hsv2rgb(x)
    return np.clip(x, 0, 1) * 255


def contrast(x, c):
    x = np.array(x) / 255.
    means = np.mean(x, axis=(0, 1), keepdims=True)
    return np.clip((x - means) * c + means, 0, 1) * 255


def defocus_blur(x, c):
    x = np.array(x) / 255.
    kernel = disk(radius=c, alias_blur=0.5)

    channels = []
    for d in range(3):
        channels.append(cv2.filter2D(x[:, :, d], -1, kernel))
    channels = np.array(channels).transpose((1, 2, 0))  # 3x224x224 -> 224x224x3

    return np.clip(channels, 0, 1) * 255


def elastic_transform(image, c):
    params = (np.array(image).shape[0] * c, np.array(image).shape[0] * 0.01, np.array(image).shape[0] * 0.02)

    image = np.array(image, dtype=np.float32) / 255.
    shape = image.shape
    shape_size = shape[:2]

    # random affine
    center_square = np.float32(shape_size) // 2
    square_size = min(shape_size) // 3
    pts1 = np.float32([center_square + square_size,
                       [center_square[0] + square_size, center_square[1] - square_size],
                       center_square - square_size])
    pts2 = pts1 + np.random.uniform(-params[2], params[2], size=pts1.shape).astype(np.float32)
    M = cv2.getAffineTransform(pts1, pts2)
    image = cv2.warpAffine(image, M, shape_size[::-1], borderMode=cv2.BORDER_REFLECT_101)

    dx = (gaussian(np.random.uniform(-1, 1, size=shape[:2]),
                   params[1], mode='reflect', truncate=3) * params[0]).astype(np.float32)
    dy = (gaussian(np.random.uniform(-1, 1, size=shape[:2]),
                   params[1], mode='reflect', truncate=3) * params[0]).astype(np.float32)
    dx, dy = dx[..., np.newaxis], dy[..., np.newaxis]

    x, y, z = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]), np.arange(shape[2]))
    indices = np.reshape(y + dy, (-1, 1)), np.reshape(x + dx, (-1, 1)), np.reshape(z, (-1, 1))
    return np.clip(map_coordinates(image, indices, order=1, mode='reflect').reshape(shape), 0, 1) * 255


def fog(x, c):
    x = np.array(x) / 255.
    max_val = x.max()
    x += c * plasma_fractal(mapsize=x.shape[0], wibbledecay=2.0)[:x.shape[0], :x.shape[0]][..., np.newaxis]
    return np.clip(x * max_val / (max_val + c), 0, 1) * 255


def frost(x, c):
    idx = np.random.randint(5)
    filename = [resource_filename(__name__, 'frost/frost1.png'),
                resource_filename(__name__, 'frost/frost2.png'),
                resource_filename(__name__, 'frost/frost3.png'),
                resource_filename(__name__, 'frost/frost4.jpg'),
                resource_filename(__name__, 'frost/frost5.jpg'),
                resource_filename(__name__, 'frost/frost6.jpg')][idx]
    frost = cv2.imread(filename)
    frost = cv2.resize(frost, (int(1.5 * np.array(x).shape[0]), int(1.5 * np.array(x).shape[1])))
    # randomly crop and convert to rgb
    x_start, y_start = np.random.randint(0, frost.shape[0] - np.array(x).shape[0]), np.random.randint(0, frost.shape[1] - np.array(x).shape[1])
    frost = frost[x_start:x_start + np.array(x).shape[0], y_start:y_start + np.array(x).shape[1]][..., [2, 1, 0]]

    return np.clip(np.array(x) + c * frost, 0, 255)


def glass_blur(x, c):
    x = np.uint8(gaussian(np.array(x) / 255., sigma=c, channel_axis=2) * 255)
    # locally shuffle pixels
    for i in range(2):
        for h in range(x.shape[0] - 1, 1, -1):
            for w in range(x.shape[1] - 1, 1, -1):
                dx, dy = np.random.randint(-1, 1, size=(2,))
                h_prime, w_prime = h + dy, w + dx
                # swap
                x[h, w], x[h_prime, w_prime] = x[h_prime, w_prime], x[h, w]

    return np.clip(gaussian(x / 255., sigma=c, channel_axis=2), 0, 1) * 255


dataset = 'mmvp'
each_time = 5
jpeg_range = [30, 70]
ba_range = [0.1, 0.5]
ca_range = [0.3, 0.7]
db_range = [1, 5]
eb_range = [0.01, 0.05]
fog_range = [0.5, 2.5]
frost_range = [0.2, 0.6]
gn_range = [0.02, 0.1]
gb_range = [0.2, 1]
for name in ['jpeg', 'brightness', 'gaussian noise', 'defocus blur']:
    if name == 'jpeg':
        params = jpeg_range
    elif name == 'brightness':
        params = ba_range
    elif name == 'contrast':
        params = ca_range
    elif name == 'defocus blur':
        params = db_range
    elif name == 'elastic blur':
        params = eb_range
    elif name == 'fog blur':
        params = fog_range
    elif name == 'frost blur':
        params = frost_range
    elif name == 'gaussian noise':
        params = gn_range
    elif name == 'glass blur':
        params = gb_range
    for i in tqdm(range(each_time)):
        if i == 0:
            param = params[0]
        else:
            param = params[0] + i * (params[1] - params[0]) / (each_time - 1)
        
        if name == 'jpeg':
            param = int(param)
        else:
            param = round(param, 2)
        new_folder_path = os.path.join(base_path, f"{dataset}_{name}_{param}/")
        # new_folder_path = base_path + dataset + '_' + name + '_' + str(param) + '/'
        os.makedirs(new_folder_path, exist_ok=True)
        # folder = base_path + dataset + '/'
        folder = os.path.join(base_path, dataset)

        for root, dirs, files in os.walk(folder):
            for file in files:
                if file.endswith(".jpg"):
                    # Open the image
                    img_path = os.path.join(root, file)
                    img = Image.open(img_path)
                    img = img.convert('RGB')

                    if name == 'jpeg':
                        transformed_img = Image.fromarray(np.uint8(jpeg_compression(img, param)))
                    elif name == 'brightness':
                        transformed_img = Image.fromarray(np.uint8(brightness(img, param)))
                    elif name == 'contrast':
                        transformed_img = Image.fromarray(np.uint8(contrast(img, param)))
                    elif name == 'defocus blur':
                        transformed_img = Image.fromarray(np.uint8(defocus_blur(img, param)))
                    elif name == 'elastic blur':
                        transformed_img = Image.fromarray(np.uint8(elastic_transform(img, param)))
                    elif name == 'fog blur':
                        transformed_img = Image.fromarray(np.uint8(fog(img, param)))
                    elif name == 'frost blur':
                        transformed_img = Image.fromarray(np.uint8(frost(img, param)))
                    elif name == 'gaussian noise':
                        transformed_img = Image.fromarray(np.uint8(gaussian_noise(img, param)))
                    elif name == 'glass blur':
                        transformed_img = Image.fromarray(np.uint8(glass_blur(img, param)))

                    # Save the flipped image to the new directory with the same name
                    new_path = img_path.replace(dataset, dataset + '_' + name + '_' + str(param))
                    new_path = new_path.replace('.jpg', '.png')
                    new_folder = '/'.join(new_path.split('/')[:-1])
                    if not os.path.exists(new_folder):
                        os.makedirs(new_folder)
                    transformed_img_path = new_path
                    transformed_img.save(transformed_img_path)

