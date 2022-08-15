from PIL import Image, ImageEnhance
import matplotlib.pyplot as plt
import os
import numpy as np

from PIL import ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True


def read_image(path, color_space=None):
    if color_space is None:
        img = Image.open(path)
    else:
        img = Image.open(path).convert(color_space)
    return img


def display_image(image, cmap='brg'):
    fig = plt.figure
    plt.imshow(image, cmap=cmap)


def display_images(images, labels=None, row_count=10, col_count=10, cmap='brg'):
    num = row_count * col_count
    fig, axes = plt.subplots(row_count, col_count,
                             figsize=(1.5 * row_count, 2 * col_count))
    for i in range(len(images)):
        ax = axes[i // row_count, i % col_count]
        ax.imshow(images[i], cmap=cmap)
        if labels != None:
            ax.set_title('Label: {}'.format(labels[i]))
    plt.tight_layout()
    plt.show()


def load_images_in_dir(path, color_space):
    images_path = get_all_files(path)
    images = [None] * len(images_path)
    for x in range(len(images_path)):
        images[x] = read_image(os.path.join(
            path, images_path[x]), color_space)
    return images


def create_dir_if_not_exists(path):
    if os.path.exists(path):
        return
    os.makedirs(path)


def get_all_files(path, get_full_path=False):
    if get_full_path:
        onlyfiles = [path + '/' + f for f in os.listdir(
            path) if os.path.isfile(os.path.join(path, f))]
    else:
        onlyfiles = [f for f in os.listdir(
            path) if os.path.isfile(os.path.join(path, f))]
    return onlyfiles


def get_new_image_dimen(image, new_dimen):
    width = image.width
    height = image.height
    new_width = 0
    new_height = 0
    aspect_ratio = width / height
    if width > height:
        new_width = new_dimen
        new_height = new_dimen / aspect_ratio
    else:
        new_height = new_dimen
        new_width = new_dimen * aspect_ratio
    return (new_width, new_height)


def pad_image(image, max_size):
    old_image_height, old_image_width, channels = image.shape
    # create new image of desired size and color (blue) for padding
    new_image_width = max_size
    new_image_height = max_size
    color = (0, 0, 0)
    result = np.full((new_image_height, new_image_width,
                      channels), color, dtype=np.uint8)
    # compute center offset
    x_center = (new_image_width - old_image_width) // 2
    y_center = (new_image_height - old_image_height) // 2
    # print(image)
    # copy img image into center of result image
    # result[y_center:y_center+old_image_height,x_center:x_center+old_image_width] = image
    result[0:old_image_height, 0:old_image_width] = image
    return result


def rgb2ycbcr(im):
    xform = np.array(
        [[.299, .587, .114], [-.1687, -.3313, .5], [.5, -.4187, -.0813]])
    ycbcr = im.dot(xform.T)
    ycbcr[:, :, [1, 2]] += 128
    return np.uint8(ycbcr)


def ycbcr2rgb(im):
    xform = np.array([[1, 0, 1.402], [1, -0.34414, -.71414], [1, 1.772, 0]])
    rgb = im.astype(np.float)
    rgb[:, :, [1, 2]] -= 128
    rgb = rgb.dot(xform.T)
    np.putmask(rgb, rgb > 255, 255)
    np.putmask(rgb, rgb < 0, 0)
    return np.uint8(rgb)


def set_image_contrast(image, factor):
    # image brightness enhancer
    enhancer = ImageEnhance.Contrast(image)
    return enhancer.enhance(factor)
