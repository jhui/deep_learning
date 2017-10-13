import numpy as np
import cv2
from gan_env import *

def get_batches(data, size, offset=0, sampling=False, read_color_hints=True):
    """
    Get a batch of data
    :param data: np.array contains all the filenames of the data
    :param size: the size of the batch
    :param offset: offset to data
    :return: the image data, the edge data, the color hints
    """
    start = offset
    stop = offset + size

    r_data = data[start:stop]
    if sampling:
        e_data = [sample_file.replace(SAMPLE_r, SAMPLE_e) for sample_file in data[start:stop]]
        m_data = [sample_file.replace(SAMPLE_r, SAMPLE_m) for sample_file in data[start:stop]]
    else:
        e_data = [sample_file.replace(DIR_r, DIR_e) for sample_file in data[start:stop]]
        m_data = [sample_file.replace(DIR_r, DIR_m) for sample_file in data[start:stop]]

    if len(e_data) == 0 or len(m_data) == 0:
        print(f"No JPG image find in {IMG_DIR}")
        exit()

    img = np.array([get_orginal_image(sample_file) for sample_file in r_data])
    img = img / 255.0

    edge = np.array([get_orginal_image(sample_file, color=False) for sample_file in e_data])
    edge = edge / 255.0
    edge = np.expand_dims(edge, 3)

    color_hints = None
    if read_color_hints:
        color_hints = np.array([get_meta_colors(sample_file) for sample_file in m_data])  # (N, 16*3)
        color_hints = color_hints / 255.0

    return img, edge, color_hints

def override_demo_image(data, size):
    """
    Override the first 8 images as the demo images. Note, this is hardcoded to change the first 8 images.
    """
    data[:size] = [os.path.join(IMG_DIR, "51.jpg"), os.path.join(IMG_DIR, "1462.jpg"),
                   os.path.join(IMG_DIR, "13712.jpg"), os.path.join(IMG_DIR, "2868.jpg")]
    data[size:size * 2] = [os.path.join(IMG_DIR, "30018.jpg"), os.path.join(IMG_DIR, "8950.jpg"),
                           os.path.join(IMG_DIR, "30324.jpg"), os.path.join(IMG_DIR, "13169.jpg")]
    return data

def get_orginal_image(image_path, color=True):
    return np.array(imread(image_path, color))

def get_image(image_path):
    return transform(imread(image_path))

def transform(image, npx=512, is_crop=True):
    cropped_image = cv2.resize(image, (256, 256))

    return np.array(cropped_image)

def get_meta_colors(filepath, index=1):
    with open(filepath + '.txt', "r") as f:
        s = f.readlines()
        base_colors = [[int(v) for v in txt.split(",")] for txt in s]
        base_colors = np.array(base_colors[index])
    return base_colors

def imread(path, color=True):
    flag = cv2.IMREAD_COLOR
    if color==False:
        flag = cv2.IMREAD_GRAYSCALE

    readimage = cv2.imread(path, flag)
    return readimage


def merge_color(images, size):
    """
        Merge size[0] x size[1] RGB images into one single image to display result.
    """
    h, w = images.shape[1], images.shape[2]
    img = np.zeros((h * size[0], w * size[1], 3))

    for idx, image in enumerate(images):
        i, j = divmod(idx, size[1])
        img[j * h:j * h + h, i * w:i * w + w, :] = image

    return img


def merge(images, size):
    """
        Merge size[0] x size[1] images into one single image to display result.
    """
    h, w = images.shape[1], images.shape[2]
    if (images.shape[3] in (3, 4)):
        c = images.shape[3]
        img = np.zeros((h * size[0], w * size[1], c))
        for idx, image in enumerate(images):
            i, j = divmod(idx, size[1])
            img[j * h:j * h + h, i * w:i * w + w, :] = image
        return img
    elif images.shape[3] == 1:
        img = np.zeros((h * size[0], w * size[1]))
        for idx, image in enumerate(images):
            i, j = divmod(idx, size[1])
            img[j * h:j * h + h, i * w:i * w + w] = image[:, :, 0]
        return img
    else:
        raise ValueError('in merge(images,size) images parameter must have dimensions: HxW or HxWx3 or HxWx4')


def ims(name, img):
    print("Saving " + name)
    cv2.imwrite(name, img * 255)