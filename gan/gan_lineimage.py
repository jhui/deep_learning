import os
import sys
import math
from glob import glob
from random import randint

import tensorflow as tf
import numpy as np
import cv2

from gan_utils import *

ROOT_DIR = os.path.join(os.sep, "Users", "venice", "dataset", "anime")

IMG_DIR = os.path.join(ROOT_DIR, "imgs")
ROUT_DIR = os.path.join(ROOT_DIR, "imgs_r")  # Cropped image to 256x256x3
LOUT_DIR = os.path.join(ROOT_DIR, "imgs_e")  # Lineart image

if __name__ == '__main__':

    if not os.path.exists(LOUT_DIR):
        os.makedirs(LOUT_DIR)
    if not os.path.exists(ROUT_DIR):
        os.makedirs(ROUT_DIR)

    data = glob(os.path.join(IMG_DIR, "*.jpg"))

    for i in range(len(data)):
        image_file = data[i]
        basename = os.path.basename(image_file)
        new_lfile = os.path.join(LOUT_DIR, basename)
        new_rfile = os.path.join(ROUT_DIR, basename)

        image = get_image(image_file)

        edge = cv2.adaptiveThreshold(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                     cv2.THRESH_BINARY, blockSize=9, C=2)
        edge = np.expand_dims(edge, 3)
        print(f"Saving {new_lfile}")
        cv2.imwrite(new_lfile, edge)
        cv2.imwrite(new_rfile, image)




