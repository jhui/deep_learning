from glob import glob
import os
import numpy as np
import cv2
from gan_utils import *

ROOT_DIR = os.path.join(os.sep, "Users", "venice", "dataset", "anime")

IMG_DIR = os.path.join(ROOT_DIR, "imgs")

IMG_DIR2 = os.path.join(ROOT_DIR, "imgs_r")

GIMG_DIR = os.path.join(ROOT_DIR, "imgs_g")     # Gray image original
GIMG_DIR2 = os.path.join(ROOT_DIR, "imgs_gr")   # Gray image 256x256

if __name__ == '__main__':

    if not os.path.exists(GIMG_DIR):
        os.makedirs(GIMG_DIR)

    data = glob(os.path.join(IMG_DIR, "*.jpg"))
    for sample_file in data:
        img = get_orginal_image(sample_file)
        gray = True
        for i in range(len(img)):
            for j in range(len(img[1])):
                if not (img[i][j][0] == img[i][j][1] == img[i][j][2]):
                    gray = False
                    break
            if gray==False:
                break
        if gray:
            basename = os.path.basename(sample_file)
            new_file = os.path.join(GIMG_DIR, basename)
            os.rename(sample_file, new_file)

            f1 = os.path.join(IMG_DIR2, basename)
            f2 = os.path.join(GIMG_DIR2, basename)
            os.rename(f1, f2)

            print(f"{gray} {sample_file}")

