import os
from glob import glob

import tensorflow as tf
import numpy as np
import cv2

from gan_layers import *
from gan_env import *

if __name__ == '__main__':

    if not os.path.exists(EDGE_DIR):
        os.makedirs(EDGE_DIR)
    if not os.path.exists(IMG_DIR):
        os.makedirs(IMG_DIR)

    data = glob(os.path.join(ORG_DIR, "*.jpg"))

    for i in range(len(data)):
        image_file = data[i]
        basename = os.path.basename(image_file)
        new_edge_file = os.path.join(EDGE_DIR, basename)
        new_image_file = os.path.join(IMG_DIR, basename)

        image = get_image(image_file)

        edge = cv2.adaptiveThreshold(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                     cv2.THRESH_BINARY, blockSize=9, C=2)
        edge = np.expand_dims(edge, 3)
        print(f"Saving {new_edge_file}")
        cv2.imwrite(new_edge_file, edge)
        cv2.imwrite(new_image_file, image)




