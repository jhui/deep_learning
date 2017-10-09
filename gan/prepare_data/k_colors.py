import numpy as np
import cv2
import os
from glob import glob
from sys import platform
from collections import Counter
import operator
from gan_env import *

data = glob(os.path.join(IMG_DIR, "*.jpg"))

for filepath in data:
    basename = os.path.basename(filepath)
    new_file = os.path.join(META_DIR, basename + ".txt")

    img = cv2.imread(filepath)
    Z = img.reshape((-1,3))

    # convert to np.float32
    Z = np.float32(Z)

    # define criteria, number of clusters(K) and apply kmeans()
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    _ , label8 ,center8  = cv2.kmeans(Z, 8, None, criteria, 5, cv2.KMEANS_RANDOM_CENTERS)
    _ , label16 ,center16 = cv2.kmeans(Z, 16, None, criteria, 5, cv2.KMEANS_RANDOM_CENTERS)
    _, label32, center32 = cv2.kmeans(Z, 32, None, criteria, 5, cv2.KMEANS_RANDOM_CENTERS)

    center8  = np.uint8(center8)
    center16 = np.uint8(center16)
    center32 = np.uint8(center32)

    d8 = dict(Counter(label8.flatten().tolist()))
    s8 = sorted(d8.items(), key=operator.itemgetter(1), reverse=True)
    a8 = np.array([x[0] for x in s8])
    res8 = center8[a8]

    d16 = dict(Counter(label16.flatten().tolist()))
    s16 = sorted(d16.items(), key=operator.itemgetter(1), reverse=True)
    a16 = np.array([x[0] for x in s16])
    res16 = center16[a16]

    d32 = dict(Counter(label32.flatten().tolist()))
    s32 = sorted(d32.items(), key=operator.itemgetter(1), reverse=True)
    a32 = np.array([x[0] for x in s32])
    res32 = center32[a32]

    txt8  = ",".join([str(x) for x in res8.flatten()]) + "\n"
    txt16 = ",".join([str(x) for x in res16.flatten()]) + "\n"
    txt32 = ",".join([str(x) for x in res32.flatten()]) + "\n"

    with open(new_file, "w") as f:
        f.writelines([txt8, txt16, txt32])

    # res = center32[label32.flatten()]
    # res2 = res.reshape((img.shape))

    # cv2.imwrite(f"{basename}.in.jpg", img)
    # cv2.imwrite(f"{basename}.out.jpg", res2)

    print(basename)
