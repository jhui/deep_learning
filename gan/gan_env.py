import os
from sys import platform

TINY = 1e-6

if platform == "linux":
    TOP_ROOT = os.path.join(os.sep, "home", "ubuntu", "gan")

    DATA_ROOT_DIR = os.path.join(TOP_ROOT, "dataset")
    APP_ROOT_DIR = os.path.join(TOP_ROOT, "app")
elif platform == "darwin":
    TOP_ROOT = os.path.join(os.sep, "Users", "venice")

    DATA_ROOT_DIR = os.path.join(TOP_ROOT, "dataset", "anime")
    APP_ROOT_DIR = os.path.join(TOP_ROOT, "Developer", "machine_learning", "gan")
else:
    print("Platform not supported")
    exit()

# DATA_ROOT_DIR : This directory must already exist with valid training dataset.

CHECKPOINT_DIR = os.path.join(APP_ROOT_DIR, "checkpoint", "tr")
RESULT_DIR = os.path.join(APP_ROOT_DIR, "results")

DIR_o = "imgs"
DIR_r = "imgs_r"
DIR_e = "imgs_e"
DIR_m = "imgs_m"

SAMPLE_o = "sample"
SAMPLE_r = "sample_r"
SAMPLE_e = "sample_e"
SAMPLE_m = "sample_m"

ORG_DIR = os.path.join(DATA_ROOT_DIR, DIR_o)  # /Users/venice/dataset/anime/imgs 512x512 images
IMG_DIR = os.path.join(DATA_ROOT_DIR, DIR_r)  # /Users/venice/dataset/anime/imgs_r 256x256 images
EDGE_DIR = os.path.join(DATA_ROOT_DIR, DIR_e)  # Edge image
META_DIR = os.path.join(DATA_ROOT_DIR, DIR_m)

SAMPLE_ORG_DIR = os.path.join(DATA_ROOT_DIR, SAMPLE_o)
SAMPLE_IMG_DIR = os.path.join(DATA_ROOT_DIR, SAMPLE_r)
SAMPLE_EDGE_DIR = os.path.join(DATA_ROOT_DIR, SAMPLE_e)
SAMPLE_META_DIR = os.path.join(DATA_ROOT_DIR, SAMPLE_m)

def prepare_dir():
    if not os.path.exists(DATA_ROOT_DIR):
        print(f"Data directory {DATA_ROOT_DIR} not exist")

    if not os.path.exists(APP_ROOT_DIR):
        print(f"App directory {APP_ROOT_DIR} not exist")

    if not os.path.exists(RESULT_DIR):
        os.makedirs(RESULT_DIR)
    if not os.path.exists(CHECKPOINT_DIR):
        os.makedirs(CHECKPOINT_DIR)
