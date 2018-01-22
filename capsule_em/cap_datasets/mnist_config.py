import argparse

def settings():
    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset_dir", type=str, default="/Users/venice/dataset/MNIST_data")
    return parser.parse_args()
