import argparse

def settings():
    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset_dir", type=str, default="/Users/venice/dataset/MNIST_data")
    parser.add_argument("--log_dir", type=str, default="./log/train", help="Directory with the log data.")
    parser.add_argument("--batch_size", type=int, default=24)

    parser.add_argument("--dataset", type=str, default=None)
    parser.add_argument("--seed", type=int, default=12345, help="Initial random seed")

    return parser.parse_args()




