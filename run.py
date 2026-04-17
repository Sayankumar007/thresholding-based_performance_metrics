import argparse
from main import *

# Create an ArgumentParser object
parser = argparse.ArgumentParser()

# Add arguments
parser.add_argument("dataset", type=str, help="Dataset name..")
parser.add_argument("--epochs", type=int, default=100, help="Number of epochs for training...")
parser.add_argument("--dir", type=str, default=None, help="Directory of dataset (if available)...")


# Parse the arguments
args = parser.parse_args()


main(args.dataset, model_list, epoch=args.epochs, dataset_dir=args.dir)

