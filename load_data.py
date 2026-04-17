# # Please Download assets from the GitHub repository
# ! wget https://github.com/tariqshaban/disaster-classification-with-xai/trunk/assets

import os
import shutil
import re
import random
from PIL import Image
from glob import glob
import torch
import numpy as np
import torchvision.transforms as transforms
from torchvision.utils import save_image
from global_vars import *
    
    

def prime_dataset():

    # Read Each Image With its Class Label
    images = []
    folders = CLASSES

    for folder in folders:
        t = folder
        x = os.listdir(f'{SOURCE_DIRECTORY}/{t}')
        for i in x:
            for j in re.split(r'[-;,\t\s]\s*', i):
                if j == '':
                    continue
                images.append({'Class': t, 'Image': j})

    # Partition Images into Training, Validation, and Testing
    for c in folders:
        os.makedirs(f'{TRAIN_DIRECTORY}{c}', exist_ok=True)
        os.makedirs(f'{VALID_DIRECTORY}{c}', exist_ok=True)
        os.makedirs(f'{TEST_DIRECTORY}{c}', exist_ok=True)

    counter = 0
    for c in folders:

        try:
            numOfFiles = len(next(os.walk(f'{SOURCE_DIRECTORY}{c}/'))[2])
            for files in random.sample(glob(f'{SOURCE_DIRECTORY}{c}/*'), int(numOfFiles * TRAIN_SPLIT)):
                shutil.move(files, f'{TRAIN_DIRECTORY}{c}')

            for files in random.sample(glob(f'{SOURCE_DIRECTORY}{c}/*'), int(numOfFiles * VALID_SPLIT)):
                shutil.move(files, f'{VALID_DIRECTORY}{c}')

            for files in glob(f'{SOURCE_DIRECTORY}{c}/*'):
                shutil.move(files, f'{TEST_DIRECTORY}{c}')
        except StopIteration:
            print(f"No files found in directory: {SOURCE_DIRECTORY}{c}")

        counter += 1

    shutil.rmtree(SOURCE_DIRECTORY)
    
    
    

def augment_class(input_dir):
    # List all image files in the input directory
    image_files = [f for f in os.listdir(input_dir) if f.endswith(('.jpg', '.png', '.jpeg'))]

    # Define the number of augmentations per image
    num_augmentations = 25  # You can adjust this number as needed

    # Define the transformations to apply for augmentation
    transform = transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(20),
        # transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
        transforms.ToTensor(),  # Convert the image to a PyTorch tensor
    ])

    # Create a dataset from the input folder
    # dataset = ImageFolder(root=input_dir, transform=transform)

    # # Create a data loader with batch size 1 (each image is processed individually)
    # data_loader = DataLoader(dataset, batch_size=1, shuffle=False)

    # Loop through the dataset and save augmented images
    for i, (image_file) in enumerate(image_files):
        image_path = os.path.join(input_dir, image_file)
        image = Image.open(image_path)

        for j in range(num_augmentations):
            augmented_image = transform(image)
            save_image(augmented_image, os.path.join(input_dir, f'augmented_{i}_{j}.png'))

    print(f"Augmentation complete. {num_augmentations} augmented images generated for each existing earthquake image.")
    


os.environ['PYTHONHASHSEED'] = str(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
np.random.seed(SEED)
# In this dataset number of images for the "earthquake" class is too low, so we have generated augmented data for this class.. 
# augmentation can be done for other classes or entire dataset too..

erthqck_path = './assets/refactored_data/train/earthquake'
os.listdir(erthqck_path)
prime_dataset()
augment_class(erthqck_path)