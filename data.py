import os
from PIL import ImageFile
from tqdm import tqdm
import torchvision.datasets as datasets
from torch.utils.data import DataLoader, random_split
import torchvision.transforms as transforms

from global_vars import device

class DATA_LOADER():
    def __init__(self, name, batch, dataset_dir):
        
        self.name = name
        self.batch = batch
        self.dataset_dir = dataset_dir

        self.dataset_dict, self.train_loader, self.valid_loader, self.test_loader = self.__load_data(name, BATCH_SIZE=batch)
        
        for inputs, labels in tqdm(self.train_loader, desc="Loading Train Data: ", leave=False):
          inputs = inputs.to(device)
          labels = labels.to(device)
        print("Train Data Loaded Successfully..")
        for inputs, labels in tqdm(self.valid_loader, desc="Loading Validation Data: ", leave=False):
          inputs = inputs.to(device)
          labels = labels.to(device)
        print("Validation Data Loaded Successfully..")
        for inputs, labels in tqdm(self.test_loader, desc="Loading Test Data: ", leave=False):
          inputs = inputs.to(device)
          labels = labels.to(device)
        print("Test Data Loaded Successfully..\n")

    

    def __load_data(self, name, BATCH_SIZE=64):
        
        dataset_dir = self.dataset_dir

        transform = transforms.Compose([
            transforms.Resize(size = (256, 256)),
            transforms.RandomCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        ImageFile.LOAD_TRUNCATED_IMAGES = True

        # Create the ImageFolder dataset
        print(f"\nCreating Dataset .... {name} \n")
        train_dataset = datasets.ImageFolder(root=os.path.join(dataset_dir, "train"), transform=transform)
        valid_dataset = datasets.ImageFolder(root=os.path.join(dataset_dir, "valid"), transform=transform)
        test_dataset = datasets.ImageFolder(root=os.path.join(dataset_dir, "tests"), transform=transform)
        

        # building dictionary
        dataset_dict =  dict(name=name,
                            folder=f"Results_{name}",
                            input_size=(224, 224),
                            nclass= len(test_dataset.classes),
                            classes=test_dataset.classes)

        # Data Loaders
        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=8, pin_memory=True)
        valid_loader = DataLoader(valid_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=8, pin_memory=True)
        test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=8, pin_memory=True)
        
        return dataset_dict, train_loader, valid_loader, test_loader