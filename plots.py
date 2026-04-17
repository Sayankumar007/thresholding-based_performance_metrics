import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
import cv2
import torchvision.transforms as transforms
import torch.nn.functional as F
from torchvision.utils import save_image
from PIL import Image

from cams import *
from global_vars import *
from models import CNN_Model




def reverse_normalize(x, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
    x[:, 0, :, :] = x[:, 0, :, :] * std[0] + mean[0]
    x[:, 1, :, :] = x[:, 1, :, :] * std[1] + mean[1]
    x[:, 2, :, :] = x[:, 2, :, :] * std[2] + mean[2]
    return x


def visualize(img, cam):
    """
    Synthesize an image with CAM to make a result image.
    Args:
        img: (Tensor) shape => (1, 3, H, W)
        cam: (Tensor) shape => (1, 1, H', W')
    Return:
        synthesized image (Tensor): shape =>(1, 3, H, W)
    """

    _, _, H, W = img.shape
    # print(cam.shape)
    cam = F.interpolate(cam, size=(H, W), mode='bilinear', align_corners=False)
    cam = 255 * cam.squeeze()
    heatmap = cv2.applyColorMap(np.uint8(cam), cv2.COLORMAP_JET)
    heatmap = torch.from_numpy(heatmap.transpose(2, 0, 1))
    heatmap = heatmap.float() / 255
    b, g, r = heatmap.split(1)
    heatmap = torch.cat([r, g, b])

    result = heatmap + img.cpu()
    result = result.div(result.max())

    return result


def plot_CAMS_CNNs(model_list, dataset_folder, dataset_dir = "/content/drive/MyDrive/Swalpa_sir/Disaster/assets/refactored_data"):
    
    for arch in model_list:
        print(f"\nBuilding Model..")
        arch = arch.lower()
        if 'vit' in arch or 'swin' in arch:
            raise ValueError(f"The Model {arch} is not a CNN based Model..")
        else:
            model = CNN_Model(arch, 12).to(device)

        model_path = f"./Results_{dataset_folder}/{arch}/checkpoint__{dataset_folder}_{arch}.pth"
        model.load_model(model_path)
        model_dict = dict(type=arch, arch=model.arch, layer_name=layer_map[arch],input_size=(224, 224))
        
        # Check if the directory exists, if not, create it..
        print("Creating a Saving Directory... ")
        save_dir = f'./CAMs/{arch}'
        os.makedirs(save_dir, exist_ok=True)

        for key in cam_dict:
            wrapped_model = cam_dict[key](model_dict)
            transform =transforms.Compose([
                transforms.Resize((256,256)),
                transforms.RandomCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                        std=[0.229, 0.224, 0.225])
            ])
            
            for path in paths:
                path = os.path.join(dataset_dir, path)
                img = Image.open(path)
                tensor = transform(img)
                
                # reshape 4D tensor (N, C, H, W)
                tensor = tensor.unsqueeze(0)
                tensor = tensor.to(device)
                
               
                print(f"\ngenerating cam.. for model --> {arch}_{key}")
                cam = wrapped_model(tensor)
                img = reverse_normalize(tensor)
                heatmap = visualize(img, cam.cpu())
                    
                # Convert the tensor back to a PIL image
                img = img.squeeze(0)  # Remove batch dimension
                img = transforms.ToPILImage()(img.cpu())

                # Save the original image after transformation
                org_dir = os.path.join(save_dir, f"{path.split('/')[-2]}_{path.split('/')[-1].split('.png')[0]}_original.png")
                if not os.path.exists(org_dir):
                    img.save(org_dir)
                    print("Created a Processed Original File..")
                
                cam_dir = os.path.join(save_dir, f"{path.split('/')[-2]}_{path.split('/')[-1].split('.png')[0]}_{arch}_{key}.png")
                if not os.path.exists(cam_dir):
                    save_image(heatmap, cam_dir)
                    


def save_it(image, dir):
  # Plot the image
  plt.imshow(image, cmap='gray', vmin=0, vmax=1)
  plt.axis('off')  # Turn off axis labels and ticks
  
  # Save the image
  plt.savefig(dir, bbox_inches='tight', pad_inches=0, dpi=400)
  plt.close()  # Close the plot to free up memory              



def vizualization(arch, cam, dataset_folder, dataset_dir = "/content/drive/MyDrive/Swalpa_sir/Disaster/assets/refactored_data"):
    
  print(f"\nBuilding Model..")
  arch = arch.lower()
  if 'vit' in arch or 'swin' in arch:
      raise ValueError(f"The Model {arch} is not a CNN based Model..")
  else:
      model = CNN_Model(arch, 12).to(device)

  model_path = f"./Results_{dataset_folder}/{arch}/checkpoint__{dataset_folder}_{arch}.pth"
  model.load_model(model_path)
  model_dict = dict(type=arch, arch=model.arch, layer_name=layer_map[arch],input_size=(224, 224))
  
  # Check if the directory exists, if not, create it..
  print("Creating a Saving Directory... ")
  save_dir = f'./Algo_VIZ/{arch}_{cam}'
  os.makedirs(save_dir, exist_ok=True)


  wrapped_model = cam_dict[cam.lower()](model_dict)
  transform =transforms.Compose([
      transforms.Resize((256,256)),
      transforms.RandomCrop(224),
      transforms.ToTensor(),
      transforms.Normalize(mean=[0.485, 0.456, 0.406],
                              std=[0.229, 0.224, 0.225])
  ])
  to_grayscale = transforms.Grayscale(num_output_channels = 1)

  for path in tqdm(paths):
      path = os.path.join(dataset_dir, path)
      imgname = f"{path.split('/')[-2]}_{path.split('/')[-1].split('.png')[0]}"
      img = Image.open(path)
      tensor = transform(img)
      gray_image = to_grayscale(tensor)
      
      # reshape 4D tensor (N, C, H, W)
      tensor = tensor.unsqueeze(0)
      data = tensor.to(device)
      
      s_map = wrapped_model(data).cpu()
      
      # print(gray_image.shape)
      # print(s_map.shape)

      s_map = s_map[0, 0, :, :].numpy()
      gray_image = gray_image[ 0, :, :].numpy()
      # print(gray_image.shape)
      # print(s_map.shape)
      
      # Normalize the Saliency map Values to 0-255
      s_map_min = s_map.min()
      s_map_max = s_map.max()
      
      # If the image has a single unique value, avoid division by zero
      if s_map_min == s_map_max:
          normalized_s_map = np.zeros_like(s_map, dtype = np.uint8)
      else:
          normalized_s_map = 255 * (s_map - s_map_min) / (s_map_max - s_map_min)
          normalized_s_map = normalized_s_map.astype(np.uint8)
          
      normalized_s_map = normalized_s_map/255
      # print(normalized_s_map)

      save_it(normalized_s_map, os.path.join(save_dir, f"{imgname}_{arch}_{cam}_smap.png"))
      
      # normalize the gray image values to 0-255
      image_min = gray_image.min()
      image_max = gray_image.max()
      
      # If the image has a single unique value, avoid division by zero
      if image_min == image_max :
          normalized_image = np.zeros_like(gray_image, dtype=np.uint8)
      else:
          normalized_image = 255 * (gray_image - image_min) / (image_max - image_min)
          normalized_image = normalized_image.astype(np.uint8)
          
      gray_image_normalized = normalized_image/255
      save_it(gray_image_normalized, os.path.join(save_dir, f"{imgname}_{arch}_{cam}_grey.png"))
      
      # binarize the normalized version of multiplied image..
      low_binarized_image = (gray_image_normalized < 0.5).astype(np.uint8)
      save_it(low_binarized_image, os.path.join(save_dir, f"{imgname}_{arch}_{cam}_grey_low_binary.png"))

      # binarize the normalized version of multiplied image..
      high_binarized_image = (gray_image_normalized >= 0.5).astype(np.uint8)
      save_it(high_binarized_image, os.path.join(save_dir, f"{imgname}_{arch}_{cam}_grey_high_binary.png"))

      # binarize the normalized version of saliency map..
      binarized_s_map = (normalized_s_map >= 0.5).astype(np.uint8)
      save_it(binarized_s_map, os.path.join(save_dir, f"{imgname}_{arch}_{cam}_smap_binary.png"))

      # intersection using logical AND
      low_intersection = np.logical_and(low_binarized_image, binarized_s_map).astype(np.uint8)
      save_it(low_intersection, os.path.join(save_dir, f"{imgname}_{arch}_{cam}__low_intersection.png"))

      high_intersection = np.logical_and(high_binarized_image, binarized_s_map).astype(np.uint8)
      save_it(high_intersection, os.path.join(save_dir, f"{imgname}_{arch}_{cam}__high_intersection.png"))


                
# model_list = ["VGG19_bn", "ResNet152", "GoogLeNet", "DenseNet201", "EfficientNet_v2", "ConvNeXt_b"]
# model_list = ["VGG19_bn"]
# model_list = ["ResNet152", "GoogLeNet", "DenseNet201", "EfficientNet_v2", "ConvNeXt_b"]
# plot_CAMS_CNNs(model_list, "Disaster")
# vizualization('Resnet152', 'gradcampp', 'Disaster')