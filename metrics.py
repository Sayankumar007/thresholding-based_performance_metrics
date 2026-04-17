import os
import torch
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
import torchvision.transforms as transforms

from cams import *
from global_vars import *
from models import CNN_Model, TR_Model
   
def metrics(model, dataloader, cam_model_list=[]):

  for idx, data in enumerate(tqdm(dataloader)):
    input_ = data[0]
    input_ = input_.squeeze()
    input_ = input_.unsqueeze(0)
    
    input_ = input_.to(device)

    for cam_model in cam_model_list:
        cam_model.metrics(model, input_)

  for cam_model in cam_model_list:
      cam_model.percentize(len(dataloader))




def win_metrics(predicted_confidence_dict, avg_win, avg_drop, avg_increase, names_exclude=[], filename='Model_metrics_CAM.csv'):
  keys = list(predicted_confidence_dict.keys())
  avg_win_keys = list(avg_win.keys())
  num_total = len(predicted_confidence_dict[keys[0]])
  names = ['gradcam','gradcampp','layercam','scorecam']
  max_conf = -1000
  max_conf_item = 'k'

  for item in avg_win:
    avg_win[item] = 0

  for idx in range(num_total):
    for index, item in enumerate(names):
       if (item in names_exclude):
          continue
#        print(item)
       if predicted_confidence_dict[item][idx] > max_conf:
          max_conf = predicted_confidence_dict[item][idx]
          max_conf_item = item

    avg_win[max_conf_item] += 1
    max_conf = -1000
    max_conf_item = ''

  for item in avg_win:
    avg_win[item] = avg_win[item]*100/num_total

  average_drop = []
  average_increase = []
  average_win = []
  names_ = []
  for item in names:
    if (item in names_exclude):
        continue
    names_.append(item)
    average_drop.append(avg_drop[item])
    average_increase.append(avg_increase[item])
    average_win.append(avg_win[item])

  data = {'name':names_,'Average_drop%':average_drop, '%Increase_in_Confidence':average_increase,'Win%':average_win}
  df = pd.DataFrame(data)
  df.to_csv(filename)
  print(filename)
  print(df)


def CAM_eval(model_list, dataset_folder, dataset_dir=''):
      
  dataset = datasets.ImageFolder(root=os.path.join(dataset_dir, "test"), transform=transform)
  data_loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=8, pin_memory=True)
  
  for arch in model_list:
    print(f"\nBuilding Model..")
    arch = arch.lower()
    if 'vit' in arch or 'swin' in arch:
        raise ValueError(f"The Model {arch} is not a CNN based Model..")
    else:
        model = CNN_Model(arch, 12).to(device)

    model_path = f".\Results_{dataset_folder}\{arch}\checkpoint__{dataset_folder}_{arch}.pth"
    model.load_model(model_path)
    model_dict = dict(type=arch, arch=model.arch, layer_name=layer_map[arch],input_size=(224, 224))
    
    # Check if the directory exists, if not, create it..
    print("Creating a Saving Directory... ")
    save_dir = f'./Standard_Eval'
    # save_dir = f'.\CAMs\{arch}'
    os.makedirs(save_dir, exist_ok=True)
    
    gradcam = GradCAM(model_dict)
    gradcampp = GradCAMpp(model_dict)
    layercam = LayerCAM(model_dict)
    scorecam = ScoreCAM(model_dict)
    cam_list = [gradcam, gradcampp, layercam, scorecam]
    
    transform =transforms.Compose([
        transforms.Resize((256,256)),
        transforms.RandomCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225])
    ])
    
    avg_drop = {'gradcam':0,'gradcampp':0,'layercam':0,'scorecam':0}
    avg_increase = {'gradcam':0,'gradcampp':0,'layercam':0,'scorecam':0}
    avg_win = {'gradcam':0,'gradcampp':0,'layercam':0,'scorecam':0}
    predicted_confidence = {'gradcam':[],'gradcampp':[],'layercam':[],
                                'scorecam':[]}
    
    
    
    metrics(model, data_loader, cam_list)
    
    predicted_confidence['gradcam'] = gradcam.predicted_confidence_cam_list[:]
    avg_drop['gradcam'] = gradcam.avg_drop
    avg_increase['gradcam'] = gradcam.avg_increase
    del gradcam

    predicted_confidence['gradcampp'] = gradcampp.predicted_confidence_cam_list[:]
    avg_drop['gradcampp'] = gradcampp.avg_drop
    avg_increase['gradcampp'] = gradcampp.avg_increase
    del gradcampp

    predicted_confidence['layercam'] = layercam.predicted_confidence_cam_list[:]
    avg_drop['layercam'] = layercam.avg_drop
    avg_increase['layercam'] = layercam.avg_increase
    del layercam

    predicted_confidence['scorecam'] = scorecam.predicted_confidence_cam_list[:]
    avg_drop['scorecam'] = scorecam.avg_drop
    avg_increase['scorecam'] = scorecam.avg_increase
    del scorecam


    win_metrics(predicted_confidence_dict=predicted_confidence, avg_win=avg_win, avg_drop=avg_drop, avg_increase=avg_increase, names_exclude=[], filename=f'./{save_dir}/standard_metrics_{arch}.csv')

    
    print('Completed Successfully')
            
            
            



def log_output(save_dir, name, cam_name, threshold, mean_v, mini, maxi):
    f = open(f'./{save_dir}/Proposed_metrics_{name}.txt', 'a')
    f.write(f"Model : {name} ----- CAM : {cam_name} ----- Threshold : {threshold} (for original) for saliency map only those regions are taken that are greater than 0.5 \nThe Result is ->\n")
    sentence1 = f'\nMean_Precision: {mean_v[0]}\nMinimum Precision: {mini[0]}\nMaximum Precision: {maxi[0]}\n'
    f.write(sentence1)
    
    sentence2 = f'\nMean_Recall: {mean_v[1]}\nMinimum Recall: {mini[1]}\nMaximum Recall: {maxi[1]}\n'
    f.write(sentence2)
    
    sentence3 = f'\nMean_F_measure: {mean_v[2]}\nMinimum F_measure: {mini[2]}\nMaximum F_measure: {maxi[2]}\n'
    f.write(sentence3)
    
    f.write("\n\n\n")
    
    f.close()
    
    
    csv_file_path = os.path.join(save_dir, f'Proposed_metrics(mean)_{name}.csv')
    columns = ['cam', 'high_precision', 'high_recall', 'high_f_score', 'low_precision', 'low_recall', 'low_f_score']
    
    # Check if the CSV file exists
    if os.path.exists(csv_file_path):
        # Load the existing CSV file into a DataFrame
        df = pd.read_csv(csv_file_path)

        if cam_name in df['cam'].values:
            # If the CAM name is present, update the row
            idx = df[df['cam'] == cam_name].index[0]
            df.at[idx, f'{threshold}_precision'] = mean_v[0]
            df.at[idx, f'{threshold}_recall'] = mean_v[1]
            df.at[idx, f'{threshold}_f_score'] = mean_v[2]
        else:
            # If the CAM name is not present, create a new row
            new_row = {'cam': cam_name,
                       f'{threshold}_precision': mean_v[0],
                       f'{threshold}_recall': mean_v[1],
                       f'{threshold}_f_score': mean_v[2]}
            df = df.append(new_row, ignore_index=True)
    else:
        # If the CSV file doesn't exist, create a new DataFrame
        data = {'cam': [cam_name],
                f'{threshold}_precision': [mean_v[0]],
                f'{threshold}_recall': [mean_v[1]],
                f'{threshold}_f_score': [mean_v[2]]}
        df = pd.DataFrame(data, columns=columns)

    # Save the DataFrame back to the CSV file
    df.to_csv(csv_file_path, index=False)
    
    
    

def CAM_eval_new(model_list, dataset_folder, dataset_dir):
      
  dataset = datasets.ImageFolder(root=os.path.join(dataset_dir, "test"), transform=transform)
  data_loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=8, pin_memory=True)
  
  for arch in model_list:
    print(f"\nBuilding Model..")
    arch = arch.lower()
    if 'vit' in arch or 'swin' in arch:
        raise ValueError(f"The Model {arch} is not a CNN based Model..")
    else:
        model = CNN_Model(arch, 12).to(device)

    model_path = f".\Results_{dataset_folder}\{arch}\checkpoint__{dataset_folder}_{arch}.pth"
    model.load_model(model_path)
    model_dict = dict(type=arch, arch=model.arch, layer_name=layer_map[arch],input_size=(224, 224))
    
    # Check if the directory exists, if not, create it..
    print("Creating a Saving Directory... ")
    save_dir = f'./Proposed_Eval'

    os.makedirs(save_dir, exist_ok=True)
    transform = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize(mean = [0.485, 0.456, 0.406],
                             std = [0.229, 0.224, 0.225])
    ])

    to_grayscale = transforms.Grayscale(num_output_channels = 1)
    
    
    
    thresholds = ['high', 'low']
    
    gradcam = GradCAM(model_dict)
    gradcampp = GradCAMpp(model_dict)
    layercam = LayerCAM(model_dict)
    scorecam = ScoreCAM(model_dict)
    cams = {'gradcam':gradcam, 'gradcampp':gradcampp, 'layercam': layercam, 'scorecam':scorecam}
    # cams = {'scorecam':scorecam}

    
    for threshold in thresholds:
        plt.figure(figsize=(20,6))
        
        for cam_name, cam_model in tqdm(cams.items(), desc='CAMs : ', leave = False):
            count = 0
            precisions=[]
            recalls = []
            f_scores = []
            
            for idx, img in enumerate(data_loader):
                data = img[0]
                gray_image = to_grayscale(data)
                
                data = data.to(device)
      
                s_map = cam_model(data).cpu()

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
                
                # binarize the normalized version of original image..
                if threshold == 'low':
                  binarized_image = (gray_image_normalized < 0.5).astype(np.uint8)
                else:
                  binarized_image = (gray_image_normalized >= 0.5).astype(np.uint8)

                # binarize the normalized version of saliency map..
                binarized_s_map = (normalized_s_map >= 0.5).astype(np.uint8)

                # intersection using logical AND
                intersection = np.logical_and(binarized_image, binarized_s_map).astype(np.uint8)
                
                # count the number of pixels in the intersection region
                intersection_pixel_count = np.sum(intersection)
                # print('\n\n', intersection_pixel_count, '\n')
                
                recall = intersection_pixel_count / np.sum(binarized_image)
                precision = intersection_pixel_count / np.sum(binarized_s_map)
                
                precisions.append(precision)
                recalls.append(recall)
                
                beta =0.3
                f_measure = ((1+beta**2) * precision * recall) / ((beta**2) * precision + recall)
                
                f_scores.append(f_measure)
                
                count = count+1
                if count == 100:
                    break
                
            recalls = np.array(recalls)
            precisions = np.array(precisions)
            
            # sort the data by recall values for visualization purpose
            sorted_indices = np.argsort(recalls)
            sorted_precisions = precisions[sorted_indices]
            sorted_recalls = recalls[sorted_indices]
            
            plt.plot(sorted_recalls, sorted_precisions, label=cam_name)
            mean_precision = sum(precisions) / len(precisions)
            mean_recall = sum(recalls) / len(recalls)
            mean_f = sum(f_scores) / len(f_scores)
            
            min_precision = min(precisions)
            max_precision = max(precisions)
            
            min_recall = min(recalls)
            max_recall = max(recalls)
            
            min_f = min(f_scores)
            max_f = max(f_scores)
            
            log_output(save_dir, arch, cam_name, threshold, [mean_precision, mean_recall, mean_f], [min_precision, min_recall, min_f], [max_precision, max_recall, max_f])
            
        # adding title and labels
        plt.title(f'Precision-Recall Curve for {arch} (threshold : {threshold})')
        plt.xlabel('Recalls')
        plt.ylabel('Precisions')
        plt.legend()
        
        # save the final plot
        plt.savefig(f'./{save_dir}/Precision_recall_curves_{arch}_{threshold}.png')
        
        # show the plot
        plt.show()
