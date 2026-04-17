import os
import torch
import torch.nn as nn
from torchvision import models

from global_vars import device

cnn_arch_values = ['vgg19_bn', 'resnet152', 'googlenet', 'densenet201', 'efficientnet_v2', 'convnext_b']
tr_arch_values = ['vit_b', 'vit_l', 'maxvit_t', 'swin_s']

arch_dict = dict(
    vgg19_bn = models.vgg19_bn,
    resnet152 = models.resnet152,
    googlenet = models.googlenet,
    densenet201 = models.densenet201,
    efficientnet_v2 = models.efficientnet_v2_s,
    vit_b = models.vit_b_32,
    vit_l = models.vit_l_32,
    maxvit_t = models.maxvit_t,
    swin_s = models.swin_s,
    convnext_b = models.convnext_base
)


# Baseline CNN Architectures...
class CNN_Model(nn.Module):
  def __init__(self, arch, num_classes, drp_rate=0.5):
    super(CNN_Model, self).__init__()
    
    self.num_classes = num_classes
    self.arch = arch
    self.drp_rate = drp_rate
    
    # Load the pretrained CNN Model...
    if self.arch in arch_dict:
      self.arch = arch_dict[self.arch](pretrained=True)
    else:
      raise ValueError(f"Invalid Architecture... Only these values are available -> {cnn_arch_values}")
    
    if'vgg' in arch:
      in_features = self.arch.classifier[0].in_features
      self.arch.classifier = nn.Sequential(
          nn.Linear(in_features, 512),
          nn.ReLU(),
          nn.Dropout(p = self.drp_rate),
          nn.Linear(512, self.num_classes))
      
    elif 'resnet' in arch or 'googlenet' in arch:
      in_features = self.arch.fc.in_features
      self.arch.fc = nn.Sequential(
          nn.Linear(in_features, 512),
          nn.ReLU(),
          nn.Dropout(p = self.drp_rate),
          nn.Linear(512, self.num_classes))
      
    elif 'efficientnet' in arch:
      in_features = self.arch.classifier[1].in_features
      self.arch.classifier = nn.Sequential(
          nn.Linear(in_features, 512),
          nn.ReLU(),
          nn.Dropout(p = self.drp_rate),
          nn.Linear(512, self.num_classes))
      
    elif 'convnext' in arch:
      in_features = self.arch.classifier[2].in_features
      self.arch.classifier[2] = nn.Sequential(
          nn.Linear(in_features, 512),
          nn.ReLU(),
          nn.Dropout(p = self.drp_rate),
          nn.Linear(512, self.num_classes))
      
    else:
      in_features = self.arch.classifier.in_features
      self.arch.classifier = nn.Sequential(
          nn.Linear(in_features, 512),
          nn.ReLU(),
          nn.Dropout(p = self.drp_rate),
          nn.Linear(512, self.num_classes))
      
  def forward(self, x):
    return self.arch(x)
        
    
  def load_model(self, model_path):
    if os.path.exists(model_path):
      print(f"\nLoading Saved Version of the Model {self.arch}....\n")
      if device == 'cuda':
        self.load_state_dict(torch.load(model_path)) 
      else:
        self.load_state_dict(torch.load(model_path, map_location='cpu')) 

    else :
      print(f"\nFOUND NO Previous Saved Version of the Model {self.arch}....\n")




# Implementation of Transformer-based architectures..
class TR_Model(nn.Module):
  def __init__(self, arch, num_classes, drp_rate=0.5):
    super(TR_Model, self).__init__()
    
    self.num_classes = num_classes
    self.arch = arch
    self.drp_rate = drp_rate
    
    # Load the pretrained CNN Model...
    if self.arch in arch_dict:
      self.arch = arch_dict[self.arch](pretrained=True)
    else:
      raise ValueError(f"Invalid Architecture... Only these values are available -> {tr_arch_values}")
    
    if 'maxvit' in arch:
      in_features = self.arch.classifier[3].in_features
      self.arch.classifier = nn.Sequential(
          nn.Linear(in_features, 512),
          nn.ReLU(),
          nn.Dropout(p = self.drp_rate),
          nn.Linear(512, self.num_classes))
    elif 'vit' in arch:
      in_features = self.arch.heads.head.in_features
      self.arch.heads.head = nn.Sequential(
          nn.Linear(in_features, 512),
          nn.ReLU(),
          nn.Dropout(p = self.drp_rate),
          nn.Linear(512, self.num_classes))
    elif 'swin' in arch:
      in_features = self.arch.head.in_features
      self.arch.head = nn.Sequential(
          nn.Linear(in_features, 512),
          nn.ReLU(),
          nn.Dropout(p = self.drp_rate),
          nn.Linear(512, self.num_classes))
    else:
      self.arch = self.arch
       
  def forward(self, x):
    return self.arch(x)
        
    
  def load_model(self, model_path):
    if os.path.exists(model_path):
      print(f"\nLoading Saved Version of the Model {self.arch}....\n")
      self.load_state_dict(torch.load(model_path)) 
    
    else :
      print(f"\nFOUND NO Previous Saved Version of the Model {self.arch}....\n")

