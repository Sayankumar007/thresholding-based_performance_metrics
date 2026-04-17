import torch
from cams import *

CLASSES = ('drought', 'earthquake', 'human_damage', 'infrastructure', 'land_slide', 'non_damage_buildings_street',
           'non_damage_human', 'non_damage_sea', 'non_damage_wildlife_forest', 'urban_fire', 'water_disaster',
           'wild_fire')


SEED = 42

TRAIN_SPLIT = 0.7
VALID_SPLIT = 0.2
TEST_SPLIT = 0.1

SOURCE_DIRECTORY = './disaster_data/'
REFACTORED_DIRECTORY = './refactored_data/'
TRAIN_DIRECTORY = './refactored_data/train/'
VALID_DIRECTORY = './refactored_data/valid/'
TEST_DIRECTORY = './refactored_data/tests/'


BATCH = 64

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


layer_map = dict(
    vgg19_bn = 'features',
    resnet152 = 'layer4',
    googlenet = 'inception5b',
    efficientnet_v2 = 'features',
    densenet201 = 'features',
    convnext = 'features'
)

cam_dict = dict(
    gradcam = GradCAM,
    gradcampp = GradCAMpp,
    layercam = LayerCAM,
    scorecam = ScoreCAM,
)


# model lists...
model_list = ["VGG19_bn", "ResNet152", "GoogLeNet", "DenseNet201", "EfficientNet_v2", "VIT_b", "VIT_l", "MaxVIT_t", "Swin_s", "ConvNeXt_b"]
# model_list = ["VIT_l", "ConvNeXt_b"]


# choosing random images manually for CAM generation..
paths = ['tests/drought/04_02_0009.png',
         'tests/earthquake/05_02_0002.png',
         'tests/human_damage/02_0052.png',
         'tests/infrastructure/05_01_0034.png',
         'tests/land_slide/04_01_0005.png',
         'tests/non_damage_buildings_street/06_02_0208.png',
         'tests/non_damage_human/06_01_0030.png',
         'tests/non_damage_sea/06_04_0006.png',
         'tests/non_damage_wildlife_forest/06_03_0015.png',
         'tests/urban_fire/01_01_0081.png',
         'tests/water_disaster/03_0049.png',
         'tests/wild_fire/01_02_0018.png'
        ]
