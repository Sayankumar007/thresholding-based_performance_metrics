# Thresholding metrics for evaluating explainable AI models in disaster response
## Enhancing interpretability and model trustworthiness 

## Authors

**Sayan Kumar Bhowmick**  
Department of Computer Science and Engineering  
Jalpaiguri Government Engineering College, West Bengal, India  
📧 sayankr.contact@gmail.com  

**Asit Barman**  
Department of Information Technology  
Siliguri Institute of Technology, West Bengal, India  
📧 mtechitasit@gmail.com  

**Swalpa Kumar Roy**  
Department of Computer Science and Engineering  
Alipurduar Government Engineering and Management College, West Bengal, India   

---

## Affiliation
Developed under **Adaptive Intelligence LAB**  
An independent research group focused on AI & Machine Learning  

🔗 https://github.com/adaptive-intelligence-laboratory

---

## Abstract

Disasters present enormous obstacles for humanity and demand prompt and precise action. In this paper, we explore a critical intersection of disaster image classification and Explainable AI (XAI) with our proposed thresholding and automatic thresholding-based performance metric approaches. This study delves into six distinct models, evaluating their performance and shedding light on the crucial question of interpretability. Employing our proposed thresholding and automatic thresholding-based performance metric approaches with cutting-edge XAI techniques, we present not only classification results but also intricate Class Activation Maps, providing insights into where these models focus their attention during the process of classification. We consider our thresholding and automatic thresholding-based performance metric approaches to calculate the scores of precision, recall, and F-measure to establish more reliable and trustworthy classifiers. 

This research aims to enhance the transparency and trustworthiness of AI models deployed on disaster datasets, ultimately contributing to more effective and accountable disaster response efforts. The CAM efficiency varies from model to model due to the high intensity of the disaster dataset. Some models are good with GradCAM while some are good with ScoreCam. We use disaster datasets to validate the system efficacy and trustworthiness of a given model i.e. Vgg19, ResNet152, GoogleNet, DenseNet121, EfficientNet, and ConvNeXt along with considering Vision Transformer (ViT) and Swin Transformer (Swin) with not only score of Avg. Drop, Increase Confidence, and Win but also a score of precision, recall, and F-measure. 

The source code will be made available publicly at :
https://github.com/Sayankumar007/thresholding-based_performance_metrics.



---

## Repository Overview

This repository provides the implementation and experimental framework associated with the above-mentioned research work. It supports the reproducibility of the empirical findings by including:

- Training and evaluation of deep CNN-based models  
- Generation of saliency maps using multiple XAI techniques  
- Implementation of proposed Thresholding-based Performance Metrics  
- Comparative analysis across datasets, models, and explainability methods  

---

## Dataset Structure

The code expects the dataset to be organized as follows:

```text
root_dir/
│
├── train/
│   ├── class_1/
│   │   ├── img_1.jpg
│   │   ├── img_2.jpg
│   │   └── ...
│   ├── class_2/
│   └── ...
│
├── val/
│   ├── class_1/
│   ├── class_2/
│   └── ...
│
└── test/
    ├── class_1/
    ├── class_2/
    └── ...
```
> **Important:** Each split (train/val/test) should contain at least one class folder with valid image files. The code includes an integrity check that will verify missing splits, empty folders, or corrupted images.


---
## Installation

Make sure Python 3.8+ is installed. Recommended to use a virtual environment.

```bash
# Clone the repository
git clone https://github.com/Sayankumar007/thresholding-based_performance_metrics.git
cd thresholding-based_performance_metrics
```

## Environment & Reproducibility

This project has been developed and tested across multiple environments, including:

- Local workstation (Windows/Linux)
- Google Colab
- Kaggle Notebook environments

Due to differences in CUDA versions, PyTorch builds, and platform-specific configurations, minor version variations may exist across executions. However, the codebase is designed to remain compatible across PyTorch 2.x and standard scientific Python stacks.

### Python Version
Tested with:
- Python 3.8 – 3.10

### Dependency Installation

We provide two dependency files:

#### 1. General Compatibility Installation
For most users:

```bash
pip install -r requirements.txt
```

This installs platform-compatible versions of required packages.

#### 2. Exact Development Environment

For replicating the original development setup:

```bash
pip install -r requirements_exact.txt
```

This installs the exact package versions used during development.

### Notes on Reproducibility

Pretrained CNN backbones are obtained from torchvision. GPU execution depends on the CUDA version installed in the host system. Minor numerical differences may occur across hardware or CUDA versions. Random seeds should be fixed for deterministic experimentation where required. The dataset integrity check ensures structural consistency across environments.

Despite multi-platform testing, the experimental pipeline, evaluation metrics, and XAI visualizations demonstrate consistent qualitative and quantitative trends across environments.


---


## Usage

Run the `run.py` script to train/evaluate models:

```bash
python run.py <dataset_name> [--epochs N] [--dir dataset_dir]
```

#### Arguments:

`dataset_name` : Dataset name to run CNN Models

`--epochs` : Number of training epochs (default: 100)

`--dir` : Root directory of the dataset (must include train, val, test subfolders)

Example:
```bash
python run.py Disaster --epochs 50
```

## Citations

If you use this code or ideas in your research, please cite our work:

```bibTex
@article{bhowmick2025thresholding,
  title={Thresholding metrics for evaluating explainable AI models in disaster response: Enhancing interpretability and model trustworthiness},
  author={Bhowmick, Sayan Kumar and Barman, Asit and Roy, Swalpa Kumar},
  journal={Digital Signal Processing},
  volume={160},
  pages={105068},
  year={2025},
  publisher={Elsevier}
}
```

## Acknowledgements

Thanks to PyTorch and the research community for open-source implementations of CNN models and XAI methods.
Special thanks to the mentors and colleagues who provided feedback during implementation.

## License

This repository is for research purposes only. Please refer to the LICENSE file for detailed terms.
