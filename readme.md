# Steering Wheel Angle Estimation for Autonomous Driving using ResNet-101 pretrained with ImageNet

![demo image](resources/semseg.gif)

## Installation

Setup Environment
```
conda create -n "driving" python=3.8

conda activate driving
```

Install Pytorch on GPU Platform
```
pip install torch==2.0.1 torchvision==0.15.2 --index-url https://download.pytorch.org/whl/cu118
```

Install other dependencies
```
pip install -r requirements.txt
```

## Download Dataset - [link](https://drive.google.com/file/d/1b_Fjxq-1LkpXAha2HkqgMsHrlkeE7VKC/view?usp=drive_link)


## Inference Demo
```
python test_and_visualize.py
```


## Training
```
python train.py
```
### The training was done on a server equipped with four NVIDIA GeForce RTX 3080 Ti GPUs.