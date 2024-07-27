import os
import random
import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image

#data_folder = '/ssd_scratch/cvit/sayandebroy/dataset/driving_dataset/'
data_folder = '/home/sayandebroy/streering_wheel_angle_estimation/driving_dataset/'
data_file = os.path.join(data_folder, 'data.txt')

xs = []
ys = []

# points to the end of the last batch
train_batch_pointer = 0
val_batch_pointer = 0

# read data.txt
with open(data_file) as f:
    for line in f:
        xs.append(os.path.join(data_folder, line.split()[0]))
        ys.append(float(line.split()[1]) * np.pi / 180)

# get number of images
num_images = len(xs)

train_xs = xs[:int(len(xs) * 0.8)]
train_ys = ys[:int(len(xs) * 0.8)]

val_xs = xs[-int(len(xs) * 0.2):]
val_ys = ys[-int(len(xs) * 0.2):]

num_train_images = len(train_xs)
num_val_images = len(val_xs)

def load_image(file_path):
    image = Image.open(file_path)
    return image

def preprocess_image(image):
    # Resize the image to [66, 200] and normalize pixel values to [0, 1]
    transform = transforms.Compose([
        transforms.Resize((66, 200)),
        transforms.ToTensor(),
    ])
    return transform(image)

def LoadTrainBatch(batch_size):
    global train_batch_pointer
    x_out = []
    y_out = []
    for i in range(0, batch_size):
        img = load_image(train_xs[(train_batch_pointer + i) % num_train_images])
        x_out.append(preprocess_image(img))
        y_out.append([train_ys[(train_batch_pointer + i) % num_train_images]])
    train_batch_pointer += batch_size
    return torch.stack(x_out), torch.tensor(y_out)

def LoadValBatch(batch_size):
    global val_batch_pointer
    x_out = []
    y_out = []
    for i in range(0, batch_size):
        img = load_image(val_xs[(val_batch_pointer + i) % num_val_images])
        x_out.append(preprocess_image(img))
        y_out.append([val_ys[(val_batch_pointer + i) % num_val_images]])
    val_batch_pointer += batch_size
    return torch.stack(x_out), torch.tensor(y_out)
