import torch
import torch.nn as nn
import torch.optim as optim
import cv2
import math
import scipy.misc
import random
from network import SelfDrivingModel  # Import the model class from model_pytorch2

# Load the trained PyTorch model
model = SelfDrivingModel()
model.load_state_dict(torch.load("saved_model/model.pth"))  # Load the saved model

img = cv2.imread('recources/steering_wheel_image.jpg', 0)
rows, cols = img.shape

smoothed_angle = 0

# Read data.txt
xs = []
ys = []
with open("driving_dataset/data.txt") as f:
    for line in f:
        xs.append("driving_dataset/" + line.split()[0])
        ys.append(float(line.split()[1]) * scipy.pi / 180)

# Get number of images
num_images = len(xs)

i = math.ceil(num_images * 0.8)
print("Starting frame of video: " + str(i))

count = 0
error = 0
while (cv2.waitKey(5) != ord('q')):
    full_image = cv2.imread("driving_dataset/" + str(i) + ".jpg", cv2.IMREAD_COLOR)
    image = cv2.resize(full_image[-150:], (200, 66)) / 255.0
    image = torch.tensor(image.transpose(2, 0, 1), dtype=torch.float32).unsqueeze(0)  # Convert to tensor

    # Use the model for inference
    with torch.no_grad():
        model.eval()  # Set the model to evaluation mode
        predicted_degrees = model(image)  # Run the model to get the predicted steering angle
        predicted_degrees = float((predicted_degrees.item() * 180) / scipy.pi)
    random_float = random.uniform(-4.0, 4.0)
    actual_degree = ys[i]
    count = (count + 1)
    
    print("Steering angle: " + str(predicted_degrees) + " (pred)\t" + str(actual_degree) + " (actual)")
    cv2.imshow("frame", cv2.cvtColor(full_image, cv2.COLOR_BGR2RGB))
    smoothed_angle += 0.2 * pow(abs((predicted_degrees - smoothed_angle)), 2.0 / 3.0) * (predicted_degrees - smoothed_angle) / abs(
        predicted_degrees - smoothed_angle)
    M = cv2.getRotationMatrix2D((cols / 2, rows / 2), -smoothed_angle, 1)
    dst = cv2.warpAffine(img, M, (cols, rows))
    cv2.imshow("steering wheel", dst)
    i += 1

    #Calculate MSE
    if(predicted_degrees < 0):
        predicted_degrees = (predicted_degrees * (-1))
    if(actual_degree < 0):
        actual_degree = (actual_degree * (-1))
    error = (error + ((predicted_degrees - actual_degree)**2))

    if(i == num_images):
        break

cv2.destroyAllWindows()

#Calculate MSE
mse = (error / count)
print('\n Mean Squared Error = ', mse)
