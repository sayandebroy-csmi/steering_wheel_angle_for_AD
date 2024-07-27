import os
import torch
import torch.nn as nn
import torch.optim as optim
import driving_data
from network import SelfDrivingModel  # Import the model class from model_pytorch2

LOGDIR = './save'
RESULT_FILE = './train_outputs.txt'  # File to store the results

# Check if CUDA is available and use 4 GPUs
if torch.cuda.is_available():
    device = torch.device("cuda")
    torch.cuda.set_device(0)  # Set the first GPU as the main device
    model = SelfDrivingModel().cuda()
    model = nn.DataParallel(model)  # Wrap the model with DataParallel to use all 4 GPUs
else:
    device = torch.device("cpu")
    model = SelfDrivingModel()

L2NormConst = 0.001

train_vars = model.parameters()  # Get the model's parameters

criterion = nn.MSELoss()
optimizer = optim.Adam(train_vars, lr=1e-4)

epochs = 30
batch_size = 128


# Create or open the results file for writing
with open(RESULT_FILE, 'w') as file:
    # train over the dataset about 30 times
    for epoch in range(epochs):
        for i in range(int(driving_data.num_images / batch_size)):
            xs, ys = driving_data.LoadTrainBatch(batch_size)
            xs = torch.tensor(xs).clone().detach().to(device)
            ys = torch.tensor(ys).clone().detach().to(device)

        
            optimizer.zero_grad()
            outputs = model(xs)  # Use 'model' to get outputs, not 'model_pytorch2'
            loss = criterion(outputs, ys) + sum(torch.norm(v) for v in train_vars) * L2NormConst
            loss.backward()
            optimizer.step()

            if i % 10 == 0:
                xs, ys = driving_data.LoadValBatch(batch_size)
                xs = torch.tensor(xs).clone().detach().to(device)
                ys = torch.tensor(ys).clone().detach().to(device)

                loss_value = criterion(model(xs), ys)  # Use 'model' to get outputs, not 'model_pytorch2'

                result_str = "Epoch: %d, Step: %d, Loss: %g\n" % (epoch, epoch * batch_size + i, loss_value)
                print(result_str)

                # Write the results to the file
                file.write(result_str)

            if i % batch_size == 0:
                if not os.path.exists(LOGDIR):
                    os.makedirs(LOGDIR)
                checkpoint_path = os.path.join(LOGDIR, "model.pth")
                torch.save(model.module.state_dict(), checkpoint_path)  # Save the model's state_dict for DataParallel
        print("Model saved in file: %s" % checkpoint_path)
        print("\n")

print("Run the command line:\n" \
      "--> tensorboard --logdir=./logs " \
      "\nThen open http://0.0.0.0:6006/ into your web browser")