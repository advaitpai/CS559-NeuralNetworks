# Import file
import os
from string import ascii_lowercase

# Imports for LSTM 
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.utils.data as data_utils
from torch.optim.lr_scheduler import StepLR

# General imports
from tqdm import tqdm

# LSTM Model
class LSTMText(nn.Module):
    def __init__(self):
        super(LSTMText,self).__init__()
        self.in_size = 27
        self.hidden = 40
        self.layers = 2
        self.out_size = 27
        
        self.lstm = nn.LSTM(input_size = self.in_size, hidden_size=self.hidden, num_layers=self.layers,dropout=0.1)
        self.linear = nn.Linear(in_features = self.hidden, out_features=self.out_size)

    def forward(self,x):
        x, (h,c) = self.lstm(x)
        x = self.linear(x)
        return x,(h,c)


# Torch init settings
device_type = "cpu"
device = torch.device(device_type)
torch.manual_seed(2702) # Fixed Seed Value

# Extracting names from the file
names = []
with open('names.txt',mode='r') as f:
    names = f.readlines()
    names = [x.lower().replace('\n',"") for x in names]
print("Length of Name:",len(names))
chars = ['EON']
chars.extend([a for a in ascii_lowercase])

# One hot encoding for each character
char_encode = dict()
for i in range(len(chars)):
    temp = np.zeros(27)
    temp[i] = 1.0
    char_encode[chars[i]] = temp

# Function to create a name array of shape (11,27)
def convert_char_array(name):
    convert_arr = []
    for i in range(11):
        if i<len(name):
            convert_arr.append(char_encode[name[i]])
        else:
            convert_arr.append(char_encode['EON'])
    return np.array(convert_arr)

# Creating (xi,yi) pairs
train_x = [] # Temporary x list with numpy arrays
train_y = [] # Temporary y list with numpy arrays
for name in names:
    train_x.append(convert_char_array(name))
    train_y.append(convert_char_array(name[1:]))

train_X = torch.tensor(np.array(train_x),dtype=torch.float32)
train_Y = torch.tensor(np.array(train_y),dtype=torch.float32)

print("train_X Shape:",train_X.shape)
print("train_Y Shape:",train_Y.shape)

dataset = data_utils.TensorDataset(train_X, train_Y) # Creating (Xi,Yi) pairs
train_loader = data_utils.DataLoader(dataset,batch_size=100)

def train(model,device,train_loader,optimizer,loss_function):
    model.train()
    total_loss = 0

    for input, label in train_loader:
        input,label = input.to(device),label.to(device)
        optimizer.zero_grad()
        output,(h,c) = model(input)
        loss = loss_function(output.permute(0,2,1),torch.argmax(label,dim=2))
        loss.backward()
        optimizer.step()
        total_loss+=loss.item()
    # print("Training Loss:",total_loss/(100))
    return total_loss/(20)

# Model Initialization
model = LSTMText().to(device)
learning_rate = 0.05
gamma = 0.9
epochs = 500
optimizer = optim.Adam(model.parameters(),lr=learning_rate)
scheduler = StepLR(optimizer,step_size=10,gamma=gamma)

loss_function = nn.CrossEntropyLoss() # Used with Logits output (no activation)
training_loss = []
print("Training for Epochs:",)
for i in tqdm(range(1,epochs+1)):
    loss = train(model,device,train_loader,optimizer,loss_function)
    training_loss.append(loss)
    scheduler.step()
print("Final Training Loss:",training_loss[-1])

model_path = "0702-677368201-Pai.pt"
torch.save(model.state_dict(), model_path)

import matplotlib.pyplot as plt

plt.title("Epochs vs Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.plot([i for i in range(1,epochs+1)], training_loss, color ="green")
plt.show()