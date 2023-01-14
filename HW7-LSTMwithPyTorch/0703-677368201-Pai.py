# General Imports
from string import ascii_lowercase
import random

# Imports for LSTM 
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.utils.data as data_utils
from torch.optim.lr_scheduler import StepLR

# Fixed Seed Value
torch.manual_seed(2702)

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

# Character List
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
    for i in range(len(name)):
        if i<len(name):
            convert_arr.append(char_encode[name[i]])
        else:
            convert_arr.append(char_encode['EON'])
    return np.array(convert_arr)

def choice_prob(x): # x is the 'char' array
    output_tensor = x[-1]
    vals, ind = torch.topk(output_tensor,4) # Get top 4 output predicted words
    index = random.choices(ind.tolist(),weights = [40,30,20,10])
    return chars[index[0]]


def generate_names(char, l, model):
    model.eval()
    input = torch.tensor(convert_char_array(char),dtype=torch.float32)
    input = input.to(device)
    output,(h,c) = model(input)
    output_char = choice_prob(output)
    if(output_char == 'EON' or len(char)==l):
        if(len(char)<3):
            return generate_names(char,l,model)
        else:
            return char
    else:
        char = char+output_char
        return generate_names(char,l,model)
    
# Reload the model

device = torch.device("cpu")
model = LSTMText().to(device)
model_path = "0702-677368201-Pai.pt"
checkpoint = torch.load(model_path,map_location=device)
model.load_state_dict(checkpoint)


gen_names = {'a':[],'x':[]}
while len(gen_names['a'])<20:
    name = generate_names('a',15,model)
    if(name not in gen_names['a']):
        gen_names['a'].append(name)
print(str(len(gen_names['a'])),"names generated for a: ")
print(gen_names['a'])

while len(gen_names['x'])<20:
    name = generate_names('x',15,model)
    if(name not in gen_names['x']):
        gen_names['x'].append(name)
print(str(len(gen_names['x'])),"names generated for x: ")
print(gen_names['x'])