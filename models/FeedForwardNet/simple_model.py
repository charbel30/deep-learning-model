# use this if the model is overfitting. simpler model that  makes less mistakes on the test set.

import torch
import torch.nn as nn
import torch.nn.functional as F
class Net(nn.Module):
    def __init__(self, input_size):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(input_size, 64) 
        self.fc2 = nn.Linear(64, 1)
        self.dropout = nn.Dropout(p=0.35)
        self.batch_norm1 = nn.BatchNorm1d(64)
           
    def forward(self, x):
        x = F.relu(self.batch_norm1(self.fc1(x)))
        x = self.dropout(x)
        x = torch.sigmoid(self.fc2(x))
        return x
