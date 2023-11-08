# best  model for most of the targets 
import torch
import torch.nn as nn
import torch.nn.functional as F
class Net(nn.Module):
    def __init__(self, input_size):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(input_size, 512)  
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 128)
        self.fc4 = nn.Linear(128, 64)
        self.fc5 = nn.Linear(64, 1)  
        self.dropout = nn.Dropout(p=0.5)  # increase the dropout rate to prevent overfitting.
        self.batch_norm1 = nn.BatchNorm1d(512)# use batch normalization to prevent overfitting.
        self.batch_norm2 = nn.BatchNorm1d(256)
        self.batch_norm3 = nn.BatchNorm1d(128)
        self.batch_norm4 = nn.BatchNorm1d(64)  
      
           
    def forward(self, x):
        x = F.relu(self.batch_norm1(self.fc1(x)))# use relu for hidden layers 
        x = self.dropout(x)
        x = F.relu(self.batch_norm2(self.fc2(x)))
        x = self.dropout(x)
        x = F.relu(self.batch_norm3(self.fc3(x)))
        x = self.dropout(x)
        x = F.relu(self.batch_norm4(self.fc4(x)))
        x = self.dropout(x)
        x = torch.sigmoid(self.fc5(x))# use sigmoid for binary classification 
        return x