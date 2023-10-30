import pandas as pd
import torch
import numpy as np
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, balanced_accuracy_score
from models.FeedForwardNet.model import Net
from utils.Preprocessing_utils import output_selection_prepro

def test_model(model_file, target):
    #load test data
    test_data =torch.load(f'data/processed/{target}/test_data.pth')
    # Load pretrained model and add the input size
    model = Net(input_size=test_data.tensors[0].shape[1])
    model.load_state_dict(torch.load(model_file))
    model.eval()
    

    batch_size = 1024
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=True)

    # Make predictions on test set
    y_true = []
    y_pred = []

    with torch.no_grad():
        for X_test, y_test in test_loader:
            y_pred.extend(model(X_test).round().detach().numpy())
            y_true.extend(y_test.detach().numpy())
        
    # Calculate metrics
    accuracy = accuracy_score(y_true, y_pred)  
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    balance_accuracy = balanced_accuracy_score(y_true, y_pred)

    return accuracy, precision, recall, balance_accuracy