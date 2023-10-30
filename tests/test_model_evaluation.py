import unittest
import sys
import pandas as pd
from utils.Preprocessing_utils import output_selection_prepro
from model import Net
import torch
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, balanced_accuracy_score

class TestModelEvaluation(unittest.TestCase):
    def test_model_evaluation(self):
        # Load your data
        df = pd.read_csv('../data/raw/dukecathr.csv')
        target = 'RCAST'
        X, y, num_cols, cat_cols = output_selection_prepro(df, target)

        # Split the data into train and test sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=121345)

        # Scale the data
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        # Convert features and labels to tensors
        X_test_tensor = torch.tensor(X_test.astype(np.float32))
        y_test_tensor = torch.tensor(y_test.values).float().unsqueeze(1)

        # Convert the data into PyTorch tensors and load them into a DataLoader
        test_data = TensorDataset(X_test_tensor, y_test_tensor)
        batch_size = 2048
        test_loader = DataLoader(test_data, shuffle=True, batch_size=batch_size)

        # Load the model
        model = Net(X.shape[1])
        model.load_state_dict(torch.load('saved_model.pth'))
        model.eval()

        # Initialize lists to store targets and predictions
        all_targets = []
        all_predictions = []

        # Iterate over the test data
        for inputs, targets in test_loader:
            # Make predictions
            outputs = model(inputs)
            predicted = torch.round(outputs)  # Round the output probabilities to get binary predictions

            # Store targets and predictions
            all_targets.extend(targets.tolist())
            all_predictions.extend(predicted.tolist())

        # Calculate metrics
        accuracy = accuracy_score(all_targets, all_predictions)
        precision = precision_score(all_targets, all_predictions)
        recall = recall_score(all_targets, all_predictions)
        balance_accuracy = balanced_accuracy_score(all_targets, all_predictions)

        # Check if the metrics are within an acceptable range
        self.assertGreater(accuracy, 0.5)
        self.assertGreater(precision, 0.5)
        self.assertGreater(recall, 0.5)
        self.assertGreater(balance_accuracy, 0.5)

if __name__ == '__main__':
    unittest.main(argv=['first-arg-is-ignored'], exit=False)
