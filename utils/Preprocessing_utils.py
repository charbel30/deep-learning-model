# %%
import os
import pandas as pd
import numpy as np
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer, SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import torch
from torch.utils.data import TensorDataset, DataLoader


# %%

def output_selection_prepro(df, target):
    csv_file = f'data/processed/{target}/imputed_data_{target}.csv'
    
    # Check if the file exists
    if os.path.isfile(csv_file):
        print(f'Loading data from {csv_file}...')
        df_imputed = pd.read_csv(csv_file)
        
        # Check if there are missing values
        missing_values = df_imputed.isnull().sum()
        if missing_values.sum() == 0:
            print('No missing values found.')
            num_cols = ['RDAYSFROMINDEX', 'RSEQCATHNUM', 'RSUBJID', 'DPCABG',
                        'DPMI','DPPCI','NUMPRMI','DIASBP_R','PULSE_R','SYSBP_R',
                        'HEIGHT_R','WEIGHT_R','CREATININE_R','HDL_R','LDL_R',
                        'GRAFTST','LVEF_R','DAYS2LKA','DSCABG','DSMI',
                        'TOTCHOL_R','DSPCI','DSSTROKE']
            cat_cols = list(set(df_imputed.columns) - set(num_cols) - set([target]))
            y = df_imputed[target]
            df1 = df_imputed.drop([target], axis=1)
            print(f'Final X columns: {df1.columns.tolist()}')      
            return df1, y, num_cols, cat_cols

    # If the file does not exist or has missing values, run the imputation
    print('Running imputation...')
    targets = ['CHFSEV', 'ACS', 'NUMDZV', 'DEATH', 'LADST', 'LCXST', 'LMST', 'PRXLADST', 'RCAST']
    cat_cols = ['YRCATH_G', 'AGE_G', 'GENDER', 'RACE_G', 'HXANGINA', 'HXCEREB', 'HXCHF', 'HXCOPD', 
                'HXDIAB', 'HXHTN', 'HXHYL', 'HXMI', 'HXSMOKE', 'CBRUITS', 'S3', 
                'CATHAPPR', 'CORDOM','DIAGCATH', 'INTVCATH', 'FUPROTCL']
    num_cols = ['RDAYSFROMINDEX', 'RSEQCATHNUM', 'RSUBJID', 'DPCABG',
                'DPMI','DPPCI','NUMPRMI','DIASBP_R','PULSE_R','SYSBP_R',
                'HEIGHT_R','WEIGHT_R','CREATININE_R','HDL_R','LDL_R',
                'GRAFTST','LVEF_R','DAYS2LKA','DSCABG','DSMI',
                'TOTCHOL_R','DSPCI','DSSTROKE']

    print('Imputing missing values...')
    # Copy the DataFrame and drop rows with missing values in the target column
    df1 = df.copy().dropna(subset = [target])

    # Define the imputer for all columns
    num_imputer = IterativeImputer(max_iter=300,imputation_order='random')
    cat_imputer = SimpleImputer(strategy='most_frequent')
    # Fit and transform the DataFrame with the imputer
    df1[num_cols] = num_imputer.fit_transform(df1[num_cols])
    df1[cat_cols] = cat_imputer.fit_transform(df1[cat_cols])
    


    y = df1[target]

    # Classifying targets
    thresholds = {
        'ACS': 1,
        'CHFSEV': 1,
        'LADST': 70,
        'LCXST': 70,
        'LMST': 70,
        'PRXLADST': 70,
        'RCAST': 70,
        'NUMDZV': 1,
        'DEATH': 1
    }

   # Transform the target variable
    y = np.where(df1[target] >= thresholds[target], 1, 0)
    # Convert y to a Series
    y = pd.Series(y, name=target)
    
    print(f'Dropping targets: {targets}')
    df1 = df1.drop(targets, axis=1)
    
    #df1 = pd.get_dummies(df1, columns=cat_cols, drop_first=True)
    y.index = df1.index
    # Save the DataFrame to a csv file
    df_imputed = pd.concat([df1, y], axis=1)
    os.makedirs(os.path.dirname(csv_file), exist_ok=True)
    #print where the file is saved
    df_imputed.to_csv(csv_file, index=False)
     #print where the file is saved
    print(f'Saved imputed data to {csv_file}')

    cat_cols = list(set(df_imputed.columns) - set(num_cols) - set([target]))
    
    print(f'Final X columns: {df1.columns.tolist()}')
 
    return df1, y, num_cols, cat_cols


    # def  train_test_split_prepro(df, target, test_size=0.3, random_state=454213):

    #     X, y, num_cols, cat_cols = output_selection_prepro(df, target)

    #     # Split the data into train and test sets
    #     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

    #     # Scale the data
    #     scaler = StandardScaler()
    #     X_train = scaler.fit_transform(X_train)
    #     X_test = scaler.transform(X_test)

    #     # Convert features and labels to tensors
    #     X_train_tensor = torch.tensor(X_train.astype(np.float32))
    #     X_test_tensor = torch.tensor(X_test.astype(np.float32))
    #     y_train_tensor = torch.tensor(y_train.values).float().unsqueeze(1)
    #     y_test_tensor = torch.tensor(y_test.values).float().unsqueeze(1)

    #     # Convert the data into PyTorch tensors and load them into a DataLoader
    #     train_data = TensorDataset(X_train_tensor, y_train_tensor)
    #     test_data = TensorDataset(X_test_tensor, y_test_tensor)

    #     # Create directories for train and test data
    #     os.makedirs(f'../data/processed/{target}/train', exist_ok=True)
    #     os.makedirs(f'../data/processed/{target}/test', exist_ok=True)

    #     # Save the train and test data
    #     torch.save(train_data, f'../data/processed/{target}/train/train_data.pth')
    #     torch.save(test_data, f'../data/processed/{target}/test/test_data.pth')

    #     # Save the labels
    #     os.makedirs(f'../data/processed/{target}/labels', exist_ok=True)
    #     y.to_csv(f'../data/processed/{target}/labels/labels.csv', index=False)


