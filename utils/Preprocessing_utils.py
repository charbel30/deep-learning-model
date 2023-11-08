import os
import pandas as pd
import numpy as np
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer, SimpleImputer


def output_selection_prepro(df, target):
    # Define the path of the CSV file where the preprocessed data will be saved
    csv_file = f'data/processed/{target}/imputed_data_{target}.csv'
    
    # Check if the file already exists
    if os.path.isfile(csv_file):
        # If the file exists, load the data from the CSV file
        print(f'Loading data from {csv_file}...')
        df_imputed = pd.read_csv(csv_file)
        
        # Check if there are any missing values in the data
        missing_values = df_imputed.isnull().sum()
        if missing_values.sum() == 0:
            # If there are no missing values, print a message and proceed
            print('No missing values found.')
            # Define the numerical and categorical columns
            num_cols = ['RDAYSFROMINDEX', 'RSEQCATHNUM', 'RSUBJID', 'DPCABG',
                        'DPMI','DPPCI','NUMPRMI','DIASBP_R','PULSE_R','SYSBP_R',
                        'HEIGHT_R','WEIGHT_R','CREATININE_R','HDL_R','LDL_R',
                        'GRAFTST','LVEF_R','DAYS2LKA','DSCABG','DSMI',
                        'TOTCHOL_R','DSPCI','DSSTROKE']
            cat_cols = list(set(df_imputed.columns) - set(num_cols) - set([target]))
            # Define the target variable
            y = df_imputed[target]
            # Drop the target variable from the dataframe
            df1 = df_imputed.drop([target], axis=1)
            print(f'Final X columns: {df1.columns.tolist()}')      
            return df1, y, num_cols, cat_cols

    # If the file does not exist or has missing values, run the imputation
    print('Running imputation...')
    # Define the targets and categorical and numerical columns
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
    
    # Define the target variable
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

   # Transform the target variable based on the thresholds
    y = np.where(df1[target] >= thresholds[target], 1, 0)
    # Convert y to a Series
    y = pd.Series(y, name=target)
    
    # Drop the target columns from the dataframe
    print(f'Dropping targets: {targets}')
    df1 = df1.drop(targets, axis=1)
    #remove comment to use one hot encoding for categorical variables
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

