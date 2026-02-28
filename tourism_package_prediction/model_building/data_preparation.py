
# 1. Import necessary libraries for data manipulation
import pandas as pd
import sklearn
import os # for creating a folder
from sklearn.model_selection import train_test_split # for data preprocessing and pipeline creation
from huggingface_hub import login, HfApi # for hugging face space authentication to upload files

#2. Authentication Using API token
api = HfApi(token=os.getenv("HF_TOKEN"))

#3. Dataset loading from HF
DATASET_PATH = "hf://datasets/svpuranik/tourism-package-prediction-data/tourism.csv"

df_main_data = pd.read_csv(DATASET_PATH)

print("[Data Preparation] Dataset loaded successfully.")

# Creating the copy of main data set for further operations
df_tourism_package_modelling = df_main_data.copy()

# 4. Data cleaning

# 4A. Drop the unique identifier
df_tourism_package_modelling.drop(columns=['Unnamed: 0', 'CustomerID'], inplace=True)

# 4B. Drop duplicates
if (df_tourism_package_modelling.duplicated().sum() > 0):
    df_tourism_package_modelling.drop_duplicates(inplace=True)
    print("[Data Preparation] Dataset has {} duplicated values.".format(df_tourism_package_modelling.duplicated().sum()))
else:
    print("[Data Preparation] Dataset has no duplicate values.")

# 4B. Convert Object columns to Category
object_columns = df_tourism_package_modelling.select_dtypes(['object'])

# Appending the object type columns in main data frame
for col in object_columns.columns:
    df_tourism_package_modelling[col] = df_tourism_package_modelling[col].astype('category')

# 4C. Convert numerical columns which are actually are objects to category
numeric_columns_as_category = ["ProdTaken", "CityTier", "PreferredPropertyStar", "PitchSatisfactionScore", "Passport", "OwnCar"]

# Appending the object type columns in main data frame
for col in numeric_columns_as_category:
    df_tourism_package_modelling[col] = df_tourism_package_modelling[col].astype('category')

# 4D. Data cleaning - Replace improper entries in the data

# Perform the replace operation
df_tourism_package_modelling['Gender'] = df_tourism_package_modelling['Gender'].replace('Fe Male', 'Female')
df_tourism_package_modelling['MaritalStatus'] = df_tourism_package_modelling['MaritalStatus'].replace('Unmarried', 'Single')

print("[Data Preparation] Dataset cleaning is successfully performed.")

# 5. Data Formulation

# Define the target variable for the classification task
target = 'ProdTaken'

# List of numerical features in the dataset
numerical_features = [
                      'Age'                      ,
                      'DurationOfPitch'          ,
                      'NumberOfPersonVisiting'   ,
                      'NumberOfFollowups'        ,
                      'NumberOfTrips'            ,
                      'NumberOfChildrenVisiting' ,
                      'MonthlyIncome'
                      ]

# List of categorical features in the dataset
categorical_features = [
                        'TypeofContact',
                        'PitchSatisfactionScore',
                        'CityTier',
                        'Occupation',
                        'Gender',
                        'ProductPitched',
                        'PreferredPropertyStar',
                        'MaritalStatus',
                        'Designation',
                        'Passport',
                        'OwnCar'
					  ]

# 6. Data Formulation as target and predictors

# Define predictor matrix (X) using selected numeric and categorical features
X_data = df_tourism_package_modelling[numerical_features + categorical_features]

# Define target variable
y_data = df_tourism_package_modelling[target]

print("[Data Preparation] Dataset formulation is successfully performed.")

# 7. Data splitting - train_test
# Split the dataset into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X_data,            # Predictors (X)
                                                    y_data,            # Target variable (y)
                                                    test_size=0.2,     # 20% of the data is reserved for testing
                                                    stratify = y_data, # Keep the class distribution same in training and testing data
                                                    random_state=42    # Ensures reproducibility by setting a fixed random seed
                                                    )


# 7. Saving dataset Locally
X_train.to_csv("tourism_package_prediction_X_train_data.csv",index=False)
X_test.to_csv("tourism_package_prediction_X_test_data.csv",index=False)
y_train.to_csv("tourism_package_prediction_Y_train_data.csv",index=False)
y_test.to_csv("tourism_package_prediction_Y_test_data.csv",index=False)

files_to_upload = ["tourism_package_prediction_X_train_data.csv",
                   "tourism_package_prediction_X_test_data.csv",
                   "tourism_package_prediction_Y_train_data.csv",
                   "tourism_package_prediction_Y_test_data.csv"]

print("[Data Preparation] Dataset splitting is successfully performed.")

# 8. Uploading dataset to HF dataset space
for file_path in files_to_upload:
    api.upload_file(
        path_or_fileobj=file_path,
        path_in_repo=file_path.split("/")[-1],  # just the filename
        repo_id="svpuranik/tourism-package-prediction-data",
        repo_type="dataset",
    )

print("[Data Preparation] Dataset upload is successfully performed.")
