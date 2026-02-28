
# 1. Import necessary Libraries
import pandas as pd
import sklearn
import sklearn.metrics as metrics
# for creating a folder
import os
# for data preprocessing and pipeline creation
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import make_column_transformer
from sklearn.pipeline import Pipeline
# for model training, tuning, and evaluation
import xgboost as xgb
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, classification_report, recall_score
# for model serialization
import joblib
# for hugging face space authentication to upload files
from huggingface_hub import login, HfApi, create_repo
from huggingface_hub.utils import RepositoryNotFoundError, HfHubHTTPError
import mlflow

# 2. Creating MLFlow experiment for tracking
mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("mlops-tourism-package-prediction-cicd-experiments")

# 3. Authentication with HuggingFace API
api = HfApi()

# 4. Reading training and testing dataset registered on huggingface dataset space
X_train_path = "hf://datasets/svpuranik/tourism-package-prediction-data/tourism_package_prediction_X_train_data.csv"
X_test_path = "hf://datasets/svpuranik/tourism-package-prediction-data/tourism_package_prediction_X_test_data.csv"
y_train_path = "hf://datasets/svpuranik/tourism-package-prediction-data/tourism_package_prediction_Y_train_data.csv"
y_test_path = "hf://datasets/svpuranik/tourism-package-prediction-data/tourism_package_prediction_Y_test_data.csv"

X_train = pd.read_csv(X_train_path)
X_test  = pd.read_csv(X_test_path)
y_train = pd.read_csv(y_train_path)
y_test  = pd.read_csv(y_test_path)

# 5. Feature extraction as target, numerical and categorical features.
# Define the target variable for the classification task
target = 'ProdTaken'

# List of numerical features in the dataset
numerical_features = ['Age'                      ,
                      'DurationOfPitch'          ,
                      'NumberOfPersonVisiting'   ,
                      'NumberOfFollowups'        ,
                      'NumberOfTrips'            ,
                      'NumberOfChildrenVisiting' ,
                      'MonthlyIncome'
                      ]

# List of categorical features in the dataset
categorical_features = ['TypeofContact',
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


# Set the class weight to handle class imbalance
class_weight = y_train.value_counts()[0] / y_train.value_counts()[1]

# 6. Initialise data preprocessing pipeline
# Define the preprocessing steps
data_preprocessor = make_column_transformer(
    (StandardScaler(), numerical_features),
    (OneHotEncoder(handle_unknown='ignore'), categorical_features)
)

# 7. Model training and Hyperparameter tuning
# Define base XGBoost model
xgb_base_classifier = xgb.XGBClassifier(scale_pos_weight=class_weight, random_state=42)

# Model pipeline
xgb_tuning_pipeline = Pipeline(steps=[
    ("Preprocessing_Step", data_preprocessor),           # Preprocesses numerical and categorical features
    ("Model_Initialisation", xgb_base_classifier) # XGBoost regressor for model training
])


# Define hyperparameter grid
param_grid = {
    'Model_Initialisation__n_estimators': [50, 75, 100, 150, 200],    # number of tree to build
    'Model_Initialisation__max_depth': [2, 3, 4],                 # maximum depth of each tree
    'Model_Initialisation__colsample_bytree': [0.4, 0.5, 0.6],    # percentage of attributes to be considered (randomly) for each tree
    'Model_Initialisation__colsample_bylevel': [0.4, 0.5, 0.6],   # percentage of attributes to be considered (randomly) for each level of a tree
    'Model_Initialisation__learning_rate': [0.01, 0.05, 0.1],     # learning rate
    'Model_Initialisation__reg_lambda': [0.4, 0.5, 0.6],          # L2 regularization factor
}

# Start MLflow run
with mlflow.start_run():
    # Hyperparameter tuning
    grid_search_xgb_tuned = GridSearchCV(xgb_tuning_pipeline, param_grid, cv=5, n_jobs=-1)
    grid_search_xgb_tuned.fit(X_train, y_train)

    # Log all parameter combinations and their mean test scores
    results = grid_search_xgb_tuned.cv_results_
    for i in range(len(results['params'])):
        param_set = results['params'][i]
        mean_score = results['mean_test_score'][i]
        std_score = results['std_test_score'][i]

        # Log each combination as a separate MLflow run
        with mlflow.start_run(nested=True):
            mlflow.log_params(param_set)
            mlflow.log_metric("mean_test_score", mean_score)
            mlflow.log_metric("std_test_score", std_score)

    # Log best parameters separately in main run
    mlflow.log_params(grid_search_xgb_tuned.best_params_)

    # Store and evaluate the best model
    xgb_tuned_model = grid_search_xgb_tuned.best_estimator_

    classification_threshold = 0.45

    y_pred_train_proba = xgb_tuned_model.predict_proba(X_train)[:, 1]
    y_pred_train = (y_pred_train_proba >= classification_threshold).astype(int)

    y_pred_test_proba = xgb_tuned_model.predict_proba(X_test)[:, 1]
    y_pred_test = (y_pred_test_proba >= classification_threshold).astype(int)

    train_report = classification_report(y_train, y_pred_train, output_dict=True)
    test_report = classification_report(y_test, y_pred_test, output_dict=True)

    # Log the metrics for the best model
    mlflow.log_metrics({
        "train_accuracy": train_report['accuracy'],
        "train_precision": train_report['1']['precision'],
        "train_recall": train_report['1']['recall'],
        "train_f1-score": train_report['1']['f1-score'],
        "test_accuracy": test_report['accuracy'],
        "test_precision": test_report['1']['precision'],
        "test_recall": test_report['1']['recall'],
        "test_f1-score": test_report['1']['f1-score']
    })

    # 8. Taking best model and logging into MLFlow
    # Save the model locally
    tuned_model_path = "optimal_tourism_package_prediction_model_v1.joblib"
    joblib.dump(xgb_tuned_model, tuned_model_path)

    # Log the model artifact
    mlflow.log_artifact(tuned_model_path, artifact_path="model")
    print(f"Model saved as artifact at: {tuned_model_path}")

    # 9. Uploading the best model to hugging face model space.
    repo_id = "svpuranik/tourism-package-prediction-models"
    repo_type = "model"

    # try: Check if the space exists
    try:
        api.repo_info(repo_id=repo_id, repo_type=repo_type)
        print(f"Space '{repo_id}' already exists. Using it.")
    except RepositoryNotFoundError:
        print(f"Space '{repo_id}' not found. Creating new space...")
        create_repo(repo_id=repo_id, repo_type=repo_type, private=False)
        print(f"Space '{repo_id}' created.")

    # 10. Data upload to Hugging face model space
    api.upload_file(
        path_or_fileobj="optimal_tourism_package_prediction_model_v1.joblib",
        path_in_repo="optimal_tourism_package_prediction_model_v1.joblib",
        repo_id=repo_id,
        repo_type=repo_type,
    )
