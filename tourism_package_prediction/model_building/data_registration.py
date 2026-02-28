
# 1. Importing Necessary Libraries
from huggingface_hub.utils import RepositoryNotFoundError, HfHubHTTPError
from huggingface_hub import HfApi, create_repo
import os

# 2. Creating HF repo for dataset.
repo_id = "svpuranik/tourism-package-prediction-data"
repo_type = "dataset"

# 3. Initialize API client
api = HfApi(token=os.getenv("HF_TOKEN"))

# 4. try: Check if the space exists
try:
    api.repo_info(repo_id=repo_id, repo_type=repo_type)
    print(f"Space '{repo_id}' already exists. Using it.")
except RepositoryNotFoundError:
    print(f"Space '{repo_id}' not found. Creating new space...")
    create_repo(repo_id=repo_id, repo_type=repo_type, private=False)
    print(f"Space '{repo_id}' created.")

# 5. Data upload to HF repository and dataset space
api.upload_folder(
    folder_path="tourism_package_prediction/data",
    repo_id=repo_id,
    repo_type=repo_type,
)
