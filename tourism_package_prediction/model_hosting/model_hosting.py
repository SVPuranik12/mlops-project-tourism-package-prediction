
# Library import
from huggingface_hub import HfApi
import os

api = HfApi(token=os.getenv("HF_TOKEN"))

api.upload_folder(
    folder_path="tourism_package_prediction/model_deployment",     # the local folder containing your files
    repo_id="svpuranik/mlops-project-tourism-package-prediction",  # the target repo
    repo_type="space",                                             # Repository type - Space
    path_in_repo="",                                               # optional: subfolder path inside the repo
)
