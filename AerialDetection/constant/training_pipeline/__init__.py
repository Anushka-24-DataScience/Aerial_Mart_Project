ARTIFACTS_DIR: str = "artifacts"

"""
Data Ingestion related constant start with DATA_INGESTION VAR NAME
"""
DATA_INGESTION_DIR_NAME: str = "data_ingestion"

DATA_INGESTION_FEATURE_STORE_DIR: str = "feature_store"

DATA_DOWNLOAD_URL: str ="https://drive.google.com/file/d/12rvJuSkLzVD8H10JjWbWJKW0YJdlog3Y/view?usp=sharing"
#"https://drive.google.com/file/d/1Zp8jkAArUp4Dt2HT-O7QChke7G8RIVDu/view?usp=sharing"
#"https://drive.google.com/file/d/1m1eUl-kTrFHEOqbvLcyviT4sqTj81YAr/view?usp=sharing"



"""
Data Validation realted contant start with DATA_VALIDATION VAR NAME
"""

DATA_VALIDATION_DIR_NAME: str = "data_validation"

DATA_VALIDATION_STATUS_FILE = 'status.txt'

DATA_VALIDATION_ALL_REQUIRED_FILES = ["test","train","valid","data.yaml"]



"""
MODEL TRAINER related constant start with MODEL_TRAINER var name
"""
MODEL_TRAINER_DIR_NAME: str = "model_trainer"

MODEL_TRAINER_PRETRAINED_WEIGHT_NAME: str = "yolov5s.pt"

MODEL_TRAINER_NO_EPOCHS: int = 100

MODEL_TRAINER_BATCH_SIZE: int = 16
