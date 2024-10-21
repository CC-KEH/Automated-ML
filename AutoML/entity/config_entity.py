from dataclasses import dataclass
from os import pipe
from pathlib import Path
import time
import os
from AutoML.constants import *

@dataclass(frozen=True)
class Data_Ingestion_Config:
    root_dir: Path
    data_path: Path
    source_path: Path
    database_name: str
    collection_name: str
    connection_url: str

@dataclass(frozen=True)
class Data_Transformation_Config:
    root_dir: Path
    data_path: Path
    train_path: Path
    test_path: Path
    
@dataclass(frozen=True)
class Model_Trainer_Config:
    root_dir: Path
    train_path: Path
    params: dict

@dataclass(frozen=True)
class Model_Evaluation_Config:
    root_dir: Path
    test_path: Path
    model_path: Path
    all_params: dict
    metric_file_name: Path
    mlflow_uri: str

@dataclass(frozen=True)
class Training_Pipeline_Config:
    pipeline_name: str = PIPELINE_NAME
    artifacts_dir: str = os.path.join(ARTIFACT_DIR, 'artifacts')
    timestamp: str = TIMESTAMP