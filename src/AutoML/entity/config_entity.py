from dataclasses import dataclass
from pathlib import Path

@dataclass(frozen=True)
class Data_Ingestion_Config:
    root_dir: str
    data_path: str
    unzip_path: str

@dataclass(frozen=True)
class Regression_Data_Transformation_Config:
    root_dir: Path
    data_path: Path

@dataclass(frozen=True)
class Classification_Data_Transformation_Config:
    root_dir: str
    data_path: str
    train_path: str
    test_path: str
    
@dataclass(frozen=True)
class Regression_Model_Trainer_Config:
    root_dir: Path
    train_path: Path
    test_path: Path
    params: dict
    target_column: str
    
@dataclass(frozen=True)
class Classification_Model_Trainer_Config:
    root_dir: Path
    train_path: Path
    test_path: Path
    params: dict
    target_column: str

@dataclass(frozen=True)
class Model_Evaluation_Config:
    root_dir: Path
    test_path: Path
    model_path: Path
    all_params: dict
    metric_file_name: Path
    target_column: str
    mlflow_uri: str