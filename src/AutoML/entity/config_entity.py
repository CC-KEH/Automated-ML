from dataclasses import dataclass
from pathlib import Path

@dataclass(frozen=True)
class Data_Ingestion_Config:
    root_dir: Path
    data_path: Path
    source_path: Path

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
    target_column: str
    mlflow_uri: str