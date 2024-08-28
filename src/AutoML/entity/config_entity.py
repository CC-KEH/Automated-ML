from dataclasses import dataclass


@dataclass(frozen=True)
class Data_Ingestion_Config:
    root_dir: str
    data_path: str
    unzip_path: str

@dataclass(frozen=True)
class Regression_Data_Transformation_Config:
    root_dir: str
    data_path: str
    train_path: str
    test_path: str

@dataclass(frozen=True)
class Classification_Data_Transformation_Config:
    root_dir: str
    data_path: str
    train_path: str
    test_path: str
    
@dataclass(frozen=True)
class Regression_Model_Trainer_Config:
    root_dir: str
    train_path: str
    model_path: str
    
@dataclass(frozen=True)
class Classification_Model_Trainer_Config:
    root_dir: str
    train_path: str
    model_path: str

@dataclass(frozen=True)
class Model_Evaluation_Config:
    root_dir: str
    test_path: str
    model_path: str