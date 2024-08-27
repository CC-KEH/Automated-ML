from src.AutoML.entity.config_entity import *
from src.AutoML.constants import *


class Configuration_Manager:
    def __init__(self, configs_path=CONFIG_PATH, params_path=PARAMS_PATH):
        self.config = configs_path
        self.params = params_path

    def get_data_ingestion_config(self) -> Data_Ingestion_Config:
        pass

    # * Regression Configurations

    def get_regression_data_transformation_config(self) -> Regression_Data_Transformation_Config:
        pass

    def get_regression_model_trainer_config(self) -> Regression_Model_Trainer_Config:
        pass

    def get_regression_model_evaluation_config(self) -> Model_Evaluation_Config:
        pass

    # * Classification Configurations

    def get_classification_data_transformation_config(self) -> Classification_Data_Transformation_Config:
        pass

    def get_classification_model_trainer_config(self) -> Classification_Model_Trainer_Config:
        pass

    def get_classification_model_evaluation_config(self) -> Model_Evaluation_Config:
        pass
