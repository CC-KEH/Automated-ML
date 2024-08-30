from src.AutoML.constants import *
from src.AutoML.entity.config_entity import *
from src.AutoML.utils.common import read_yaml,create_directories

class Configuration_Manager:
    def __init__(self, configs_path=CONFIG_PATH, params_path=PARAMS_PATH):
        self.config = read_yaml(configs_path)
        self.params = read_yaml(params_path)
        create_directories([self.config.artifacts_root])
    
    def get_data_ingestion_config(self) -> Data_Ingestion_Config:
        config = self.config.data_ingestion
        create_directories([config.root_dir])
        data_ingestion_config = Data_Ingestion_Config(
            root_dir=config.root_dir,
            data_path=config.local_data_file,
            unzip_path=config.unzip_dir
        )
        return data_ingestion_config

    # * Regression Configurations

    def get_regression_data_transformation_config(self) -> Regression_Data_Transformation_Config:
        config = self.config.data_transformation
        create_directories([config.root_dir])
        data_transformation_config = Regression_Data_Transformation_Config(root_dir=config.root_dir,
                                                                           train_path=config.train_path,
                                                                           test_path=config.test_path,
                                                                           data_path=config.data_path)
        return data_transformation_config

    def get_regression_model_trainer_config(self) -> Regression_Model_Trainer_Config:
        config = self.config.model_trainer
        schema = self.schema.TARGET_COLUMN
        params = self.params
        create_directories([config.root_dir])
        model_trainer_config = Regression_Model_Trainer_Config(root_dir=config.root_dir,
                                                               train_path=config.train_path,
                                                               test_path=config.test_path,
                                                               target_column=schema,
                                                               params=params)
        return model_trainer_config

    def get_regression_model_evaluation_config(self) -> Model_Evaluation_Config:
        config = self.config.model_evaluation
        schema = self.schema.TARGET_COLUMN
        params = self.params #! Fix the issue of params
        
        create_directories([config.root_dir])
        
        model_evaluation_config = Model_Evaluation_Config(root_dir=config.root_dir,
                                                          test_path=config.test_path,
                                                          model_path=config.model_path,
                                                          all_params=params,
                                                          metric_file_name=config.metric_file_name,
                                                          mlflow_uri="https://dagshub.com/CC-KEH/AutomatedML.mlflow",
                                                          target_column=schema.name)
        return model_evaluation_config
    

    # * Classification Configurations

    def get_classification_data_transformation_config(self) -> Classification_Data_Transformation_Config:
        config = self.config.data_transformation
        create_directories([config.root_dir])
        data_transformation_config = Classification_Data_Transformation_Config(root_dir=config.root_dir,
                                                                               data_path=config.data_path,
                                                                               train_path=config.train_path,
                                                                               test_path=config.test_path)
        return data_transformation_config

    def get_classification_model_trainer_config(self) -> Classification_Model_Trainer_Config:
        config = self.config.model_trainer
        schema = self.schema.TARGET_COLUMN
        params = self.params
        create_directories([config.root_dir])
        model_trainer_config = Classification_Model_Trainer_Config(root_dir=config.root_dir,
                                                                   train_path=config.train_path,
                                                                   test_path=config.test_path,
                                                                   target_column=schema,
                                                                   params=params)
        return model_trainer_config

    def get_classification_model_evaluation_config(self) -> Model_Evaluation_Config:
        config = self.config.model_evaluation
        schema = self.schema.TARGET_COLUMN
        params = self.params #! Fix the issue of params
        
        create_directories([config.root_dir])
        
        model_evaluation_config = Model_Evaluation_Config(root_dir=config.root_dir,
                                                          test_path=config.test_path,
                                                          model_path=config.model_path,
                                                          all_params=params,
                                                          metric_file_name=config.metric_file_name,
                                                          mlflow_uri="https://dagshub.com/CC-KEH/AutomatedML.mlflow",
                                                          target_column=schema.name)
        return model_evaluation_config