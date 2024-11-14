import pandas as pd
from AutoML.logger import logger
from AutoML.components.model_trainer import Regression_Model_Trainer, Classification_Model_Trainer, Clustering_Model_Trainer
from AutoML.config.configuration import Configuration_Manager

STAGE_NAME = 'Model Trainer Stage'

class ModelTrainerTrainingPipeline:
    def __init__(self):
        pass
    
    def main(self,manual_config=None):
        if manual_config == None:
            data = pd.read_csv('artifacts/data_ingestion/data.csv')
            if data['target'].nunique() < 10:
                task_type = 'classification'
            else:
                task_type = 'regression'
                
        else:
            task_type = manual_config['task_type']
        
        try:
            if task_type == 'regression':
                config = Configuration_Manager()
                model_trainer_config = config.get_regression_model_trainer_config()
                model_trainer = Regression_Model_Trainer(config=model_trainer_config)
                model_trainer.initiate_model_training(manual_config)

            elif task_type == 'classification':
                config = Configuration_Manager()
                model_trainer_config = config.get_classification_model_trainer_config()
                model_trainer = Classification_Model_Trainer(config=model_trainer_config)
                model_trainer.initiate_model_training(manual_config)
            
            elif task_type == 'clustering':
                config = Configuration_Manager()
                model_trainer_config = config.get_clustering_model_trainer_config()
                model_trainer = Clustering_Model_Trainer(config=model_trainer_config)
                model_trainer.initiate_model_training(manual_config)
            
            else:
                raise ValueError(f"Invalid mode: {task_type}. Please enter either 'regression' or 'classification")
        except Exception as e:
            raise e

if __name__ == "__main__":
    try:
        logger.info(f'>>>>>>> STAGE {STAGE_NAME} Started <<<<<<<')
        obj = ModelTrainerTrainingPipeline()
        obj.main()
        logger.info(f'>>>>>>> STAGE {STAGE_NAME} Completed <<<<<<<\n\nx=============x')
    except Exception as e:
        logger.exception(e)
        raise e