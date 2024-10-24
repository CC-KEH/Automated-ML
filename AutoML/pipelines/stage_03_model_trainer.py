import pandas as pd
from AutoML.logger import logger
from AutoML.components.model_trainer import Regression_Model_Trainer, Classification_Model_Trainer 
from AutoML.config.configuration import Configuration_Manager

STAGE_NAME = 'Model Trainer Stage'

class ModelTrainerTrainingPipeline:
    def __init__(self):
        pass
    
    def main(self):
        data = pd.read_csv('artifacts/data_transformation/train.csv')
        if data['target'].dtype == 'float64':
            mode = 'regression'
        else:
            mode = 'classification'
            
        try:
            if mode == 'regression':
                config = Configuration_Manager()
                model_trainer_config = config.get_regression_model_trainer_config()
                model_trainer = Regression_Model_Trainer(config=model_trainer_config)
                model_trainer.initiate_model_training()

            elif mode == 'classification':
                config = Configuration_Manager()
                model_trainer_config = config.get_classification_model_trainer_config()
                model_trainer = Classification_Model_Trainer(config=model_trainer_config)
                model_trainer.initiate_model_training()
            else:
                raise ValueError(f"Invalid mode: {mode}. Please enter either 'regression' or 'classification")
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