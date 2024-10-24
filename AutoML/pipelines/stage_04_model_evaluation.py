from AutoML.logger import logger
from AutoML.components.model_evaluation import Regression_Model_Evaluation, Classification_Model_Evaluation 
from AutoML.config.configuration import Configuration_Manager
import pandas as pd


STAGE_NAME = 'Model Evaluation Stage'

class ModelEvaluationTrainingPipeline:
    def __init__(self):
        pass
    
    def main(self):
        data = pd.read_csv('artifacts/data_transformation/test.csv')
        if data['target'].dtype == 'float64':
            mode = 'regression'
        else:
            mode = 'classification'
            
        try:
            if mode == 'regression':
                config = Configuration_Manager()
                model_evaluation_config = config.get_regression_model_evaluation_config()
                model_evaluation = Regression_Model_Evaluation(config=model_evaluation_config)
                model_evaluation.initiate_model_evaluation()

            elif mode == 'classification':
                config = Configuration_Manager()
                model_evaluation_config = config.get_classification_model_evaluation_config()
                model_evaluation = Classification_Model_Evaluation(config=model_evaluation_config)
                model_evaluation.initiate_model_evaluation()
            else:
                raise ValueError(f"Invalid mode: {mode}. Please enter either 'regression' or 'classification")
        except Exception as e:
            raise e

if __name__ == "__main__":
    try:
        logger.info(f'>>>>>>> STAGE {STAGE_NAME} Started <<<<<<<')
        obj = ModelEvaluationTrainingPipeline()
        obj.main()
        logger.info(f'>>>>>>> STAGE {STAGE_NAME} Completed <<<<<<<\n\nx=============x')
    except Exception as e:
        logger.exception(e)
        raise e