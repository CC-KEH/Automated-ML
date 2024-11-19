from AutoML.logger import logger
from AutoML.components.model_evaluation import Regression_Model_Evaluation, Classification_Model_Evaluation, Clustering_Model_Evaluation 
from AutoML.config.configuration import Configuration_Manager
import pandas as pd


STAGE_NAME = 'Model Evaluation Stage'

class ModelEvaluationTrainingPipeline:
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
                model_evaluation_config = config.get_regression_model_evaluation_config()
                model_evaluation = Regression_Model_Evaluation(config=model_evaluation_config)
                model_evaluation.initiate_model_evaluation(manual_config)

            elif task_type == 'classification':
                config = Configuration_Manager()
                model_evaluation_config = config.get_classification_model_evaluation_config()
                model_evaluation = Classification_Model_Evaluation(config=model_evaluation_config)
                model_evaluation.initiate_model_evaluation(manual_config)
            
            elif task_type == 'clustering':
                config = Configuration_Manager()
                model_evaluation_config = config.get_clustering_model_evaluation_config()
                model_evaluation = Clustering_Model_Evaluation(config=model_evaluation_config)
                model_evaluation.initiate_model_evaluation(manual_config)
            
            else:
                raise ValueError(f"Invalid mode: {task_type}. Please enter either 'regression' or 'classification")
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