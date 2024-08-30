from src.AutoML.utils.common import logger
from src.AutoML.components.data_transformation import Regression_Data_Transformation, Classification_Data_Transformation 
from src.AutoML.config.configuration import Configuration_Manager

STAGE_NAME = 'Data Transformation'

class DataTransformationTrainingPipeline:
    def __init__(self):
        pass
    
    def main(self, mode='regression'):    
        try:
            if mode == 'regression':
                config = Configuration_Manager()
                data_transformation_config = config.get_regression_data_transformation_config()
                data_transformation = Regression_Data_Transformation(config=data_transformation_config)
                data_transformation.initiate_data_transformation()
            elif mode == 'classification':
                config = Configuration_Manager()
                data_transformation_config = config.get_classification_data_transformation_config()
                data_transformation = Classification_Data_Transformation(config=data_transformation_config)
                data_transformation.initiate_data_transformation()
            else:
                raise ValueError(f"Invalid mode: {mode}. Please enter either 'regression' or 'classification")
        except Exception as e:
            raise e

if __name__ == "__main__":
    try:
        logger.info(f'>>>>>>> STAGE {STAGE_NAME} Started <<<<<<<')
        obj = DataTransformationTrainingPipeline()
        obj.main()
        logger.info(f'>>>>>>> STAGE {STAGE_NAME} Completed <<<<<<<\n\nx=============x')
    except Exception as e:
        logger.exception(e)
        raise e
    