from AutoML.logger import logger
from AutoML.components.data_transformation import Data_Transformation
from AutoML.config.configuration import Configuration_Manager

STAGE_NAME = 'Data Transformation'

class DataTransformationTrainingPipeline:
    def __init__(self):
        pass
    
    def main(self,manual_config):    
        try:
            config = Configuration_Manager()
            data_transformation_config = config.get_data_transformation_config()
            data_transformation = Data_Transformation(config=data_transformation_config)
            data_transformation.initiate_data_transformation(manual_config)
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
    