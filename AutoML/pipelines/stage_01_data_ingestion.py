from AutoML.config.configuration import Configuration_Manager
from AutoML.components.data_ingestion import Data_Ingestion
from AutoML.utils.main_utils import logger


STAGE_NAME = 'Data Ingestion Stage'


class DataIngestionTrainingPipeline:
    def __init__(self):
        pass

    def main(self):
        config = Configuration_Manager()
        data_ingestion_config = config.get_data_ingestion_config()
        data_ingestion = Data_Ingestion(config=data_ingestion_config)
        data_ingestion.initiate_data_ingestion()

if __name__ == "__main__":
    try:
        zipped_data = None
        logger.info(f'>>>>>>> STAGE {STAGE_NAME} Started <<<<<<<')
        obj = DataIngestionTrainingPipeline()
        obj.main(zipped_data)
        logger.info(f'>>>>>>> STAGE {STAGE_NAME} Completed <<<<<<<\n\nx=============x')
    except Exception as e:
        logger.exception(e)
        raise e