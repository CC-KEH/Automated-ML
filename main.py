from AutoML.utils import logger
from AutoML.pipelines.stage_01_data_ingestion import DataIngestionTrainingPipeline
from AutoML.pipelines.stage_02_data_transformation import DataTransformationTrainingPipeline
from AutoML.pipelines.stage_03_model_trainer import ModelTrainerTrainingPipeline
from AutoML.pipelines.stage_04_model_evaluation import ModelEvaluationTrainingPipeline

STAGE_NAME = 'Data Ingestion Stage'

try:
    logger.info(f'>>>>>>> STAGE {STAGE_NAME} Started <<<<<<<')
    obj = DataIngestionTrainingPipeline()
    obj.main()
    logger.info(f'>>>>>>> STAGE {STAGE_NAME} Completed <<<<<<<\n\nx============================x')
except Exception as e:
    logger.exception(e)
    raise e

STAGE_NAME = 'Data Transformation'

try:
    logger.info(f'>>>>>>> STAGE {STAGE_NAME} Started <<<<<<<')
    obj = DataTransformationTrainingPipeline()
    obj.main()
    logger.info(f'>>>>>>> STAGE {STAGE_NAME} Completed <<<<<<<\n\nx============================x')
except Exception as e:
    logger.exception(e)
    raise e

STAGE_NAME = 'Model Trainer Stage'

try:
    logger.info(f'>>>>>>> STAGE {STAGE_NAME} Started <<<<<<<')
    obj = ModelTrainerTrainingPipeline()
    obj.main()
    logger.info(f'>>>>>>> STAGE {STAGE_NAME} Completed <<<<<<<\n\nx============================x')
except Exception as e:
    logger.exception(e)
    raise e

STAGE_NAME = 'Model Evaluation Stage'

try:
    logger.info(f'>>>>>>> STAGE {STAGE_NAME} Started <<<<<<<')
    obj = ModelEvaluationTrainingPipeline()
    obj.main()
    logger.info(f'>>>>>>> STAGE {STAGE_NAME} Completed <<<<<<<\n\nx=============x')
except Exception as e:
    logger.exception(e)
    raise e