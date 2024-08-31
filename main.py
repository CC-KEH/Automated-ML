from src.AutoML.utils import logger
from src.AutoML.pipelines.stage_01_data_ingestion import DataIngestionTrainingPipeline
from src.AutoML.pipelines.stage_02_data_transformation import DataTransformationTrainingPipeline
from src.AutoML.pipelines.stage_03_model_trainer import ModeLTrainerTrainingPipeline
from src.AutoML.pipelines.stage_04_model_evaluation import ModelEvaluationTrainingPipeline

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
    obj = ModeLTrainerTrainingPipeline()
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


def start():
    print("1. Insert Dataset")
    print("2. Choose What to predict")
    
    data_ingestion = Data_Ingestion()
    data_validation = Data_Validation()
    data_tranformation = Data_Transformation()
    model_trainer = Model_Trainer()
    model_evaluation = Model_Evaluation()
    
    data_ingestion.initiate_data_ingestion()
    
    choice = input("What do you want to predict?\n")
    
    data_validation.initiate_data_validation()
    data_tranformation.initiate_data_transformation(choice)
    model_trainer.initiate_model_training()
    model_evaluation.initiate_model_evaluation()