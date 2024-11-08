import os
import json
from pyexpat import model
from AutoML.logger import logger
from AutoML.pipelines.stage_01_data_ingestion import DataIngestionTrainingPipeline
from AutoML.pipelines.stage_02_data_transformation import (
    DataTransformationTrainingPipeline,
)
from AutoML.pipelines.stage_03_model_trainer import ModelTrainerTrainingPipeline
from AutoML.pipelines.stage_04_model_evaluation import ModelEvaluationTrainingPipeline


def data_ingestion(manual_config=None):
    STAGE_NAME = "Data Ingestion Stage"
    try:
        logger.info(f">>>>>>> STAGE {STAGE_NAME} Started <<<<<<<")
        obj = DataIngestionTrainingPipeline()
        obj.main(manual_config)
        logger.info(
            f">>>>>>> STAGE {STAGE_NAME} Completed <<<<<<<\n\nx============================x"
        )
    except Exception as e:
        logger.exception(e)
        raise e


def data_transformation(manual_config=None):
    STAGE_NAME = "Data Transformation Stage"
    try:
        logger.info(f">>>>>>> STAGE {STAGE_NAME} Started <<<<<<<")
        obj = DataTransformationTrainingPipeline()
        obj.main(manual_config)
        logger.info(
            f">>>>>>> STAGE {STAGE_NAME} Completed <<<<<<<\n\nx============================x"
        )
    except Exception as e:
        logger.exception(e)
        raise e

def model_training(manual_config=None):
    STAGE_NAME = "Model Training Stage"
    try:
        logger.info(f">>>>>>> STAGE {STAGE_NAME} Started <<<<<<<")
        obj = ModelTrainerTrainingPipeline()
        obj.main(manual_config)
        logger.info(
            f">>>>>>> STAGE {STAGE_NAME} Completed <<<<<<<\n\nx============================x"
        )
    except Exception as e:
        logger.exception(e)
        raise e


def model_evaluation(manual_config=None):
    STAGE_NAME = "Model Evaluation Stage"
    try:
        logger.info(f">>>>>>> STAGE {STAGE_NAME} Started <<<<<<<")
        obj = ModelEvaluationTrainingPipeline()
        obj.main(manual_config)
        logger.info(
            f">>>>>>> STAGE {STAGE_NAME} Completed <<<<<<<\n\nx============================x"
        )
    except Exception as e:
        logger.exception(e)
        raise e


if __name__ == "__main__":

    is_auto = True
    
    if os.path.exists("manual_config.json"):
        is_auto = False
        data_ingestion_config = {}
        data_transformation_config = {}
        model_training_config = {}
        model_evaluation_config = {}
        
        with open("manual_config.json", "r") as f:
            manual_config = json.load(f)
            data_ingestion_config = manual_config["data_ingestion"]
            data_transformation_config = manual_config["transformation_config"]
            model_training_config["algorithm"] = manual_config["algorithm"]
            model_training_config["task_type"] = manual_config["task_type"]
            model_training_config["evaluation_metric"] = manual_config["evaluation_metric"]
            model_evaluation_config["evaluation_metric"] = manual_config["evaluation_metric"]
            model_evaluation_config["task_type"] = manual_config["task_type"]
            
            if manual_config["hyperparameters"]:
                model_training_config["hyperparameters"] = manual_config["hyperparameters"]

        print("Manual Config Loaded Successfully")
        print("Data Ingestion Config: ", data_ingestion_config)
        print("Data Transformation Config: ", data_transformation_config)
        print("Model Training Config: ", model_training_config)
        print("Model Evaluation Config: ", model_evaluation_config)
        
    if not is_auto:
        data_ingestion(data_ingestion_config)
        data_transformation(data_transformation_config)
        model_training(model_training_config)
        model_evaluation(model_evaluation_config)
        
    else:
        data_ingestion()
        data_transformation()
        model_training()
        model_evaluation()