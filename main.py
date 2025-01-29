from fooddelivery.components.data_ingestion import DataIngestion
from fooddelivery.components.data_validation import DataValidation
from fooddelivery.components.data_transformation import DataTransformation
from fooddelivery.components.model_trainer import ModelTrainer

from fooddelivery.exception.exception import FoodDeliveryException
from fooddelivery.logging.logger import logging

from fooddelivery.entity.config_entity import DataIngestionConfig, DataValidationConfig, DataTransformationConfig, ModelTrainerConfig
from fooddelivery.entity.config_entity import TrainingPipelineConfig

import sys


if __name__ == "__main__":
    
    try:
        trainingpipelineconfig = TrainingPipelineConfig()

        # data ingestion
        dataingestionconfig = DataIngestionConfig(training_pipeline_config=trainingpipelineconfig)
        data_ingestion = DataIngestion(data_ingestion_config=dataingestionconfig)        
        logging.info("Initiate the data ingestion process.")
        dataingestionartifact = data_ingestion.initiate_data_ingestion()
        logging.info("Data Initiation Completed")
        print(dataingestionartifact)

        # data validation
        data_validation_config = DataValidationConfig(training_pipeline_config=trainingpipelineconfig)
        data_validation = DataValidation(data_ingestion_artifact=dataingestionartifact, data_validation_config=data_validation_config)
        logging.info("Initiate Data Validation")
        data_validation_artifact = data_validation.initiate_data_validation()
        logging.info("Data Validation Completed")
        print(data_validation_artifact)

        # data transformation
        data_transformation_config = DataTransformationConfig(training_pipeline_config=trainingpipelineconfig)
        data_transformation = DataTransformation(data_validation_artifact=data_validation_artifact,data_transformation_config=data_transformation_config)
        logging.info("Initiate Data Transformation")
        data_transformation_artifact = data_transformation.initiate_data_transformation()
        logging.info("Data Transformation Completed")
        print(data_transformation_artifact)

        # model trainer
        logging.info("Model Trainer Started")
        model_trainer_config = ModelTrainerConfig(trainingpipelineconfig)
        model_trainer = ModelTrainer(data_transformation_artifact=data_transformation_artifact, model_trainer_config=model_trainer_config)
        logging.info("Initiate Model Trainer")
        model_trainer_artifact = model_trainer.initiate_model_trainer()
        logging.info("Model Trainer Artifact Created.")
        logging.info("Model Trainer Completed")

    except Exception as e:
        raise FoodDeliveryException(e,sys)