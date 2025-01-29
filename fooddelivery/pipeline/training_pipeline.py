import os
import sys

from fooddelivery.exception.exception import FoodDeliveryException
from fooddelivery.logging.logger import logging

from fooddelivery.components.data_ingestion import DataIngestion
from fooddelivery.components.data_validation import DataValidation
from fooddelivery.components.data_transformation import DataTransformation
from fooddelivery.components.model_trainer import ModelTrainer

from fooddelivery.entity.config_entity import (
    TrainingPipelineConfig,
    DataIngestionConfig,
    DataValidationConfig,
    DataTransformationConfig,
    ModelTrainerConfig
)

from fooddelivery.entity.artifact_entity import (
    DataIngestionArtifact,
    DataValidationArtifact,
    DataTransformationArtifact,
    ModelTrainerArtifact
)

from fooddelivery.constant.training_pipeline import TRAINING_BUCKET_NAME


class TrainingPipeline:
    def __init__(self):
        self.training_pipeline_config = TrainingPipelineConfig()


    def start_data_ingestion(self):
        try:
            self.data_ingestion_config = DataIngestionConfig(training_pipeline_config=self.training_pipeline_config)
            logging.info("Start Data Ingestion")
            data_ingestion = DataIngestion(data_ingestion_config=self.data_ingestion_config)
            data_ingestion_artifact = data_ingestion.initiate_data_ingestion()
            logging.info(f"Data Ingestion Completed and artifact: {data_ingestion_artifact}")
            return data_ingestion_artifact
        except Exception as e:
            raise FoodDeliveryException(e,sys)
        
    def start_data_validation(self, data_ingestion_artifact:DataIngestionArtifact):
        try:
            data_validation_config = DataValidationConfig(training_pipeline_config=self.training_pipeline_config)
            data_validation = DataValidation(data_ingestion_artifact=data_ingestion_artifact,data_validation_config=data_validation_config)
            logging.info("Initiate Data Validation")
            data_validation_artifact = data_validation.initiate_data_validation()
            logging.info("Data Validation Completed")
            return data_validation_artifact
        except Exception as e:
            raise FoodDeliveryException(e,sys)
        
    def start_data_transformation(self, data_validation_artifact: DataValidationArtifact):
        try:
            data_transformation_config = DataTransformationConfig(training_pipeline_config=self.training_pipeline_config)
            data_transformation = DataTransformation(data_validation_artifact=data_validation_artifact, data_transformation_config=data_transformation_config)
            logging.info("Initiate Data Transformation")
            data_transformation_artifact = data_transformation.initiate_data_transformation()
            logging.info("Data Transformation Completed")
            return data_transformation_artifact
        except Exception as e:
            raise FoodDeliveryException(e,sys)
        
    def start_model_trainer(self, data_transformation_artifact:DataTransformationArtifact) -> ModelTrainerArtifact:
        try:
            logging.info("Model Trainer Started")
            self.model_trainer_config: ModelTrainerConfig = ModelTrainerConfig(
                training_pipeline_config=self.training_pipeline_config
            )

            model_trainer = ModelTrainer(
                model_trainer_config=self.model_trainer_config,
                data_transformation_artifact= data_transformation_artifact
            )

            model_trainer_artifact = model_trainer.initiate_model_trainer()
            logging.info("Model Trainer Artifact Created")
            logging.info("Model Trainer Completed")

            return model_trainer_artifact
        except Exception as e:
            raise FoodDeliveryException(e,sys)
        
    
    def run_pipeline(self):
        try:
            data_ingestion_artifact = self.start_data_ingestion()
            data_validation_artifact = self.start_data_validation(data_ingestion_artifact=data_ingestion_artifact)
            data_transformation_artifact = self.start_data_transformation(data_validation_artifact=data_validation_artifact)
            model_trainer_artifact = self.start_model_trainer(data_transformation_artifact=data_transformation_artifact)

            return model_trainer_artifact
        except Exception as e:
            raise FoodDeliveryException(e,sys)