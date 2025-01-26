from fooddelivery.components.data_ingestion import DataIngestion

from fooddelivery.exception.exception import FoodDeliveryException
from fooddelivery.logging.logger import logging

from fooddelivery.entity.config_entity import DataIngestionConfig
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

    except Exception as e:
        raise FoodDeliveryException(e,sys)