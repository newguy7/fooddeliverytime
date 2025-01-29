import sys
import os

from fooddelivery.constant.training_pipeline import SAVED_MODEL_DIR, MODEL_FILE_NAME

from fooddelivery.exception.exception import FoodDeliveryException
from fooddelivery.logging.logger import logging

class NetworkModel:
    def __init__(self,preprocessor,model):
        try:
            self.preprocessor = preprocessor
            self.model = model
        except Exception as e:
            raise FoodDeliveryException(e,sys)
        
    def predict(self,x):
        try:
            x_transform = self.preprocessor.transform(x)
            y_hat = self.model.predict(x_transform)
            return y_hat
        except Exception as e:
            raise FoodDeliveryException(e,sys)
            