import os
import sys
import numpy as np

from fooddelivery.exception.exception import FoodDeliveryException
from fooddelivery.logging.logger import logging

from fooddelivery.entity.artifact_entity import (
    DataTransformationArtifact,
    ModelTrainerArtifact
)
from fooddelivery.entity.config_entity import ModelTrainerConfig

from fooddelivery.utils.main_utils.utils import save_object, load_object
from fooddelivery.utils.main_utils.utils import load_numpy_array_data, evaluate_models
from fooddelivery.utils.ml_metrics.metric.regression_metric import get_regression_value
from fooddelivery.utils.ml_metrics.model.estimator import NetworkModel

from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import (
    AdaBoostRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor,
        
)
import xgboost
from xgboost import XGBRegressor

from sklearn.utils.estimator_checks import check_estimator

import mlflow

# Track MLFLOW EXPERIMENT TRACKING WITH REMOTE RESPONSE
import dagshub

DAGSHUB_USER = os.getenv("DAGSHUB_USER")
DAGSHUB_TOKEN = os.getenv("DAGSHUB_TOKEN")
DAGSHUB_HOST = "https://dagshub.com"

# Debugging: Print values to ensure they're set (Remove in production)
if not DAGSHUB_USER or not DAGSHUB_TOKEN:
    raise ValueError("DAGSHUB_USER or DAGSHUB_TOKEN is missing!")

# Authenticate with Dagshub before initializing MLflow tracking
dagshub.auth.add_app_token(DAGSHUB_USER, DAGSHUB_TOKEN, host=DAGSHUB_HOST)

dagshub.init(repo_owner='newguy7', repo_name='fooddeliverytime', mlflow=True)


class ModelTrainer:
    def __init__(self,model_trainer_config: ModelTrainerConfig, data_transformation_artifact:DataTransformationArtifact):
        try:
            self.model_trainer_config= model_trainer_config
            self.data_transformation_artifact=data_transformation_artifact
        except Exception as e:
            raise FoodDeliveryException(e,sys)
        
    ## Model experiment tracking with MLFlow    

    def track_mlflow(self,best_model,regressionmetric):
        with mlflow.start_run():
            mae_value = regressionmetric.mae_value
            rmse_value = regressionmetric.rmse_value
            r2_value = regressionmetric.r2_value
            mape_value = regressionmetric.mape_value
            
            mlflow.log_metric("mae_value", mae_value)
            mlflow.log_metric("rmse_value", rmse_value)
            mlflow.log_metric("r2_value", r2_value)
            mlflow.log_metric("mape_value", mape_value)
            mlflow.sklearn.log_model(best_model, "model")

    def train_model(self,X_train,y_train,X_test,y_test):
        try:
            # Reduced verbosity (verbose=0)
            # Added random_state for consistency
            models = {
                "Random Forest": RandomForestRegressor(random_state=42, verbose=0),
                "Decision Tree": DecisionTreeRegressor(random_state=42),
                "Gradient Boosting": GradientBoostingRegressor(random_state=42, verbose=0),                
                "Linear Regression": LinearRegression(),
                "K-Neighbors Regressor": KNeighborsRegressor(),
                "XGBRegressor": XGBRegressor(objective='reg:squarederror',random_state=42),                
                # "AdaBoost Regressor": AdaBoostRegressor(random_state=42)
            }

            params = {
                "Decision Tree": {
                    'criterion':['squared_error','absolute_error']                    
                },
                "Random Forest": {                    
                    'max_features': ['sqrt', 'log2'],
                    'n_estimators': [8,16,32,64]
                },
                "Gradient Boosting": {      
                    'loss': ['squared_error', 'huber'],              
                    'learning_rate':[.1,.01,.05,.001],
                    'subsample': [.6,.7,.75,.8],                    
                    'n_estimators': [32,64, 128]                                    
                },
                "Linear Regression" : {},
                "K-Neighbors Regressor": {
                    'n_neighbors': [5,7,9,11]
                },
                "XGBRegressor":{
                    'learning_rate': [.1,.01,.05,.001],
                    'n_estimators': [8,16,32,64]
                },                
                # "AdaBoost Regressor": {
                #     'learning_rate': [.1,.01,.05,.001],
                #     'n_estimators': [32,64,128],
                #     'loss': ['linear', 'square', 'exponential']
                # }
            }            
            
            model_report:dict = evaluate_models(X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test,
                                                models=models, param=params)
            
            # To get the best model score from dict
            best_model_score = max(sorted(model_report.values()))

            # To get the best model name from dict
            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]

            best_model = models[best_model_name]

            logging.info(f"The best model selected: {best_model_name}")
            logging.info(f"Best model detail: {best_model}")

            y_train_pred = best_model.predict(X_train)
            regression_train_metric = get_regression_value(y_true=y_train, y_pred=y_train_pred)

            # Track the experiment with MLFlow
            self.track_mlflow(best_model,regression_train_metric)

            y_test_pred = best_model.predict(X_test)
            regression_test_metric = get_regression_value(y_true=y_test, y_pred=y_test_pred)

            # Track with MLFlow
            self.track_mlflow(best_model, regression_test_metric)


            preprocessor = load_object(file_path=self.data_transformation_artifact.transformed_object_file_path)

            model_dir_path = os.path.dirname(self.model_trainer_config.trained_model_file_path)
            os.makedirs(model_dir_path,exist_ok=True)

            network_model = NetworkModel(preprocessor=preprocessor, model=best_model)
            save_object(file_path=self.model_trainer_config.trained_model_file_path, obj=network_model)

            # save it in the final_models folder, for model pusher purpose
            save_object("final_models/model.pkl", best_model)

            # Model Trainer Artifact
            model_trainer_artifact = ModelTrainerArtifact(trained_model_file_path=self.model_trainer_config.trained_model_file_path,
                                                          train_metric_artifact=regression_train_metric,
                                                          test_metric_artifact=regression_test_metric)
            
            logging.info(f"Model trainer artifact: {model_trainer_artifact}")
            return model_trainer_artifact
        
        except Exception as e:
            raise FoodDeliveryException(e,sys)
        
    def validate_data(self,X, y, dataset_name):
        if np.any(np.isnan(X)) or np.any(np.isnan(y)):
            raise ValueError(f"{dataset_name} contains NaN values.")
        if np.any(np.isinf(X)) or np.any(np.isinf(y)):
            raise ValueError(f"{dataset_name} contains infinite values.")
        if len(X) == 0 or len(y) == 0:
            raise ValueError(f"{dataset_name} is empty.")    
        

    def initiate_model_trainer(self) -> ModelTrainerArtifact:
        try:
            train_file_path = self.data_transformation_artifact.transformed_train_file_path
            test_file_path = self.data_transformation_artifact.transformed_test_file_path

            # loading training array and testing array
            train_arr = load_numpy_array_data(train_file_path)
            test_arr = load_numpy_array_data(test_file_path)

            x_train, y_train, x_test, y_test = (
                train_arr[:,:-1],
                train_arr[:,-1],
                test_arr[:,:-1],
                test_arr[:,-1]
            )

            # Validate training and testing data
            self.validate_data(x_train, y_train, "Training Data")
            self.validate_data(x_test, y_test, "Testing Data")

            # Train the model
            model = self.train_model(X_train=x_train, y_train=y_train, X_test=x_test, y_test=y_test)

        except Exception as e:
            raise FoodDeliveryException(e,sys)
   