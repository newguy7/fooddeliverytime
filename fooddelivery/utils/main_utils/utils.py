import yaml
from fooddelivery.exception.exception import FoodDeliveryException
from fooddelivery.logging.logger import logging
import os
import sys
import numpy as np
import pandas as pd
import pickle
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import r2_score
from sklearn.utils.estimator_checks import check_estimator
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor
from sklearn.tree import DecisionTreeRegressor


def read_yaml_file(file_path: str) -> dict:
    try:
        with open(file_path, "rb") as yaml_file:
            return yaml.safe_load(yaml_file)
    except Exception as e:
        raise FoodDeliveryException(e,sys)


def write_yaml_file(file_path: str, content: object, replace: bool = False) -> None:
    """
    Writes a Python object to a YAML file. Handles numpy types and invalid values like np.nan.

    Args:
        file_path (str): The path to the YAML file.
        content (object): The Python object to write to the file.
        replace (bool): If True, replaces the existing file if it exists. Default is False.
    """
    def convert_numpy_to_python(obj):
        """Recursively converts numpy objects to native Python types."""
        if isinstance(obj, np.generic):  # Converts numpy scalar types
            return obj.item()
        elif isinstance(obj, dict):  # Recursively convert dictionary
            return {key: convert_numpy_to_python(value) for key, value in obj.items()}
        elif isinstance(obj, list):  # Recursively convert list
            return [convert_numpy_to_python(item) for item in obj]
        elif isinstance(obj, float) and np.isnan(obj):  # Handle NaN values
            return None
        return obj  # Return the object as is if not a numpy type
    
    try:
        # Handle numpy objects in the content
        content = convert_numpy_to_python(content)

        if replace:
            if os.path.exists(file_path):
                os.remove(file_path)
        os.makedirs(os.path.dirname(file_path),exist_ok=True)
        with open(file_path, "w") as file:
            yaml.dump(content, file, default_flow_style=False, Dumper=yaml.SafeDumper)
    except Exception as e:
        raise FoodDeliveryException(e,sys)
    
def save_numpy_array_data(file_path: str, array: np.array):
    """
    Save numpy array data to file
    file_path: str location of file to save
    array: np.array data to save
    """
    try:      

        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path,exist_ok=True)
        with open(file_path, "wb") as file_obj:
            np.save(file_obj, array)
    except Exception as e:
        raise FoodDeliveryException(e.sys) from e
    
def save_object(file_path: str, obj:object) -> None:
    try:
        logging.info("Entered the save_object method of MainUtils class")
        os.makedirs(os.path.dirname(file_path),exist_ok=True)
        with open(file_path, "wb") as file_obj:
            pickle.dump(obj,file_obj)
        logging.info("Exited the save_object method of MainUtils class")

    except Exception as e:
        raise FoodDeliveryException(e,sys) from e
    
def load_object(file_path: str) -> object:
    try:
        if not os.path.exists(file_path):
            raise Exception(f"The file: {file_path} does not exists.")
        with open(file_path, "rb") as file_obj:
            print(file_obj)
            return pickle.load(file_obj)
    except Exception as e:
        raise FoodDeliveryException(e,sys)
    

def load_numpy_array_data(file_path: str) -> np.array:
    """
    load numpy array data from file
    file_path: str location of file to load
    return: np.array data loaded
    """
    try:
        with open(file_path, "rb") as file_obj:
            return np.load(file_obj, allow_pickle=True)
    except Exception as e:
        raise FoodDeliveryException(e,sys)
    
def evaluate_models(X_train, y_train, X_test, y_test, models, param):
    try:
        report = {}

        for i in range(len(list(models))):
            model_name = list(models.keys())[i]
            model = list(models.values())[i]
            para = param[model_name]                 

            gs = GridSearchCV(model, para, cv=3, scoring='r2', error_score=np.nan)            
            gs.fit(X_train, y_train)

            # Set best parameters and retrain the model
            model.set_params(**gs.best_params_)
            model.fit(X_train,y_train)

            # Predict on training and testing data
            y_train_pred = model.predict(X_train)
            y_test_pred = model.predict(X_test)

            # Calculate R2 scores
            train_model_score = r2_score(y_train, y_train_pred)
            test_model_score = r2_score(y_test, y_test_pred)

            # Store the model performance
            report[model_name] = test_model_score

        return report
    except Exception as e:
        raise FoodDeliveryException(e,sys)

