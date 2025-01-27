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