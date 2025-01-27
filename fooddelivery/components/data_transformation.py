import sys
import os
import requests
import numpy as np
import pandas as pd

from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

from fooddelivery.constant.training_pipeline import TARGET_COLUMN
from fooddelivery.constant.training_pipeline import DATA_TRANSFORMATION_IMPUTER_PARAMS

from fooddelivery.entity.artifact_entity import DataValidationArtifact, DataTransformationArtifact
from fooddelivery.entity.config_entity import DataTransformationConfig
from fooddelivery.exception.exception import FoodDeliveryException
from fooddelivery.logging.logger import logging
from fooddelivery.utils.main_utils.utils import save_numpy_array_data, save_object

class DataTransformation:
    def __init__(self,data_validation_artifact: DataValidationArtifact,
                 data_transformation_config:DataTransformationConfig):
        try:
            self.data_validation_artifact:DataValidationArtifact = data_validation_artifact
            self.data_transformation_config:DataTransformationConfig = data_transformation_config
        except Exception as e:
            raise FoodDeliveryException(e,sys)
        
    @staticmethod
    def read_data(file_path) -> pd.DataFrame:
        try:
            return pd.read_csv(file_path)
        except Exception as e:
            raise FoodDeliveryException(e,sys)
        
    @staticmethod
    def calculate_distance(row):
        try:
            # Extract latitude and longitude value from the row
            restaurant_lat = row['Restaurant_latitude']
            restaurant_lon = row['Restaurant_longitude']
            delivery_lat = row['Delivery_location_latitude']
            delivery_lon = row['Delivery_location_longitude']

            # Define the OSRM API endpoint
            osrm_endpoint = "http://router.project-osrm.org/route/v1/driving/"

            # Build the request URL
            coordinates = f"{restaurant_lon},{restaurant_lat};{delivery_lon},{delivery_lat}"
            url = f"{osrm_endpoint}{coordinates}?overview=false"

            # Send the request to OSRM API
            response = requests.get(url)
            data = response.json()

            if response.status_code == 200 and 'routes' in data:
                return data['routes'][0]['distance'] / 1000  # Convert to kilometers
            else:
                logging.warning(f"Failed for row {row.name}: {data}")
                return None
        except Exception as e:
            logging.error(f"Error calculating distance for row {row.name}: {e}")
            raise FoodDeliveryException(e, sys)

    def get_data_transformer_object(cls) -> Pipeline:
        """
        It initialises a SimpleImputer object with the parameters specified in the training_pipeline.py file
        and returns a pipeline object with the SimpleImputer object as the first step.

        Args:
            cls: DataTransformation

        Returns:
            A pipeline object
        """
        try:
            imputer = SimpleImputer(**DATA_TRANSFORMATION_IMPUTER_PARAMS)
            logging.info(
                f"Initialise SimpleImputer with {DATA_TRANSFORMATION_IMPUTER_PARAMS}"
            )
            processor:Pipeline = Pipeline([("imputer", imputer)])
            return processor
        except Exception as e:
            raise FoodDeliveryException(e,sys)
        
    def initiate_data_transformation(self) -> DataTransformationArtifact:
        logging.info("Entered initiate_data_transformation method of DataTransformation class")

        try:
            logging.info("Starting Data Transformation")
            train_df = DataTransformation.read_data(self.data_validation_artifact.valid_train_file_path)
            test_df = DataTransformation.read_data(self.data_validation_artifact.valid_test_file_path)

            # Column Unnamed: 14 only has NaN value. 
            # Columns - ID and Delivery_person_ID are only used for tracking purpose. It wont impact the delivery time prediction
            
            columns_to_drop = ['Unnamed: 14', 'ID', 'Delivery_person_ID']
            train_df.drop(columns=columns_to_drop, inplace=True, errors='ignore')
            test_df.drop(columns=columns_to_drop, inplace=True, errors='ignore')

            # Since missing values for Restaurant_latitude, Restaurant_longitude means the starting point is unknown, 
            # we will remove the rows that has these nan values
            train_df.dropna(subset=['Restaurant_latitude', 'Restaurant_longitude'],inplace=True)
            test_df.dropna(subset=['Restaurant_latitude', 'Restaurant_longitude'],inplace=True)

            # For Distance (km), we will use function to calculate distance using OSRM API to fill the missing values. 
            # train_df['Distance (km)'] = train_df.apply(self.calculate_distance, axis=1)
            # test_df['Distance (km)'] = test_df.apply(self.calculate_distance, axis=1)

            # After applying calculate distance function, if there are still some nan values, we will remove the rows.
            train_df.dropna(subset=['Distance (km)'], inplace=True)
            test_df.dropna(subset=['Distance (km)'], inplace=True)

            # The output column "TARGET" has 541 missing values. We dont want to include generated data for this column as it impacts the models learning accuracy.
            # So we will remove the rows that has nan values for column TARGET
            train_df.dropna(subset=['TARGET'], inplace=True)
            test_df.dropna(subset=['TARGET'], inplace=True)

            # training dataframe
            input_feature_train_df = train_df.drop(columns=[TARGET_COLUMN],axis=1)
            target_feature_train_df = train_df[TARGET_COLUMN]

            # testing dataframe
            input_feature_test_df = test_df.drop(columns=[TARGET_COLUMN],axis=1)
            target_feature_test_df = test_df[TARGET_COLUMN]

            # initialize get_data_transformer_object to apply SimpleImputer (on column 'Traffic_Level')
            processor = self.get_data_transformer_object()
            preprocessor_object = processor.fit(input_feature_train_df[['Traffic_Level']])

            ## if Imputer was applied to entire df
            # preprocessor_object = processor.fit(input_feature_train_df)
            # transformed_input_train_feature = preprocessor_object.transform(input_feature_train_df)
            # transformed_input_test_feature = preprocessor_object.transform(input_feature_test_df)

            # Ensure the output is 1D when assigning to a single column
            input_feature_train_df['Traffic_Level'] = preprocessor_object.transform(input_feature_train_df[['Traffic_Level']])[:, 0]
            input_feature_test_df['Traffic_Level'] = preprocessor_object.transform(input_feature_test_df[['Traffic_Level']])[:, 0]

            # train_arr = np.c_[transformed_input_train_feature,np.array(target_feature_train_df)]
            # test_arr = np.c_[transformed_input_test_feature,np.array(target_feature_test_df)]

            # The SimpleImputer transformation only affects a specific column (Traffic_Level) within the pandas DataFrame. 
            # The features are still pandas DataFrames, so we need to convert the entire feature set into a NumPy array.
            train_arr = np.c_[input_feature_train_df.to_numpy(), target_feature_train_df.to_numpy()]
            test_arr = np.c_[input_feature_test_df.to_numpy(), target_feature_test_df.to_numpy()]

            # save numpy array data
            save_numpy_array_data(self.data_transformation_config.transformed_train_file_path, array=train_arr)
            save_numpy_array_data(self.data_transformation_config.transformed_test_file_path, array=test_arr)
            save_object(self.data_transformation_config.transformed_object_file_path, preprocessor_object)

            # Save it in the final model folder for model pusher purpose
            # save_object("final_models/preprocessor.pkl", preprocessor_object)

            # preparing artifacts
            data_transformation_artifacts = DataTransformationArtifact(
                transformed_object_file_path=self.data_transformation_config.transformed_object_file_path,
                transformed_train_file_path=self.data_transformation_config.transformed_train_file_path,
                transformed_test_file_path=self.data_transformation_config.transformed_test_file_path
            )

            return data_transformation_artifacts
        
        except Exception as e:
            raise FoodDeliveryException(e,sys)



            


