import sys
import os
import requests
import numpy as np
import pandas as pd

from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer

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

    def get_data_transformer_object(cls, numerical_columns: list, categorical_columns: list) -> Pipeline:
        """
        Creates a ColumnTransformer pipeline with preprocessing steps for both numerical and categorical data.
        
        Numerical:
            - Impute missing values with mean.
            - Standardize features with StandardScaler.
            
        Categorical:
            - Impute missing values with the most frequent value.
            - Apply OneHotEncoder to convert categories to numeric form.

        Args:
            cls: DataTransformation
            numerical_columns: List of numerical column names.
            categorical_columns: List of categorical column names.

        Returns:
            A ColumnTransformer object for preprocessing.
        """
        try:
            # Categorical preprocessing: Impute and one-hot encode
            categorical_pipeline = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='most_frequent')),  # Impute missing with most frequent
                ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))  # One-hot encode
            ])

            # Numerical preprocessing: Impute and scale
            numerical_pipeline = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='mean')),  # Impute missing with mean
                ('scaler', StandardScaler())  # Scale features
            ])

            # Combine pipelines in a ColumnTransformer
            processor = ColumnTransformer(
                transformers=[
                    ('num', numerical_pipeline, numerical_columns),
                    ('cat', categorical_pipeline, categorical_columns)
                ]
            )

            logging.info("Created data transformer pipeline successfully.")
            return processor

        except Exception as e:
            raise FoodDeliveryException(e, sys)

        
    def initiate_data_transformation(self) -> DataTransformationArtifact:
        logging.info("Entered initiate_data_transformation method of DataTransformation class")


        try:            
            logging.info("Starting Data Transformation")
            train_df = DataTransformation.read_data(self.data_validation_artifact.valid_train_file_path)
            test_df = DataTransformation.read_data(self.data_validation_artifact.valid_test_file_path)

            logging.info(f"Training data shape before cleaning: {train_df.shape}")
            logging.info(f"Testing data shape before cleaning: {test_df.shape}")

            # Columns to drop            
            # Columns - ID and Delivery_person_ID are only used for tracking purpose. It wont impact the delivery time prediction            
            columns_to_drop = ['ID', 'Delivery_person_ID']
            train_df.drop(columns=columns_to_drop, inplace=True, errors='ignore')
            test_df.drop(columns=columns_to_drop, inplace=True, errors='ignore')

            # Since missing values for Restaurant_latitude, Restaurant_longitude means the starting point is unknown, 
            # we will remove the rows that has these nan values            

            # The output column "TARGET" has 541 missing values. We dont want to include generated data for this column as it impacts the models learning accuracy.
            # So we will remove the rows that has nan values for column TARGET
            
            # Remove rows with missing values in critical columns
            critical_columns = ['Restaurant_latitude', 'Restaurant_longitude', 'TARGET']
            train_df.dropna(subset=critical_columns, inplace=True)
            test_df.dropna(subset=critical_columns, inplace=True)

            # For Distance (km), we will use function to calculate distance using OSRM API to fill the missing values. 
            # train_df['Distance (km)'] = train_df.apply(self.calculate_distance, axis=1)
            # test_df['Distance (km)'] = test_df.apply(self.calculate_distance, axis=1)

            # After applying calculate distance function, if there are still some nan values, we will remove the rows.
            train_df.dropna(subset=['Distance (km)'], inplace=True)
            test_df.dropna(subset=['Distance (km)'], inplace=True)

            # Ensure dataframes are not empty
            if train_df.empty or test_df.empty:
                raise FoodDeliveryException("Training or testing data is empty after preprocessing.", sys)

            logging.info(f"Training data shape after cleaning: {train_df.shape}")
            logging.info(f"Testing data shape after cleaning: {test_df.shape}")
            
            # Separate input features and target variable
            # training dataframe
            input_feature_train_df = train_df.drop(columns=[TARGET_COLUMN],axis=1)
            target_feature_train_df = train_df[TARGET_COLUMN]
            # Ensure target column is numeric
            target_feature_train_df = pd.to_numeric(target_feature_train_df, errors='coerce')

            # testing dataframe
            input_feature_test_df = test_df.drop(columns=[TARGET_COLUMN],axis=1)
            target_feature_test_df = test_df[TARGET_COLUMN]
            # Ensure target column is numeric
            target_feature_test_df = pd.to_numeric(target_feature_test_df, errors='coerce')

            # Verify the target variable
            logging.info(f"Target column (train) dtype: {target_feature_train_df.dtype}")
            logging.info(f"Target column (test) dtype: {target_feature_test_df.dtype}")

            # # Dynamically detect numerical and categorical columns
            numerical_columns = input_feature_train_df.select_dtypes(include=['int64', 'float64']).columns.tolist()
            categorical_columns = input_feature_train_df.select_dtypes(include=['object', 'category']).columns.tolist()

            logging.info(f"Numerical columns: {numerical_columns}")
            logging.info(f"Categorical columns: {categorical_columns}")

            # check for any leading or trailing whitespace in categorical columns and remove them
            for col in categorical_columns:
                input_feature_train_df[col] = input_feature_train_df[col].str.strip()
                input_feature_test_df[col] = input_feature_test_df[col].str.strip()

            # # Restrict columns to numerical and categorical columns for train and test
            # input_feature_train_df = input_feature_train_df[numerical_columns + categorical_columns]

            # Align test dataset columns with train dataset
            input_feature_test_df = input_feature_test_df.reindex(columns=input_feature_train_df.columns, fill_value=0)            

            # Get the preprocessing object
            processor = self.get_data_transformer_object(numerical_columns=numerical_columns, categorical_columns=categorical_columns)
            preprocessor_object = processor.fit(input_feature_train_df)            
            transformed_input_train_feature = preprocessor_object.transform(input_feature_train_df)
            transformed_input_test_feature = preprocessor_object.transform(input_feature_test_df)            

            # Combine transformed features with the target variable
            train_arr = np.c_[transformed_input_train_feature,np.array(target_feature_train_df)]
            test_arr = np.c_[transformed_input_test_feature,np.array(target_feature_test_df)]            

            # save numpy array data
            save_numpy_array_data(self.data_transformation_config.transformed_train_file_path, array=train_arr)
            save_numpy_array_data(self.data_transformation_config.transformed_test_file_path, array=test_arr)
            save_object(self.data_transformation_config.transformed_object_file_path, preprocessor_object)

            # Save it in the final model folder for model pusher purpose
            save_object("final_models/preprocessor.pkl", preprocessor_object)

            # preparing artifacts
            data_transformation_artifacts = DataTransformationArtifact(
                transformed_object_file_path=self.data_transformation_config.transformed_object_file_path,
                transformed_train_file_path=self.data_transformation_config.transformed_train_file_path,
                transformed_test_file_path=self.data_transformation_config.transformed_test_file_path
            )

            return data_transformation_artifacts
        
        except Exception as e:
            raise FoodDeliveryException(e,sys)



            


