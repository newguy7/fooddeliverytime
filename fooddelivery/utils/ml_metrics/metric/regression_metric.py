import sys
import numpy as np

from fooddelivery.entity.artifact_entity import RegressionMetricArtifact
from fooddelivery.exception.exception import FoodDeliveryException
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.metrics import mean_absolute_percentage_error, r2_score

def get_regression_value(y_true, y_pred) -> RegressionMetricArtifact:
    try:
        model_mae_value = mean_absolute_error(y_true, y_pred)
        model_rmse_value = np.sqrt(mean_squared_error(y_true, y_pred))
        model_mape_value = mean_absolute_percentage_error(y_true, y_pred)
        model_r2_value = r2_score(y_true, y_pred)

        regression_metric = RegressionMetricArtifact(mae_value=model_mae_value,
                                                     rmse_value=model_rmse_value,
                                                     r2_value=model_r2_value,
                                                     mape_value=model_mape_value)
        
        return regression_metric
    except Exception as e:
        raise FoodDeliveryException(e,sys)
