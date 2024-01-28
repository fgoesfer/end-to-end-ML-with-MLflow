import os
from urllib.parse import urlparse
from pathlib import Path

import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import mlflow
import mlflow.sklearn
import numpy as np
import joblib

from mlProject.entity.config_entity import ModelEvaluationConfig
from mlProject.utils.common import save_json
from mlProject import logger

class ModelEvaluation:
    def __init__(self, config: ModelEvaluationConfig) -> tuple:
        self.config = config
    
    def eval_metrics(self, ytrue: np.array, ypred: np.array):
        """
        Evaluate regression metrics for the predicted values.

        Args:
            ytrue (np.ndarray): Array of true target values.
            ypred (np.ndarray): Array of predicted target values.

        Returns:
            tuple: A tuple containing the root mean squared error (RMSE),
                mean absolute error (MAE), and R-squared (R2) scores.
        """
    
        r2 = r2_score(ytrue, ypred)
        mae = mean_absolute_error(ytrue, ypred)
        rmse = mean_squared_error(ytrue, ypred)
        
        return rmse, mae, r2
    
    def log_into_mlflow(self):
        
        test_data = pd.read_csv(self.config.test_data_path)
        model = joblib.load(self.config.model_path)
        
        test_x = test_data.drop([self.config.target_column], axis=1)
        test_y = test_data[[self.config.target_column]]
        
        
        mlflow.set_registry_uri(self.config.mlflow_uri)
        os.environ['MLFLOW_TRACKING_URI'] = self.config.mlflow_uri
        tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme
        
        params = self.config.all_params
        run_name = ""
        for k in params:
            run_name += f"{k}_{params[k]}_"
        run_name = run_name[:-1]
        with mlflow.start_run(run_name=run_name):
            logger.info(">>> Mlflow run started")
            predicted_qualities = model.predict(test_x)
            (rmse, mae, r2) = self.eval_metrics(test_y, predicted_qualities)
            
            # saving metrics as local
            logger.info(">>> Saving metrics")
            scores = {"rmse": rmse, "mae": mae, "r2": r2}
            save_json(path=Path(self.config.metric_file_name), data=scores)
            
            mlflow.log_params(self.config.all_params)
            
            mlflow.log_metrics(scores)
            
            # Model registry does not work with file store
            logger.info(">>> Model registration")
            mlflow.sklearn.log_model(model, "model", registered_model_name="ElasticnetModel")
            if tracking_url_type_store != "file":

                # Register the model
                # There are other ways to use the Model Registry, which depends on the use case,
                # please refer to the doc for more information:
                # https://mlflow.org/docs/latest/model-registry.html#api-workflow
                mlflow.sklearn.log_model(model, "model", registered_model_name="ElasticnetModel")
            else:
                mlflow.sklearn.log_model(model, "model")
