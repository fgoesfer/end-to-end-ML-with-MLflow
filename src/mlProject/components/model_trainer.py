import os

import pandas as pd
from sklearn.linear_model import ElasticNet

import joblib
from mlProject import logger
from mlProject.entity.config_entity import ModelTrainerConfig


class ModelTrainer:
    def __init__(self, config: ModelTrainerConfig):
        self.config = config

    def trainer(self):
        train_data = pd.read_csv(self.config.train_data_path)

        x_train = train_data.drop([self.config.target_column], axis=1)

        y_train = train_data[self.config.target_column]

        # model
        lr = ElasticNet(
            alpha=self.config.alpha, l1_ratio=self.config.l1_ratio, random_state=42
        )
        logger.info(">>> Fitting model")
        lr.fit(x_train, y_train)
        logger.info(">>> Model fitted")
        joblib.dump(lr, os.path.join(self.config.root_dir, self.config.model_name))
        