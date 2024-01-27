import os

from sklearn.model_selection import train_test_split
import pandas as pd

from mlProject.entity.config_entity import DataTransformationConfig
from mlProject import logger


class DataTransformation:
    def __init__(self, config: DataTransformationConfig) -> None:
        """ 
        Class to include execute all the traditional ML transformation technics,
        such as PCA, normalizations etc.
        """
        self.config = config
        
    def train_test_spliting(self):
        """ perform train test esplitting """
        
        self.df = pd.read_csv(self.config.data_path)
        train, test = train_test_split(self.df, random_state=42)
        
        logger.info("Train and test data splitted")
        logger.info(train.shape)
        logger.info(test.shape)
        
        train.to_csv(os.path.join(self.config.root_dir, "train.csv"))
        logger.info("train.csv succesfully saved")
        test.to_csv(os.path.join(self.config.root_dir, "test.csv"))
        logger.info("test.csv succesfully saved")