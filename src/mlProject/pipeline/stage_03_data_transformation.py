from pathlib import Path

from mlProject.config.configuration import ConfigurationManager
from mlProject.components.data_transformation import DataTransformation
from mlProject import logger

STAGE_NAME = "Data transformation"

class DataTransformationTrainingPipeline:
    def __init__(self) -> None:
        pass
    
    def main(self):
        try:
            with open(Path("artifacts/data_validation/validation_status.txt"), "r") as f:
                status = f.read().split(" ")[-1]
                
            if "True" in status:
                obj = ConfigurationManager()
                config = obj.get_data_transformation_config()
                data_transfomator = DataTransformation(config)
                data_transfomator.train_test_spliting()
            else:
                raise Exception("Your data schema is not valid")
            
        except Exception as e:
            logger.error(e)
            raise print(e)
        
if __name__ == "__main__":
    try:
        logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
        obj = DataTransformationTrainingPipeline()
        obj.main()
        logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
    except Exception as e:
        logger.exception(e)
        raise e