from mlProject.components.data_validation import DataValidation
from mlProject.config.configuration import ConfigurationManager
from mlProject import logger

STAGE_NAME = "Data Validation stage"

class DataValidationTrainingPipeline:
    def __init__(self) -> None:
        pass
    def main(self):
        obj = ConfigurationManager()
        config = obj.get_data_validation_config()
        data_validator = DataValidation(config=config)
        data_validator.validate_all_columns()
        
        
if __name__ == "__main__":
    try:
        logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
        obj = DataValidationTrainingPipeline()
        obj.main()
        logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
    except Exception as e:
        logger.exception(e)
        raise e