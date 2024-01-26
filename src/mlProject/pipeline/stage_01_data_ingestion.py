from mlProject.config.configuration import ConfigurationManager
from mlProject.components.data_ingestion import DataIngestion
from mlProject import logger


STAGE_NAME = "Data Ingestion stage"

class DataIngestionTrainingPipeline:
    def __init__(self) -> None:
        pass
    def main(self):
        obj = ConfigurationManager() # criação do manager
        config = obj.get_data_ingestion_config() # pegar a configuração do yaml
        data_ingestor = DataIngestion(config=config)
        data_ingestor.download_file() # fazer o download zip dos arquivos
        data_ingestor.extract_zip_file() # extrair os arquivos
        
if __name__ == "__main__":
    try:
        logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
        obj = DataIngestionTrainingPipeline()
        obj.main()
        logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
    except Exception as e:
        logger.exception(e)