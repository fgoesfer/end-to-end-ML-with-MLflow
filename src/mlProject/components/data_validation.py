import os
from pathlib import Path

import pandas as pd

from mlProject.entity.config_entity import DataValidationConfig
from mlProject import logger
from mlProject.utils.common import write_string_to_file


class DataValidation:
    def __init__(self, config: DataValidationConfig):
        self.config = config

    def validate_all_columns(self) -> bool:
        try:
            validation_status = False
            df = pd.read_csv(self.config.unzip_data_dir)
            cols = list(df.columns)
            schema = self.config.all_schema

            for col in cols:
                if col not in list(schema.keys()):
                    validation_status = False
                    write_string_to_file(
                        f"Validation status {validation_status} on column {col}",
                        Path(self.config.STATUS_FILE),
                    )
                    break
                elif df[col].dtype != schema[col]:
                    validation_status = False
                    write_string_to_file(
                        f"Validation status {validation_status} on column {col}",
                        Path(self.config.STATUS_FILE),
                    )
                    break
                
                else:
                    validation_status = True
                    write_string_to_file(
                        f"Validation status {validation_status} on column {col}",
                        Path(self.config.STATUS_FILE),
                    )
            validation_status_file = os.path.join(self.config.root_dir, "validation_status.txt")
            write_string_to_file(str(validation_status), Path(validation_status_file))
            return validation_status
        except Exception as e:
            raise e
