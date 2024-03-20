import os
import sys
from dataclasses import dataclass

import pandas as pd

from src.config.confriguration import ConfrigurationManager
from src.constant import *
from src.entity.config_entity import DataIngestionConfig
from src.exception.exception import CustomException
from src.logger.logging import logging
from src.utils.common import create_directories, load_json, load_yaml


class DataIngestion:
     def __init__(self,config:DataIngestionConfig):
          self.config=config
     
     def get_data(self):
          data=pd.read_csv(self.config.source_URL)
          return data
     
     def save_file(self,data):
          save_path=os.path.join(self.config.root_dir,'maintence_costreduction_pred.csv')
          data.to_csv(save_path,index=False)
          
          
@dataclass       
class DataIngestionstart:
     def main(self):
          try:
               config=ConfrigurationManager()
               data_ingestion_config=config.get_data_ingestion_config()
               data_ingestion=DataIngestion(data_ingestion_config)
               data=data_ingestion.get_data()
               data_ingestion.save_file(data)
          except Exception as e:
               raise e