from pathlib import Path

from src.constant import *
from src.entity.config_entity import (DataIngestionConfig,
                                      DataTransformationConfig,
                                      ModelTrainerConfig, PredictionConfig)
from src.utils.common import create_directories, load_yaml


class ConfrigurationManager:
     def __init__(self,
                  config_filepath=CONFIG_PATH,
                  params_filepath=PARAMS_PATH,
                  schema_filepath=SCHEMA_PATH):
          self.config=load_yaml(config_filepath)
          self.params=load_yaml(params_filepath)
          self.schema=load_yaml(schema_filepath)
          create_directories([Path(self.config['artifacts_root'])])
     
     def get_data_ingestion_config(self)->DataIngestionConfig:
          config=self.config.data_ingestion
          create_directories([Path(config['root_dir'])])
          data_ingestion_config=DataIngestionConfig(
               root_dir=config.root_dir,
               source_URL=config.source_URL
          )
          return data_ingestion_config
     
     def get_data_tranformation_initiate(self)->DataTransformationConfig:
          config=self.config.data_transformation
          
          data_tranfromation_config=DataTransformationConfig(
               data_path=config.data_path,
               train_path=config.train_path,
               test_path=config.test_path,
               cleaned_data=config.cleaned_data
               
          )
          return data_tranfromation_config
          
     def get_model_trainer_config(self)->ModelTrainerConfig:
          config=self.config.model_trainer
          
          model_trainer_config=ModelTrainerConfig(
               train_path=config.train_path,
               test_path=config.test_path,
               model_path=config.model_path
               
          )
          return model_trainer_config
     
     def get_prediction_config(self)->PredictionConfig:
          config=self.config.prediction_trainer
          
          model_trainer_config=PredictionConfig(
               scale_path=config.scale_path,
               model_path=config.model_path
               
          )
          return model_trainer_config
          