# from src.components.data_ingestion import DataIngestionstart
# from src.components.data_transformation import DataTransformation_start
# from src.components.model_trainer import Modeltrainingstart
# from src.pipeline.training import Modeltrainingstart


# ml=Modeltrainingstart()
# ml.run()

from src.components.data_ingestion import DataIngestionstart
from src.components.data_transformation import DataTransformation_start
from src.components.model_trainer import Modeltrainingstart
from dataclasses import dataclass


@dataclass
class TrainingPipeline:
     def main(self):
          Data_ingestion=DataIngestionstart()
          Data_ingestion.main()
          Data_Transformation=DataTransformation_start()
          Data_Transformation.main1()
          model_trainier=Modeltrainingstart()
          model_trainier.run()
     
     
     
@dataclass
class TrainingStart:
     def run(self):
          model_train=TrainingPipeline()
          model_train.main()
          
