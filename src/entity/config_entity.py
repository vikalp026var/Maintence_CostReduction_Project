from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class DataIngestionConfig:
     root_dir: Path
     source_URL: str
     
     
@dataclass(frozen=True)
class DataTransformationConfig:
     data_path: Path
     train_path: Path
     test_path:  Path
     cleaned_data: Path
     
     
@dataclass(frozen=True)
class ModelTrainerConfig:
     train_path: Path
     test_path:  Path
     model_path: Path
     
     
@dataclass(frozen=True)
class PredictionConfig:
   scale_path: Path
   model_path: Path
     