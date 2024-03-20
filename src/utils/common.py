import json
import os
import pickle
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import yaml
from box import ConfigBox
from ensure import ensure_annotations

from src.exception.exception import CustomException
from src.logger.logging import logging


@ensure_annotations
def create_directories(path: list,verbose=True):
     logging.info("Enter into the create directores in method")
     try:
          for pah in path:
               os.makedirs(pah,exist_ok=True)
               if verbose:
                    logging.info(f"{pah} is create >>>>")
     except Exception as e:
          raise CustomException(e,sys)
     
@ensure_annotations    
def load_yaml(path:Path):
     logging.info(f"start the loading  the {path} file ")
     try:
          with open(path) as f:
               content=yaml.safe_load(f)
          logging.info(f"Load the {path} successfully")
          return ConfigBox(content)
     except Exception as e:
          raise CustomException(e,sys)
     
          
          
@ensure_annotations
def save_json(path: Path, data: dict):
    with open(path, "w") as f:
        json.dump(data, f, indent=4)

    logging.info(f"json file saved at: {path}")



@ensure_annotations
def load_json(path: Path) -> ConfigBox:
    with open(path) as f:
        content = json.load(f)

    logging.info(f"json file loaded succesfully from: {path}")
    return ConfigBox(content)

@ensure_annotations
def load_bin(path: Path) -> Any:
    data = joblib.load(path)
    logging.info(f"binary file loaded from: {path}")
    return data



@ensure_annotations
def get_size(path: Path) -> str:
    size_in_kb = round(os.path.getsize(path)/1024)
    return f"~ {size_in_kb} KB"


@ensure_annotations
def save_model(path,model):
    with open(path,'wb') as f:
        pickle.dump(model,f)
        
    logging.info("save the best model ")
    
@ensure_annotations
def load_pickle(file_path):
    logging.info("Enter into the load pickle method")
    with open(file_path,'rb') as file:
        model=pickle.load(file)
    logging.info("Successfully load the model ")
    return model