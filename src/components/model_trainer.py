import os
import sys
from dataclasses import dataclass
from pathlib import Path
import pickle
import numpy as np
import pandas as pd
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from src.config.confriguration import ConfrigurationManager
from src.constant import *
# from src.entity.config_entity import DataTransformationConfig
from src.exception.exception import CustomException
from src.logger.logging import logging
from src.utils.common import create_directories, load_json, load_yaml

models={
     'LogisticRegression':LogisticRegression(),
     'SVC':SVC(),
     'DecisionTreeClassifier':DecisionTreeClassifier(),
     'GaussianNB':GaussianNB(),
     'RandomForestClassifier':RandomForestClassifier(),
     'KNeighborsClassifier':KNeighborsClassifier(),
     'AdaBoostClassifier':AdaBoostClassifier(learning_rate=1)
}



class ModelTrainer:
     def __init__(self, config):
          self.config = config
          
     def read_data(self):
          try:
              train_data = pd.read_csv(self.config.train_path)
              test_data = pd.read_csv(self.config.test_path)
              return train_data, test_data
          except Exception as e:
               raise CustomException(e, sys)
          
     def split_data(self):
          try:
               train_data, test_data = self.read_data()
               X_train = train_data.drop(columns=['failure'])
               y_train = train_data['failure']
               X_test = test_data.drop(columns=['failure'])
               y_test = test_data['failure']
               return X_train, X_test, y_train, y_test
          
          except Exception as e:
               raise CustomException(e, sys)
          
     def standard_scale(self):
          try:
               X_train, X_test, y_train, y_test = self.split_data()
               model_directory = os.path.join(os.getcwd(), self.config.model_path)
               if not os.path.exists(model_directory):
                    print(f"Creating directory: {model_directory}")  # Debug print
                    os.makedirs(model_directory)
               else:
                    print(f"Directory already exists: {model_directory}")  # Debug print
               scale_path = os.path.join(model_directory, 'scale.pkl')
               scaler = StandardScaler()
               print("Standarad Scale saved successfully!")
               X_train_scaled = scaler.fit_transform(X_train)
               X_test_scaled = scaler.transform(X_test)
               with open(scale_path, 'wb') as f:
                    pickle.dump(scaler, f)
               print("Standarad Scale saved successfully!")
               return X_train_scaled, X_test_scaled, y_train, y_test
          except Exception as e:
               raise CustomException(e, sys)
     
     def model_train(self, models):
          model_accuracies = {}
          model_objects = {}
          try:
               X_train_scaled,X_test_scaled,  y_train, y_test = self.standard_scale()
               for model_name, model in models.items():
                    model.fit(X_train_scaled, y_train)
                    y_pred = model.predict(X_test_scaled)
                    accuracy = accuracy_score(y_test, y_pred)
                    model_accuracies[model_name] = accuracy
                    model_objects[model_name] = model 
                    print(f"{model_name}: Accuracy score is: {accuracy}")
                    print(confusion_matrix(y_test, y_pred))
                    print("="*40)
               return model_accuracies, model_objects, X_train_scaled, y_train
          except Exception as e:
               raise CustomException(e, sys)
          
     def get_best_model(self, models):
        try:
            model_accuracies, model_objects, X_train_scaled, y_train = self.model_train(models)
            best_model_name = max(model_accuracies, key=model_accuracies.get)
            best_model = model_objects[best_model_name]  # Retrieve the best model object
            best_accuracy = model_accuracies[best_model_name]
            print(f"Best Model: {best_model_name}")
            return best_model, best_accuracy, X_train_scaled, y_train
        except Exception as e:
            raise CustomException(e, sys)
    
     def save_model_best(self, models):
        try:
            best_model, best_accuracy, X_train_scaled, y_train = self.get_best_model(models)
            best_model.fit(X_train_scaled, y_train)  # Ensure best_model is the actual model object
            
            # Construct the directory path for the model
            model_directory = os.path.join(os.getcwd(), self.config.model_path)
            print(f"Model directory: {model_directory}")  # Debug print
            
            # Check if the directory exists, and if not, create it
            if not os.path.exists(model_directory):
                print(f"Creating directory: {model_directory}")  # Debug print
                os.makedirs(model_directory)
            else:
                print(f"Directory already exists: {model_directory}")  # Debug print
            
            # Construct the full path for the model
            model_path = os.path.join(model_directory, 'model.pkl')
            print(f"Saving model to: {model_path}")  # Debug print
            
            # Save the model
            with open(model_path, 'wb') as f:
                pickle.dump(best_model, f)
            print("Model saved successfully!")  # Debug print
            
        except Exception as e:
            raise CustomException(e, sys)
       
       
@dataclass
class Modeltrainingstart:
     def run(self):
          try:
               config=ConfrigurationManager()
               data_ingestion_config=config.get_model_trainer_config()
               data_ingestion=ModelTrainer(data_ingestion_config)
               data=data_ingestion.save_model_best(models)
               # data_ingestion.save_file(data)
          except Exception as e:
               raise e