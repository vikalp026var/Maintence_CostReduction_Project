import os
import sys
from dataclasses import dataclass
from pathlib import Path
from imblearn.over_sampling import SMOTE
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder,StandardScaler
from src.config.confriguration import ConfrigurationManager
from src.constant import *
from src.entity.config_entity import DataTransformationConfig
from src.exception.exception import CustomException
from src.logger.logging import logging
from src.utils.common import create_directories, load_json, load_yaml


class Data_Transformation:
     def __init__(self,config:DataTransformationConfig):
          self.config=config
     
     def convert_date(self,df):
          try:
               df['date']=pd.to_datetime(df['date'])

               df['Year'] = df['date'].apply(lambda x: x.year)
               df['months'] = df['date'].apply(lambda x: x.month)
               df['day'] = df['date'].apply(lambda x: x.day)
               # print(df)

               df.drop(columns=['Year'],inplace=True,axis=1)
               return df
          except Exception as e:
               raise CustomException(e,sys)

     
     def convert_skewness(self,data):
          try:
               date=self.convert_date(data)
               for num in ["2","3","4","7","8","9"]:
                    date[f'metric{num}'] = np.log1p(date[f'metric{num}'])
               date.drop(columns=['metric8'],inplace=True,axis=1)
               return date
          except Exception as e:
               raise CustomException(e)
          
          
          
     def convert_model_name(self,data):
          try:
               data1=self.convert_skewness(data)
               data1['model_name']=data1['device'].apply(lambda x:x[:4])
               # Assuming `data` is your DataFrame and you want to remove rows where `model_name` equals 'Z1F2'
               # data = data[data['model_name'] != 'Z1F2']
               label=LabelEncoder()
               data1['model_name']=label.fit_transform(data1['model_name'])
               return data1
               

          except Exception as e:
               raise CustomException(e,sys)
          
     def clean_data(self,data):
          try:
               data2=self.convert_model_name(data)
               data2.drop(columns=['date','device'],inplace=True,axis=1)
               # print(data2)
               save_path=os.path.join(self.config.cleaned_data,'cleaned_data')
               
               # import os

               # # Assuming `save_path` is the full path to the file, including the filename
               # save_path = os.path.join(self.config.cleaned_data, 'cleaned_data.csv')

               # Create the directory if it does not exist
               os.makedirs(os.path.dirname(save_path), exist_ok=True)

               # Now you can safely save the DataFrame
               # data2.to_csv(save_path, index=False)

               data2.to_csv(save_path,index=False)
               return data2
               
          except Exception as e:
               raise CustomException(e,sys)
          
     def oversampling(self,data):
          try:
               data_over=self.clean_data(data)
               oversample=SMOTE()
               X_over,y_over=oversample.fit_resample(data_over[[ 'metric1', 'metric2', 'metric3', 'metric4', 'metric5',
       'metric6', 'metric7', 'metric9', 'months', 'day', 'model_name']],data_over['failure'])
               return X_over,y_over
          except Exception as e:
               raise CustomException(e,sys) 
          
          
     def train_test_split(self):
          try:
               data1=pd.read_csv(self.config.data_path)
               X,y=self.oversampling(data1)
               data=pd.concat([X,y],axis=1)
               train_data,test_data=train_test_split(data,test_size=0.22,random_state=42)
               # train_data.drop(columns=['date','device'],inplace=True,axis=1)
               # test_data.drop(columns=['date','device'],inplace=True,axis=1)
               save_path1=os.path.join(self.config.train_path,'train_data.csv')
               os.makedirs(os.path.dirname(save_path1), exist_ok=True)
               # data.to_csv(save_path,index=False)
               save_path2=os.path.join(self.config.test_path,'test_data.csv')
               os.makedirs(os.path.dirname(save_path2), exist_ok=True)
               train_data.to_csv(save_path1,index=False)
               test_data.to_csv(save_path2,index=False)
               return train_data,test_data
               
          except Exception as e:
               raise CustomException(e,sys)
               
               
@dataclass
class DataTransformation_start:
     def main1(self):
          try:
               config=ConfrigurationManager()
               logging.info("Start....")
               data_ingestion_config=config.get_data_tranformation_initiate()
               logging.info("Call the Data Transformation")
               data_ingestion=Data_Transformation(data_ingestion_config)
               logging.info("Object of data Transformation")
               train,test=data_ingestion.train_test_split()
               # data_ingestion.save_file(data)
          except Exception as e:
               raise e