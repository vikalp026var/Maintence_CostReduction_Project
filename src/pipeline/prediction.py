import os
import pickle
import sys

from flask import request
from src.config.confriguration import ConfrigurationManager
from src.constant import *
from src.exception.exception import CustomException
from src.logger.logging import logging
from src.utils.common import load_pickle

# Assuming CustomException is correctly defined elsewhere

class Prediction:
    def __init__(self, config):
        self.config = config

    def load_pickle(self, path):
        with open(path, 'rb') as f:
            return pickle.load(f)

    def prediction(self):
        try:
            if request.method=='POST':
                path = os.path.join(self.config.model_path, 'model.pkl')
                model = self.load_pickle(path)
                print("Model Load success")
                
                scale_path = os.path.join(self.config.scale_path, 'scale.pkl')
                scale = self.load_pickle(scale_path)
                print("Scale load Successfully")
                
                metric1=request.form.get('metric1')
                metric2=float(request.form.get('metric2'))
                metric3=float(request.form.get('metric3'))
                metric4=float(request.form.get('metric4'))
                metric5=request.form.get('metric5')
                metric6=request.form.get('metric6')
                metric7=float(request.form.get('metric7'))
                metric9=float(request.form.get('metric9'))
                months=request.form.get('months')
                day=request.form.get('day')
                model_name=request.form.get('model_name')
                #   model_name_mapping = {'S1F0':0, 'S1F1':1, 'W1F0':2, 'W1F1':3, 'Z1F0':4, 'Z1F1':5, 'Z1F2':6}
                #   model_name_converted = model_name_mapping.get(request.form.get('model_name'), -1)  # Default or error handling

               # Ensure your feature array matches the expected shape and data preprocessing steps
            pred = model.predict(scale.transform([[metric1,metric2,metric3,metric4,metric5,metric6,metric7,metric9,months,day,model_name]]))
            # print("Prediction is :",type(pred[0]))
            if pred[0]==1:
                pred1="Yes, Maintence is necessary "
            else:
                pred1="No, Maintence is not necessary "
            print(pred1)  
            return pred1
        except Exception as e:
            raise CustomException(e, sys) from e
    def run_pipeline(self):
        try:
            logging.info("Enter into the Prediction now Prediction is too bee start.... ")
            result=self.prediction()
            logging.info("Prediction is now Complete ",result)
            return result
        except Exception as e:
            logging.error(e)
            raise CustomException(e,sys) from e
        
# c=ConfrigurationManager()
# config=c.get_prediction_config()       
# pr=Prediction(config)
# pr.run_pipeline()