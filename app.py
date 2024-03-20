import os

from flask import Flask, jsonify, render_template, request

from src.config.confriguration import ConfrigurationManager
from src.exception.exception import CustomException
# Assuming the Prediction class is defined in prediction.py
# and it's correctly handling the prediction logic
from src.pipeline.prediction import Prediction


from main import TrainingStart



# predictor = Prediction(pred_config)

app = Flask(__name__)

# Dummy configuration for the prediction model and scaler paths

# config = Config()



@app.route('/')
def index():
    # Renders the prediction form
    return render_template('index.html')

@app.route('/train')
def train():
     try:
         train=TrainingStart()
         train.run()
     
     except Exception as e:
          raise CustomException(e,sys)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        con=ConfrigurationManager()
        pred_config=con.get_prediction_config()
        predictor = Prediction(pred_config)
        result=predictor.run_pipeline()
        
        # Return the prediction result as JSON
        return render_template('index.html',result=result)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
