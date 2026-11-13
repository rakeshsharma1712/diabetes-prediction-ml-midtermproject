from flask import Flask
from flask import request
from flask import jsonify
import xgboost as xgb
import os 
import pickle
print(xgb.__version__)

import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

model_file = os.path.join(
    BASE_DIR,
    'models',
    'xgb_tuned_eta_0_1_depth_4_min_child_60.bin'
)


print("Model path:", model_file)
print("File exists:", os.path.exists(model_file))

with open(model_file, 'rb') as f_in:
    dv, model = pickle.load(f_in)

print("Model loaded successfully!")

app = Flask('diabetes') # give an identity to your web service



@app.route('/predict', methods=['POST']) # use decorator to add Flask's functionality to our function
def predict():
    # parse input which is in JSON format
    patient = request.get_json()

    X = dv.transform([patient])
    Xtest = xgb.DMatrix(X, feature_names=dv.get_feature_names_out().tolist())

    y_pred = model.predict(Xtest)
    diabetes_risk = y_pred >= 0.5

    result = {
        "diabetes probability" : float(y_pred[0]),
        "diabetes_risk" : bool(diabetes_risk[0])
    }
    return jsonify(result)


if __name__ == '__main__':
   app.run(debug=True, host='0.0.0.0', port=9696)

