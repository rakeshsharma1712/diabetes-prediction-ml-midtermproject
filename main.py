from flask import Flask
import xgboost as xgb

app = Flask('ping') # give an identity to your web service

@app.route('/ping', methods=['GET']) # use decorator to add Flask's functionality to our function
def ping():
    return 'PONG'

if __name__ == '__main__':
   app.run(debug=True, host='0.0.0.0', port=9696)


def pred(patient):
    X = dv.transform([patient])
    Xtest = xgb.DMatrix(X, feature_names=dv.get_feature_names())
    y_pred = model.predict(Xtest)
    return y_pred

