import os

from flask import Flask, jsonify
from shaktiutils.gcp_utils.googlebucket import gcs_download_file
from joblib import load
from dotenv import load_dotenv

app = Flask(__name__)

model = None


@app.before_request
def load_resources():
    '''Flask template is currently only for scikit-learn models'''
    load_dotenv()
    if not model:
        model_path = gcs_download_file(os.environ.get("MODEL_PATH"))
        model = load(model_path)


def transform_data(input_data):
    # no transform by default
    return input_data


@app.route('/')
def predict(input_data):
    transformed_data = transform_data(input_data)
    prediction = model.predict(transformed_data)
    return jsonify({"prediction": prediction})


if __name__ == "__main__":
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 8080)))