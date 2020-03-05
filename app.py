import os

from flask import Flask, jsonify, request
from joblib import load
from dotenv import load_dotenv
import sklearn
import numpy as np
import glob

app = Flask(__name__)
model = None


@app.before_request
def load_resources():
    '''Flask template is currently only for scikit-learn models'''
    load_dotenv()
    # in future don't make user have to write flask server themselves
    # that way don't need to use global b/c not thread-safe & using gunicorn
    global model
    if not model:
        model = load(glob.glob("*joblib")[0])


def transform_data(input_data):
    # will convert from 1D to required 2D
    return np.array(input_data.tolist()[:784]).reshape(1, -1)


@app.route("/", methods=["GET", "POST"])
def predict():
    img_nparray = np.fromstring(request.files["image"].read(), np.uint8)
    transformed_data = transform_data(img_nparray)
    prediction = model.predict(transformed_data).tolist()[0]
    return jsonify({"prediction": prediction})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 8080)))
