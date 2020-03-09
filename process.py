from flask import Flask, jsonify, request
import numpy as np

from __main__ import app


def preprocess(input_data):
    # will convert from 1D to required 2D
    img_nparray = np.fromstring(request.files["image"].read(), np.uint8)
    reshaped_data = np.array(img_nparray.tolist()[:784]).reshape(1, -1)
    return reshaped_data


def postprocess(input_data):
    return input_data.tolist()[0]
