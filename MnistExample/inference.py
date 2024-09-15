from __future__ import print_function

import io
import os
import json
import base64
import flask
import numpy as np
import tensorflow as tf
from PIL import Image

prefix = "/opt/ml/"
model_path = os.path.join(prefix, "model")

class ScoringService(object):
    """A singleton class for managing the model."""
    
    model = None  # Holds the loaded model
    
    @classmethod
    def get_model(cls):
        """Load the model if it's not already loaded."""
        if cls.model is None:
            print(f"Model directory contents: {os.listdir(model_path)}")
            model_file_path = os.path.join(model_path, "model.h5")
            cls.model = tf.keras.models.load_model(model_file_path)
        return cls.model

    @classmethod
    def predict(cls, input_data: np.ndarray):
        """Make predictions on the input data."""
        model = cls.get_model()
        predictions = model.predict(input_data)
        predicted_digit = np.argmax(predictions, axis=1)
        return predicted_digit[0]

app = flask.Flask(__name__)

@app.route("/ping", methods=["GET"])
def ping():
    """Health check endpoint to ensure the service is working."""
    health = ScoringService.get_model() is not None
    status = 200 if health else 404
    return flask.Response(response="\n", status=status, mimetype="application/json")

@app.route("/invocations", methods=["POST"])
def transformation():
    """Endpoint to perform inference on input data."""
    if flask.request.content_type != "application/json":
        return flask.Response(
            response="This predictor only supports JSON data",
            status=415,
            mimetype="text/plain"
        )
    
    try:
        request_data = flask.request.get_json()
        if 'image' not in request_data:
            return flask.Response(
                response="Missing 'image' key in request data",
                status=400,
                mimetype="text/plain"
            )
        
        image_data = request_data['image']
        
        if isinstance(image_data, str):
            image_data = base64.b64decode(image_data)
        
        image = Image.open(io.BytesIO(image_data)).convert('L')
        image = image.resize((28, 28))
        image_np = np.array(image).astype('float32') / 255.0
        image_np = np.reshape(image_np, (1, 28, 28, 1))
        
        print(f"Invoked with image of size: {image_np.shape}")
        
        predicted_digit = ScoringService.predict(image_np)
        
        result = {"predicted_digit": int(predicted_digit)}
        
        return flask.Response(response=json.dumps(result), status=200, mimetype="application/json")
    
    except Exception as e:
        print(f"Error occurred: {e}")
        return flask.Response(
            response="Internal Server Error",
            status=500,
            mimetype="text/plain"
        )
