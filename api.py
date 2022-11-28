import tensorflow as tf
import pandas as pd
import numpy as np
import json

#Very simply rest api flask app

from model import Model
from flask import Flask, jsonify, request, abort

app = Flask(__name__)
sessions = {}

def validate(json_payload):
    keys = json_payload.keys()
    try:
        assert "key" in keys, "Requires Session key"

        if json_payload["key"] not in sessions.keys():
            sessions[json_payload["key"]] = {}

        if "inference_type" in keys:
            assert "data" in keys, "Missing Data"
            assert json_payload["inference_type"] in ["session", "pre_trained"], "Inference must be session or pre_trained"
            if json_payload["inference_type"] == "pre_trained":
                assert "pretrain_name" in json_payload.keys(), "Specify a pre-trained name"
                possible_models = Model((1,)).pre_trained_models
                assert json_payload["pretrain_name"] in possible_models, f"Model must be in {possible_models}"
            assert type(json_payload["data"]) == list, "Inference data must be in a list"

        elif "epochs" in keys:
            assert "data" in keys, "Must include training data"
            assert set(json_payload["data"].keys()) == {"features", "target"}, "Must include features and target"

        error = False
        message = "OK"

    except AssertionError as e:
        message = repr(e)
        error = True

    return error, message


@app.route("/inference", methods=["POST"])
def inference():
    payload = request.json
    error, message = validate(payload)

    if error:
        return jsonify({"message": message}), 400
    else:
        session = sessions[payload["key"]]
        inference_type = payload["inference_type"]

        if inference_type == "session":
            try:
                assert "model" in session.keys()
                model = session["model"]

                result = model.inference([payload["data"]])
                return jsonify({"result": result, "message":"OK"}), 200

            except AssertionError as e:
                return jsonify({"message": "No trained model in session"}), 400

        elif inference_type == "pre_trained":
            payload_data = payload["data"]
            model = Model(input_shape=(len(payload_data),))
            model.load_pre_trained(payload["pretrain_name"])
            session["model"] = model

            result = model.inference([payload_data])
            result = dict(enumerate(result.astype(float).flatten(), 0))
            return jsonify({"result": result, "message": "OK"}), 200


@app.route("/train", methods=["POST"])
def train_new():
    payload = request.json
    error, message = validate(payload)

    if error:
        return jsonify({"message": message}), 400
    else:
        session = sessions[payload["key"]]
        model_data = payload["data"]
        shape = (np.array(model_data["features"]).shape[1],)
        model = Model(input_shape=shape)

        try:
            epochs = payload["epochs"]
            model.train(model_data, epochs=epochs)

            session["model"] = model
            model_quality = model.evaluate()

            return jsonify({"quality": model_quality})

        except AssertionError as e:
            return jsonify({"message": "Required data not passed"}), 400


@app.route("/save_model", methods=["GET"])
def save_model():
    # Gives the user the session loaded model
    payload = request.json
    error, message = validate(payload)
    if error:
        return jsonify({"message": message}), 400

    else:
        session = sessions[payload["key"]]

        try:
            assert "model" in session.keys()
            model = session["model"]
            model_weights = pd.DataFrame(model.model.get_weights()).to_json()
            model_arch = model.model.to_json()
            return jsonify({"model_weights": model_weights, "model":model_arch}), 200

        except AssertionError as e:
            return jsonify({"message": "No model in session"}), 400


@app.route('/clear', methods=["GET"])
def clear():
    payload = request.json
    error, message = validate(payload)
    if error:
        return jsonify({"message": message}), 400

    else:
        session = sessions[payload["key"]]
        session.pop("model", None)
        return jsonify({"message": "OK"}), 200


@app.route("/loss_data", methods=["GET"])
def loss_data():
    payload = request.json
    error, message = validate(payload)
    if error:
        return jsonify({"message": message}), 400

    else:
        session = sessions[payload["key"]]
        try:
            model = session["model"]
            loss = model.train_results
            assert loss is not None, "No model trained"
            return jsonify({"loss": loss}, 200)

        except AssertionError as e:
            return jsonify({"message": "No model trained"}), 400


if __name__ == "__main__":
    app.run(debug=True)
