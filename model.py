import tensorflow as tf
import pandas as pd
from sklearn.model_selection import train_test_split as tts
from sklearn.metrics import mean_squared_error, mean_absolute_error
import numpy as np
import json


class Model:
    def __init__(self, input_shape=(None,)):
        self.model = self.init_model(input_shape)
        self.test_data = None
        self.train_results = None

        self.pre_trained_models = ["boston", "diabetes"]

    def load_pre_trained(self, pre_trained_name):
        assert pre_trained_name in self.pre_trained_models

        model_path = f"trained_{pre_trained_name}_weights.h5"
        self.model.load_weights(model_path)

        model_history_path = f"trained_{pre_trained_name}_history.json"

        with open(model_history_path, "r") as f:
            self.train_results = json.load(f)

    def init_model(self, input_size=(None,)):
        input_layer = tf.keras.layers.Input(input_size)
        x = tf.keras.layers.Dense(20)(input_layer)
        x = tf.keras.layers.Dropout(0.4)(x)
        output_layer = tf.keras.layers.Dense(1)(x)
        model = tf.keras.Model(inputs=input_layer, outputs=output_layer)
        model.compile(optimizer="adam", loss='mse')

        return model

    def train(self, training_data, epochs):
        train, val = self.split_data(training_data)
        history = self.model.fit(x=train[0], y=train[1], validation_data=val, epochs=epochs).history

        self.train_results = history
        return self.evaluate()

    def inference(self, inference_data):
        data = np.array(inference_data)
        print(type(data))
        prediction = self.model.predict(data)
        return prediction

    def evaluate(self):
        assert self.test_data is not None
        prediction = self.inference(self.test_data[0])

        return {
            "mse": mean_squared_error(self.test_data[1], prediction),
            "mae": mean_absolute_error(self.test_data[1], prediction)
        }

    def split_data(self, training_data):
        features = pd.DataFrame(training_data["features"])
        target = pd.DataFrame(training_data["target"])
        train, val, train_target, val_target = tts(features, target, train_size=0.6)

        val, test, val_target, test_target = tts(val, val_target, train_size=0.5)
        self.test_data = (test, test_target)

        return (train, train_target), (val, val_target)


if __name__ == "__main__":
    def load_boston():
        train, test = tf.keras.datasets.boston_housing.load_data(test_split=0)

        return {
            "features": train[0],
            "target": train[1]
        }

    def load_diabetes():
        from sklearn.datasets import load_diabetes
        features, target = load_diabetes(return_X_y=True)
        return {
            "features": np.array(features),
            "target": np.array(target)
        }


    for name in ["diabetes"]:
        data = {"boston": load_boston, "diabetes": load_diabetes}[name]()
        shape = (data["features"].shape[1],)
        model = Model(input_shape=shape)
        results = model.train(data, 30)

        results["history"] = model.train_results

        model_path = f"trained_{name}_weights.h5"
        model_history_path = f"trained_{name}_history.json"
        with open(model_history_path, "w") as f:
            json.dump(results, f, ensure_ascii=True)

        model.model.save_weights(model_path)