import numpy as np
import tensorflow as tf
from sklearn.metrics import confusion_matrix
from flwr.server import start_server
from flwr.client import start_client, NumPyClient
from flwr.common import ndarrays_to_parameters
from typing import Tuple

def load_colorectal_polyps_data():
    train_images = np.random.rand(100, 224, 224, 3)
    train_labels = np.random.randint(0, 2, 100)
    test_images = np.random.rand(30, 224, 224, 3)
    test_labels = np.random.randint(0, 2, 30)
    return train_images, train_labels, test_images, test_labels

def create_model():
    base_model = tf.keras.applications.ResNet50(weights="imagenet", include_top=False, input_shape=(224, 224, 3))
    x = tf.keras.layers.Flatten()(base_model.output)
    x = tf.keras.layers.Dense(512, activation="relu")(x)
    outputs = tf.keras.layers.Dense(2, activation="softmax")(x)
    model = tf.keras.Model(inputs=base_model.input, outputs=outputs)
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return model

class ColorectalClient(NumPyClient):
    def __init__(self, model, train_data, test_data):
        self.model = model
        self.train_images, self.train_labels = train_data
        self.test_images, self.test_labels = test_data

    def get_parameters(self):
        return ndarrays_to_parameters(self.model.get_weights())

    def fit(self, parameters, config):
        self.model.set_weights(parameters)
        self.model.fit(self.train_images, self.train_labels, epochs=1, batch_size=32, verbose=0)
        return self.get_parameters(), len(self.train_images), {}

    def evaluate(self, parameters, config):
        self.model.set_weights(parameters)
        loss, accuracy = self.model.evaluate(self.test_images, self.test_labels, verbose=0)
        predictions = np.argmax(self.model.predict(self.test_images), axis=1)
        accuracy, sensitivity, specificity = calculate_metrics(self.test_labels, predictions)
        return loss, len(self.test_images), {"accuracy": accuracy, "sensitivity": sensitivity, "specificity": specificity}

def calculate_metrics(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    sensitivity = tp / (tp + fn)
    specificity = tn / (tn + fp)
    return accuracy, sensitivity, specificity

def start_flower_server():
    start_server(config={"num_rounds": 5})

def start_flower_client(train_data, test_data):
    model = create_model()
    client = ColorectalClient(model, train_data, test_data)
    start_client(client)

if __name__ == "__main__":
    train_images, train_labels, test_images, test_labels = load_colorectal_polyps_data()
    train_data = (train_images, train_labels)
    test_data = (test_images, test_labels)

    # Start the Flower server and clients
    import multiprocessing
    server_process = multiprocessing.Process(target=start_flower_server)
    client_process = multiprocessing.Process(target=start_flower_client, args=(train_data, test_data))

    server_process.start()
    client_process.start()

    server_process.join()
    client_process.join()
    