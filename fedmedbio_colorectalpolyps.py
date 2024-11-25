import numpy as np
import tensorflow as tf
from sklearn.metrics import confusion_matrix
from fedbiomed.researcher.environments.environments import FedBioMedResearcherEnv
from fedbiomed.common.constants import ResearcherRequestStatus
from fedbiomed.common.message_types import Messages
from fedbiomed.researcher.requests.model_request import ModelRequest


env = FedBioMedResearcherEnv()

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

def calculate_metrics(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    sensitivity = tp / (tp + fn)
    specificity = tn / (tn + fp)
    return accuracy, sensitivity, specificity

def federated_training(env, train_images, train_labels, test_images, test_labels):
    model = create_model()
    model_request = ModelRequest(
        model=model,
        train_data=(train_images, train_labels),
        test_data=(test_images, test_labels),
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )
    result = env.run(model_request)

    if result.status == ResearcherRequestStatus.SUCCESS:
        predictions = np.argmax(model.predict(test_images), axis=1)
        accuracy, sensitivity, specificity = calculate_metrics(test_labels, predictions)
        return {"accuracy": accuracy, "sensitivity": sensitivity, "specificity": specificity}
    else:
        print("Training failed!")
        return None

if __name__ == "__main__":
    train_images, train_labels, test_images, test_labels = load_colorectal_polyps_data()

    metrics = federated_training(env, train_images, train_labels, test_images, test_labels)

    if metrics:
        print(f"Accuracy: {metrics['accuracy']:.2f}")
        print(f"Sensitivity: {metrics['sensitivity']:.2f}")
        print(f"Specificity: {metrics['specificity']:.2f}")
