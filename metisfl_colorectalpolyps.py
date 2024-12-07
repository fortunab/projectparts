import numpy as np
import tensorflow as tf
from sklearn.metrics import confusion_matrix
from metisfl.common.dtypes import DatasetSplit, TrainingStrategy, EvaluationResults
from metisfl.client.client import Client
from metisfl.server.server import Server
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras.applications import ResNet50, AlexNet


def get_model(model_name):
    if model_name == "resnet":
        base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
        x = Flatten()(base_model.output)
        x = Dense(512, activation='relu')(x)
        outputs = Dense(2, activation='softmax')(x)
        model = Model(inputs=base_model.input, outputs=outputs)
    elif model_name == "alexnet":
        model = tf.keras.Sequential([
            Conv2D(96, kernel_size=(11, 11), strides=(4, 4), activation='relu', input_shape=(224, 224, 3)),
            MaxPooling2D(pool_size=(3, 3), strides=(2, 2)),
            Conv2D(256, kernel_size=(5, 5), activation='relu', padding='same'),
            MaxPooling2D(pool_size=(3, 3), strides=(2, 2)),
            Flatten(),
            Dense(4096, activation='relu'),
            Dense(4096, activation='relu'),
            Dense(2, activation='softmax')
        ])
    elif model_name == "zfnet":
        model = tf.keras.Sequential([
            Conv2D(96, kernel_size=(7, 7), strides=(2, 2), activation='relu', input_shape=(224, 224, 3)),
            MaxPooling2D(pool_size=(3, 3), strides=(2, 2)),
            Conv2D(256, kernel_size=(5, 5), activation='relu', padding='same'),
            MaxPooling2D(pool_size=(3, 3), strides=(2, 2)),
            Flatten(),
            Dense(4096, activation='relu'),
            Dense(4096, activation='relu'),
            Dense(2, activation='softmax')
        ])
    elif model_name == "bionnica":
        model = tf.keras.Sequential([
            Conv2D(64, kernel_size=(3, 3), activation='relu', input_shape=(224, 224, 3)),
            MaxPooling2D(pool_size=(2, 2)),
            Conv2D(128, kernel_size=(3, 3), activation='relu'),
            MaxPooling2D(pool_size=(2, 2)),
            Flatten(),
            Dense(512, activation='relu'),
            Dense(2, activation='softmax')
        ])
    else:
        raise ValueError("Unsupported model name")

    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

def calculate_metrics(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    sensitivity = tp / (tp + fn)
    specificity = tn / (tn + fp)
    return accuracy, sensitivity, specificity

def federated_training(models_list, train_data, test_data):
    results = {}
    for model_name in models_list:
        print(f"Training model: {model_name}")
        model = get_model(model_name)
        server = Server(training_strategy=TrainingStrategy.SYNCHRONOUS)
        clients = [Client(model=model, dataset_split=DatasetSplit(train=train_data))]
        server.initialize(clients=clients)
        server.train(rounds=5)
        test_images, test_labels = test_data
        predictions = np.argmax(model.predict(test_images), axis=1)
        accuracy, sensitivity, specificity = calculate_metrics(test_labels, predictions)

        results[model_name] = {
            "accuracy": accuracy,
            "sensitivity": sensitivity,
            "specificity": specificity
        }

    return results

if __name__ == "__main__":
    train_images, train_labels, test_images, test_labels = load_colorectal_polyps_data()
    train_data = (train_images, train_labels)
    test_data = (test_images, test_labels)
    models_to_train = ["resnet", "alexnet", "zfnet", "bionnica"]
    metrics = federated_training(models_to_train, train_data, test_data)
    for model_name, model_metrics in metrics.items():
        print(f"Metrics for {model_name}:")
        print(f"  Accuracy: {model_metrics['accuracy']:.2f}")
        print(f"  Sensitivity: {model_metrics['sensitivity']:.2f}")
        print(f"  Specificity: {model_metrics['specificity']:.2f}")
