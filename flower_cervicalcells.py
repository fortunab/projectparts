import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import confusion_matrix, roc_auc_score, accuracy_score
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.optimizers import Adam
import flwr as fl

def load_cervical_cell_data():
    images = np.random.rand(500, 64, 64, 3)
    labels = np.random.randint(0, 2, 500)
    return images, labels

def get_model(model_name, input_shape=(64, 64, 3)):
    if model_name == "resnet":
        base_model = ResNet50(weights="imagenet", include_top=False, input_shape=(224, 224, 3))
        x = Flatten()(base_model.output)
        x = Dense(128, activation="relu")(x)
        outputs = Dense(2, activation="softmax")(x)
        model = Model(inputs=base_model.input, outputs=outputs)
    elif model_name == "alexnet":
        model = Sequential([
            Conv2D(96, kernel_size=(11, 11), strides=(4, 4), activation="relu", input_shape=input_shape),
            MaxPooling2D(pool_size=(3, 3), strides=(2, 2)),
            Conv2D(256, kernel_size=(5, 5), activation="relu", padding="same"),
            MaxPooling2D(pool_size=(3, 3), strides=(2, 2)),
            Flatten(),
            Dense(4096, activation="relu"),
            Dense(4096, activation="relu"),
            Dense(2, activation="softmax")
        ])
    elif model_name == "zfnet":
        model = Sequential([
            Conv2D(96, kernel_size=(7, 7), strides=(2, 2), activation="relu", input_shape=input_shape),
            MaxPooling2D(pool_size=(3, 3), strides=(2, 2)),
            Conv2D(256, kernel_size=(5, 5), activation="relu", padding="same"),
            MaxPooling2D(pool_size=(3, 3), strides=(2, 2)),
            Flatten(),
            Dense(4096, activation="relu"),
            Dense(4096, activation="relu"),
            Dense(2, activation="softmax")
        ])
    elif model_name == "bionnica":
        model = Sequential([
            Conv2D(64, (3, 3), activation="relu", input_shape=input_shape),
            MaxPooling2D(pool_size=(2, 2)),
            Conv2D(128, (3, 3), activation="relu"),
            MaxPooling2D(pool_size=(2, 2)),
            Flatten(),
            Dense(128, activation="relu"),
            Dense(2, activation="softmax")
        ])
    elif model_name == "bfnet":
        model = Sequential([
            Conv2D(32, (3, 3), activation="relu", input_shape=input_shape),
            MaxPooling2D(pool_size=(2, 2)),
            Conv2D(64, (3, 3), activation="relu"),
            Flatten(),
            Dense(64, activation="relu"),
            Dense(2, activation="softmax")
        ])
    else:
        raise ValueError("Unsupported model name.")

    model.compile(optimizer=Adam(learning_rate=0.001),
                  loss="sparse_categorical_crossentropy",
                  metrics=["accuracy"])
    return model

def calculate_metrics(y_true, y_pred_prob, threshold=0.5):
    y_pred = (y_pred_prob[:, 1] > threshold).astype(int)
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    accuracy = accuracy_score(y_true, y_pred)
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    roc_auc = roc_auc_score(y_true, y_pred_prob[:, 1])
    return accuracy, sensitivity, specificity, roc_auc

class CervicalCellClient(fl.client.NumPyClient):
    def __init__(self, model, train_data, test_data):
        self.model = model
        self.train_images, self.train_labels = train_data
        self.test_images, self.test_labels = test_data

    def get_parameters(self):
        return self.model.get_weights()

    def fit(self, parameters, config):
        self.model.set_weights(parameters)
        self.model.fit(self.train_images, self.train_labels, epochs=5, batch_size=32, verbose=0)
        return self.model.get_weights(), len(self.train_images), {}

    def evaluate(self, parameters, config):
        self.model.set_weights(parameters)
        loss, accuracy = self.model.evaluate(self.test_images, self.test_labels, verbose=0)
        y_pred_prob = self.model.predict(self.test_images)
        accuracy, sensitivity, specificity, roc_auc = calculate_metrics(self.test_labels, y_pred_prob)
        return loss, len(self.test_images), {"accuracy": accuracy, "sensitivity": sensitivity, "specificity": specificity, "roc_auc": roc_auc}

def federated_kfold_cross_validation(images, labels, models, k=5):
    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=42)
    results = []
    for model_name in models:
        print(f"Training {model_name} model across {k} folds...")
        fold_results = []
        for fold, (train_idx, test_idx) in enumerate(skf.split(images, labels)):
            print(f"Fold {fold + 1}/{k} for {model_name}")
            x_train, x_test = images[train_idx], images[test_idx]
            y_train, y_test = labels[train_idx], labels[test_idx]

            model = get_model(model_name, input_shape=(64, 64, 3))

            client = CervicalCellClient(model, train_data=(x_train, y_train), test_data=(x_test, y_test))
            fl.client.start_numpy_client(server_address="localhost:8080", client=client)

            y_pred_prob = model.predict(x_test)
            accuracy, sensitivity, specificity, roc_auc = calculate_metrics(y_test, y_pred_prob)

            fold_results.append({
                "fold": fold + 1,
                "accuracy": accuracy,
                "sensitivity": sensitivity,
                "specificity": specificity,
                "roc_auc": roc_auc
            })
        model_results = pd.DataFrame(fold_results).mean()
        results.append({
            "model": model_name,
            **model_results.to_dict()
        })

    return pd.DataFrame(results)

if __name__ == "__main__":
    images, labels = load_cervical_cell_data()
    models_to_evaluate = ["resnet", "alexnet", "zfnet", "bionnica", "bfnet"]
    results = federated_kfold_cross_validation(images, labels, models=models_to_evaluate, k=5)
    print("Final cross-validation results:")
    print(results)
