import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix
from substrafl.nodes import TrainDataNode, TestDataNode, TrainNode, AggregationNode, OutputNode
from substrafl.schemas import Dataset, Objective
from substrafl.strategies import FedAvg
from substrafl.algorithms import TorchFLAlgorithm
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset

class ColorectalDataset(Dataset):
    def __init__(self, images, labels, transform=None):
        self.images = images
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)

        return image, label

def get_model(model_name):
    if model_name == "resnet":
        model = models.resnet18(pretrained=True)
        model.fc = nn.Linear(model.fc.in_features, 2)
    elif model_name == "alexnet":
        model = models.alexnet(pretrained=True)
        model.classifier[6] = nn.Linear(model.classifier[6].in_features, 2)
    elif model_name == "zfnet":
        model = models.resnet50(pretrained=True)
        model.fc = nn.Linear(model.fc.in_features, 2)
    elif model_name == "bionnica":
        class BionnicaNet(nn.Module):
            def __init__(self):
                super(BionnicaNet, self).__init__()
                self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1)
                self.conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
                self.fc1 = nn.Linear(128 * 56 * 56, 512)
                self.fc2 = nn.Linear(512, 2)

            def forward(self, x):
                x = torch.relu(self.conv1(x))
                x = torch.relu(self.conv2(x))
                x = x.view(x.size(0), -1)
                x = torch.relu(self.fc1(x))
                x = self.fc2(x)
                return x
        model = BionnicaNet()
    else:
        raise ValueError("Model not supported")

    return model

def calculate_metrics(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    sensitivity = tp / (tp + fn)
    specificity = tn / (tn + fp)
    return accuracy, sensitivity, specificity

def federated_training_pipeline(models_list, train_data, test_data, transforms=None):
    results = {}
    for model_name in models_list:
        print(f"Training model: {model_name}")
        model = get_model(model_name)
        fl_algorithm = TorchFLAlgorithm(
            model=model,
            optimizer=torch.optim.Adam(model.parameters(), lr=0.001),
            criterion=nn.CrossEntropyLoss(),
        )
        train_node = TrainDataNode(data=train_data, transforms=transforms)
        test_node = TestDataNode(data=test_data, transforms=transforms)
        agg_node = AggregationNode(strategy=FedAvg())
        fl_algorithm.fit(
            train_node=train_node,
            test_node=test_node,
            agg_node=agg_node,
            epochs=5
        )
        y_true = []
        y_pred = []
        for images, labels in test_node:
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            y_true.extend(labels.numpy())
            y_pred.extend(preds.numpy())
        accuracy, sensitivity, specificity = calculate_metrics(y_true, y_pred)
        results[model_name] = {
            "accuracy": accuracy,
            "sensitivity": sensitivity,
            "specificity": specificity
        }

    return results

if __name__ == "__main__":
    images_train = np.random.rand(100, 3, 224, 224)
    labels_train = np.random.randint(0, 2, 100)
    images_test = np.random.rand(30, 3, 224, 224)
    labels_test = np.random.randint(0, 2, 30)

    train_dataset = ColorectalDataset(images_train, labels_train, transform=transforms.ToTensor())
    test_dataset = ColorectalDataset(images_test, labels_test, transform=transforms.ToTensor())
    models_to_train = ["resnet", "alexnet", "zfnet", "bionnica"]
    metrics = federated_training_pipeline(models_to_train, train_dataset, test_dataset)

    for model_name, model_metrics in metrics.items():
        print(f"Metrics for {model_name}:")
        print(f"  Accuracy: {model_metrics['accuracy']:.2f}")
        print(f"  Sensitivity: {model_metrics['sensitivity']:.2f}")
        print(f"  Specificity: {model_metrics['specificity']:.2f}")
