import pandas as pd
import os
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from sklearn.model_selection import train_test_split


def load_cervical_cell_data(csv_file, img_dir, target_size=(64, 64)):
    data = pd.read_csv(csv_file)
    images, labels = [], []
    for _, row in data.iterrows():
        img_path = os.path.join(img_dir, row["image_filename"])
        img = load_img(img_path, target_size=target_size)
        images.append(img_to_array(img) / 255.0)
        labels.append(row["label"])
    images = np.array(images)
    labels = np.array(labels)
    x_train, x_test, y_train, y_test = train_test_split(images, labels, test_size=0.2, random_state=42)
    return (x_train, y_train), (x_test, y_test)

(x_train, y_train), (x_test, y_test) = load_cervical_cell_data(
    "cervical.csv", "samples_data"
)

import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

def load_colorectal_polyps_data(image_dir, target_size=(224, 224)):
    datagen = ImageDataGenerator(rescale=1.0/255.0, validation_split=0.2)  # Normalize pixel values

    train_data = datagen.flow_from_directory(
        image_dir,
        target_size=target_size,
        batch_size=32,
        class_mode="binary",
        subset="training"
    )

    val_data = datagen.flow_from_directory(
        image_dir,
        target_size=target_size,
        batch_size=32,
        class_mode="binary",
        subset="validation"
    )

    return train_data, val_data

train_data, val_data = load_colorectal_polyps_data("sample_data/colorectal.csv")

