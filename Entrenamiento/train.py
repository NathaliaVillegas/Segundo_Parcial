import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import cv2
import os
from preprocess import preprocess

IMG_SIZE = (224, 224)
classes = ["coca", "fanta", "pepsi", "salvietti"]

def load_data(folder):
    X, y = [], []

    for i, cls in enumerate(classes):
        path = os.path.join(folder, cls)

        for img_name in os.listdir(path):
            img_path = os.path.join(path, img_name)

            img = cv2.imread(img_path)
            if img is None:
                continue

            img = preprocess(img)

            X.append(img)
            y.append(i)

    return np.array(X), np.array(y)

print("Cargando dataset...")

X_train, y_train = load_data("dataset_split/train")
X_test, y_test = load_data("dataset_split/test")

print(X_train.shape, X_test.shape)

# 🔥 MODELO MEJORADO (con generalización)

model = keras.Sequential([
    layers.Input(shape=(224, 224, 3)),

    # augmentation (CLAVE)
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.1),
    layers.RandomZoom(0.1),

    layers.Conv2D(32, 3, activation="relu"),
    layers.MaxPooling2D(),

    layers.Conv2D(64, 3, activation="relu"),
    layers.MaxPooling2D(),

    layers.Conv2D(128, 3, activation="relu"),
    layers.MaxPooling2D(),

    layers.Dropout(0.3),

    layers.Flatten(),
    layers.Dense(128, activation="relu"),
    layers.Dense(len(classes), activation="softmax")
])

model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

model.fit(
    X_train, y_train,
    validation_data=(X_test, y_test),
    epochs=15
)

model.save("bebidas_model.keras")

print("MODELO LISTO")