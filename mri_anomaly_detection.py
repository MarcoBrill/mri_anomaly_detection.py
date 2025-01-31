import os
import pydicom
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

# Constants
IMG_SIZE = (128, 128)  # Resize images to 128x128
BATCH_SIZE = 32
EPOCHS = 50
LEARNING_RATE = 1e-4
NUM_CLASSES = 2  # Binary classification: Normal (0) vs Anomaly (1)

# Load and preprocess DICOM files
def load_dicom_images(directory, img_size=IMG_SIZE):
    images = []
    labels = []
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith(".dcm"):
                dicom_path = os.path.join(root, file)
                dicom = pydicom.dcmread(dicom_path)
                image = dicom.pixel_array
                image = tf.image.resize(image, img_size)
                image = tf.keras.applications.resnet50.preprocess_input(image)
                images.append(image)
                # Assuming directory structure: /path/to/data/normal/ and /path/to/data/anomaly/
                label = 0 if "normal" in root.lower() else 1
                labels.append(label)
    return np.array(images), np.array(labels)

# Build CNN model
def build_cnn_model(input_shape, num_classes):
    inputs = tf.keras.Input(shape=input_shape)
    x = layers.Conv2D(32, (3, 3), activation="relu")(inputs)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Conv2D(64, (3, 3), activation="relu")(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Conv2D(128, (3, 3), activation="relu")(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Flatten()(x)
    x = layers.Dense(128, activation="relu")(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)
    model = models.Model(inputs, outputs)
    return model

# Main function
def main():
    # Load data
    data_dir = "path/to/your/dicom/data"  # Update this path
    images, labels = load_dicom_images(data_dir)
    labels = tf.keras.utils.to_categorical(labels, NUM_CLASSES)

    # Split data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.2, random_state=42)

    # Data augmentation
    datagen = ImageDataGenerator(
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode="nearest"
    )

    # Build and compile model
    model = build_cnn_model((IMG_SIZE[0], IMG_SIZE[1], 1), NUM_CLASSES)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
                  loss="categorical_crossentropy",
                  metrics=["accuracy"])

    # Callbacks
    callbacks = [
        EarlyStopping(patience=5, restore_best_weights=True),
        ModelCheckpoint("best_model.h5", save_best_only=True)
    ]

    # Train model
    history = model.fit(datagen.flow(X_train, y_train, batch_size=BATCH_SIZE),
                        validation_data=(X_test, y_test),
                        epochs=EPOCHS,
                        callbacks=callbacks)

    # Evaluate model
    test_loss, test_acc = model.evaluate(X_test, y_test)
    print(f"Test Accuracy: {test_acc:.4f}")

    # Save model
    model.save("mri_anomaly_detection_model.h5")

if __name__ == "__main__":
    main()
