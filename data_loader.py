import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical

# CIFAR-10 class names
CLASS_NAMES = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']

def load_and_preprocess_data(img_size=(96, 96), num_train=10000, num_test=2000):
    """
    Load CIFAR-10 dataset, resize images, normalize and one-hot encode labels.
    """
    print(" Loading CIFAR-10 dataset...")
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    x_train, y_train = x_train[:num_train], y_train[:num_train]
    x_test,  y_test  = x_test[:num_test],   y_test[:num_test]

    print(f" Train samples: {x_train.shape[0]}")
    print(f" Test samples : {x_test.shape[0]}")

    # Resize images from 32x32 to 96x96 for MobileNet
    print(f" Resizing images to {img_size}...")
    x_train = tf.image.resize(x_train, img_size).numpy()
    x_test  = tf.image.resize(x_test,  img_size).numpy()

    # Normalize pixel values to [0, 1]
    x_train = x_train / 255.0
    x_test  = x_test  / 255.0

    # One-hot encode labels
    y_train = to_categorical(y_train, num_classes=10)
    y_test  = to_categorical(y_test,  num_classes=10)

    return x_train, y_train, x_test, y_test


def augment_data(x_train, y_train):
    """
    Apply data augmentation to training set.
    """
    print(" Applying data augmentation...")

    data_augmentation = tf.keras.Sequential([
        tf.keras.layers.RandomFlip("horizontal"),
        tf.keras.layers.RandomRotation(0.1),
        tf.keras.layers.RandomZoom(0.1),
    ])

    # Create a tf.data pipeline
    train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    train_dataset = (train_dataset
                     .shuffle(10000)
                     .batch(32)
                     .map(lambda x, y: (data_augmentation(x, training=True), y),
                          num_parallel_calls=tf.data.AUTOTUNE)
                     .prefetch(tf.data.AUTOTUNE))

    print(" Augmentation pipeline ready!")
    return train_dataset


def get_test_dataset(x_test, y_test):
    """
    Wrap test data into a tf.data Dataset.
    """
    test_dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test))
    test_dataset = test_dataset.batch(32).prefetch(tf.data.AUTOTUNE)
    return test_dataset


# Quick test
if __name__ == "__main__":
    x_train, y_train, x_test, y_test = load_and_preprocess_data()
    train_ds = augment_data(x_train, y_train)
    test_ds  = get_test_dataset(x_test, y_test)

    print(f"\n Image shape : {x_train[0].shape}")
    print(f" Label shape : {y_train[0].shape}")
    print(f" Classes     : {CLASS_NAMES}")
    