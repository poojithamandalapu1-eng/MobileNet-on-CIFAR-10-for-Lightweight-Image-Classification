import os
import tensorflow as tf
from data_loader import load_and_preprocess_data, augment_data, get_test_dataset
from models.model import build_mobilenet_model, unfreeze_model, get_model_size

Img_SiZE      = (96,96)
BATCH_SIZE    = 32
EPOCHS_PHASE1 = 10
EPOCHS_PHASE2 = 10
MODEL_SAVE_PATH = "mobilenet_cifar10.h5"

def train():

    x_train, y_train, x_test, y_test = load_and_preprocess_data(IMG_SIZE)
    train_ds = augment_data(x_train, y_train)
    test_ds = get_test_dataset(x_test, y_test)

    model, base_model = build_mobilenet_model(
        input_shape = (IMG_SIZE[0], IMG_SIZE[1],3)
    )
    get_model_size(model)

    print("\n Phase 1: Training top layers...")
    model.compile(
          optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
           loss='categorical_crossentropy',
           metrics=['accuracy']
    )
        callbacks_phase1 = [
        tf.keras.callbacks.EarlyStopping(
            monitor='val_accuracy', patience=3, restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss', factor=0.5, patience=2, verbose=1),
    ]
    history1 = model.fit(
        train_ds,
        epochs=EPOCHS_PHASE1,
        validation_data=test_ds,
        callbacks=callbacks_phase1,
        verbose=1
    )
    print("\n Phase 2: Fine-tuning...")
    model = unfreeze_model(model, base_model, unfreeze_layers=30)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    callbacks_phase2 = [
        tf.keras.callbacks.EarlyStopping(
            monitor='val_accuracy', patience=5, restore_best_weights=True),
        tf.keras.callbacks.ModelCheckpoint(
            MODEL_SAVE_PATH, save_best_only=True,
            monitor='val_accuracy', verbose=1),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss', factor=0.5, patience=2, verbose=1),
    ]

    history2 = model.fit(
        train_ds,
        epochs=EPOCHS_PHASE2,
        validation_data=test_ds,
        callbacks=callbacks_phase2,
        verbose=1
    )
    print("\n Final Evaluation on Test Set:")
    loss, accuracy = model.evaluate(test_ds, verbose=0)
    print(f"Test Accuracy : {accuracy * 100:.2f}%")
    print(f" Test Loss     : {loss:.4f}")
    print(f"\n  Model saved to: {MODEL_SAVE_PATH}")

    return history1, history2


if __name__ == "__main__":
    train()
