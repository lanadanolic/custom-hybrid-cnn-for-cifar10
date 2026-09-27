# Functions for compiling and training CNN models

import time
from tensorflow import keras

SEED = 42
BATCH_SIZE = 128
EPOCHS = 50
LEARNING_RATE = 0.001


# compiles the model before training
def compile_model(model):
    model.compile(
        optimizer=keras.optimizers.Adam(
            learning_rate=LEARNING_RATE
        ),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )


# creates callbacks used during model training
def create_callbacks():

    return [
        # reduces the learning rate when the validation loss stops improving for several epochs
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=4,
            min_lr=1e-6
        ),

        # stops training if the validation loss does not improve and restores the weights from the best epoch
        keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=8,
            restore_best_weights=True
        )
    ]


# trains a model using the same settings for all CNN architectures and measures the total training time
def train_model(
    model,
    model_name,
    x_train,
    y_train,
    x_val,
    y_val
):
    keras.utils.set_random_seed(SEED)

    # measuring training time
    start_time = time.perf_counter()

    history = model.fit(
        x_train,
        y_train,
        validation_data=(x_val, y_val),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=create_callbacks(),
        shuffle=True,
        verbose=1
    )
    training_time = time.perf_counter() - start_time

    print(
        f"{model_name} - training time: "
        f"{training_time:.2f} s"
    )

    return history, training_time