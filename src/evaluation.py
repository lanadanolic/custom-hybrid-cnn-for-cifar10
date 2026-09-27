# Functions for evaluating CNN models and visualizing their performance

import time
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    ConfusionMatrixDisplay
)

BATCH_SIZE = 128

def plot_history(history, model_name):

    plt.figure(figsize=(6, 4))

    plt.plot(
        history.history["accuracy"],
        label="Train accuracy"
    )

    plt.plot(
        history.history["val_accuracy"],
        label="Validation accuracy"
    )

    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title(
        f"{model_name} - accuracy during training"
    )

    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(6, 4))

    plt.plot(
        history.history["loss"],
        label="Train loss"
    )

    plt.plot(
        history.history["val_loss"],
        label="Validation loss"
    )

    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(
        f"{model_name} - loss during training"
    )

    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.show()


# measures the average time required by the model to classify one image from the test set
def measure_avg_prediction_time_per_image(
    model,
    x_test
):
    start_time = time.perf_counter()

    model.predict(
        x_test,
        batch_size=BATCH_SIZE,
        verbose=0
    )

    end_time = time.perf_counter()

    prediction_time = end_time - start_time

    avg_prediction_time_per_image = (
        prediction_time / len(x_test)
    ) * 1000.0

    return avg_prediction_time_per_image


# evaluates a trained model on the test set and calculates all metrics used for model comparison
def evaluate_model(
    model,
    model_name,
    training_time,
    x_test,
    y_test,
    class_names
):
    # calculates test loss and accuracy
    test_loss, test_accuracy = model.evaluate(
        x_test,
        y_test,
        batch_size=BATCH_SIZE,
        verbose=0
    )

    # generates class probabilities for the test images
    y_prob = model.predict(
        x_test,
        batch_size=BATCH_SIZE,
        verbose=0
    )

    # converts probabilities into predicted class labels
    y_pred = np.argmax(
        y_prob,
        axis=1
    )

    # calculates macro-averaged classification metrics
    precision = precision_score(
        y_test,
        y_pred,
        average="macro",
        zero_division=0
    )

    recall = recall_score(
        y_test,
        y_pred,
        average="macro",
        zero_division=0
    )

    f1 = f1_score(
        y_test,
        y_pred,
        average="macro",
        zero_division=0
    )

    # counts the total number of model parameters
    parameters = model.count_params()

    # measures the average prediction time per image
    prediction_time_per_image = (
        measure_avg_prediction_time_per_image(
            model,
            x_test
        )
    )

    print(model_name)

    print(
        f"Test loss: {test_loss:.4f}"
    )

    print(
        f"Accuracy: {test_accuracy:.4f} "
        f"({test_accuracy * 100:.2f}%)"
    )

    print(
        f"Precision: {precision:.4f}"
    )

    print(
        f"Recall: {recall:.4f}"
    )

    print(
        f"F1-score: {f1:.4f}"
    )

    print(
        f"Number of parameters: {parameters}"
    )

    print(
        f"Training time: {training_time:.4f} s"
    )

    print(
        "Prediction time per image: "
        f"{prediction_time_per_image:.4f} ms"
    )

    print("\nClassification report:")

    print(
        classification_report(
            y_test,
            y_pred,
            target_names=class_names,
            digits=4,
            zero_division=0
        )
    )

    fig, ax = plt.subplots(
        figsize=(6, 6)
    )

    ConfusionMatrixDisplay.from_predictions(
        y_test,
        y_pred,
        display_labels=class_names,
        xticks_rotation=45,
        ax=ax
    )

    ax.set_title(
        f"{model_name} - Confusion Matrix"
    )

    plt.tight_layout()
    plt.show()

    results = {
        "Model": model_name,
        "Test loss": test_loss,
        "Accuracy": test_accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1-score": f1,
        "Parameters": parameters,
        "Training time (s)": training_time,
        "Prediction time per image (ms)":
            prediction_time_per_image
    }

    return results