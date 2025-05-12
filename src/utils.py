import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
import numpy as np

def plot_history(history, title: str = "Training Curves"):
    plt.figure(figsize=(7, 4))
    plt.plot(history.history["accuracy"], label="train")
    plt.plot(history.history["val_accuracy"], label="val")
    plt.ylabel("accuracy")
    plt.xlabel("epoch")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.show()

def evaluate_and_plot(model, x_test, y_test, label_names):
    y_pred = np.argmax(model.predict(x_test), axis=1)
    print(classification_report(y_test, y_pred, target_names=label_names))

    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=label_names,
        yticklabels=label_names,
    )
    plt.ylabel("true")
    plt.xlabel("predicted")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.show()
