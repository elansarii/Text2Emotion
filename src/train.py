from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam
from src.utils import plot_history, evaluate_and_plot

def compile_and_train(
    model,
    x_train,
    y_train,
    x_val,
    y_val,
    epochs: int = 15,
    batch_size: int = 32,
    lr: float = 0.001,
):
    model.compile(
        optimizer=Adam(learning_rate=lr),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    callback = EarlyStopping(patience=3, restore_best_weights=True)
    history = model.fit(
        x_train,
        y_train,
        validation_data=(x_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[callback],
        verbose=2,
    )
    plot_history(history)
    return model

def full_run(
    model_builder,
    data_tuple,
    epochs: int = 15,
    batch_size: int = 32,
    lr: float = 0.001,
):
    (
        x_train,
        y_train,
        x_val,
        y_val,
        x_test,
        y_test,
        _,
        num_classes,
        label_names,
    ) = data_tuple

    model = model_builder(num_classes=num_classes)
    model = compile_and_train(
        model, x_train, y_train, x_val, y_val, epochs, batch_size, lr
    )
    evaluate_and_plot(model, x_test, y_test, label_names)
    return model
