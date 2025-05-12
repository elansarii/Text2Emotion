import os, random, json
import numpy as np
import tensorflow as tf
import pandas as pd

from src.load_data import load_and_prepare_data
from src.build_models import (
    build_rnn_model,
    build_gru_model,
    build_lstm_model,
)
from src.train import compile_and_train, evaluate_and_plot

# ────────────────────────────────────────────────────────────────────────
#  Reproducibility
# ────────────────────────────────────────────────────────────────────────
SEED = 42
os.environ["PYTHONHASHSEED"] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

# ────────────────────────────────────────────────────────────────────────
#  Data
# ────────────────────────────────────────────────────────────────────────
(
    x_train,
    y_train,
    x_val,
    y_val,
    x_test,
    y_test,
    tokenizer,
    num_classes,
    label_names,
) = load_and_prepare_data()

# ────────────────────────────────────────────────────────────────────────
#  Model zoo
# ────────────────────────────────────────────────────────────────────────
MODEL_BUILDERS = {
    "SimpleRNN": build_rnn_model,
    "BiGRU": build_gru_model,
    "BiLSTM": build_lstm_model,
}

results = []

for name, builder in MODEL_BUILDERS.items():
    print(f"\n🟢 Training {name} …")
    model = builder(vocab_size=10_000, max_len=50, num_classes=num_classes)

    model = compile_and_train(
        model,
        x_train,
        y_train,
        x_val,
        y_val,
        epochs=15,
        batch_size=32,
        lr=1e-3,
    )

    # — evaluation —
    y_pred = np.argmax(model.predict(x_test), axis=1)
    acc = (y_pred == y_test).mean()

    from sklearn.metrics import precision_recall_fscore_support

    p, r, f1, _ = precision_recall_fscore_support(
        y_test, y_pred, average="macro", zero_division=0
    )
    results.append(
        {
            "model": name,
            "accuracy": acc,
            "precision": p,
            "recall": r,
            "f1": f1,
        }
    )

    # save weights & tokenizer per model
    save_dir = f"models/{name}"
    os.makedirs(save_dir, exist_ok=True)
    model.save(f"{save_dir}/weights.h5")
    with open(f"{save_dir}/tokenizer.json", "w") as f:
        json.dump(tokenizer.to_json(), f)

    # confusion matrix plot (saved, not just shown)
    evaluate_and_plot(model, x_test, y_test, label_names)
    tf.keras.backend.clear_session()  # free RAM before next loop

# ────────────────────────────────────────────────────────────────────────
#  Summary table
# ────────────────────────────────────────────────────────────────────────
df = pd.DataFrame(results).round(4)
print("\n🏁  FINAL COMPARISON\n", df)
df.to_csv("results.csv", index=False)
