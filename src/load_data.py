import pandas as pd
from sklearn.model_selection import train_test_split
import tensorflow as tf
import pathlib

LABEL_MAP = {
    0: "sadness",
    1: "joy",
    2: "love",
    3: "anger",
    4: "fear",
    5: "surprise",
}

def load_csv(name: str, data_dir: pathlib.Path) -> pd.DataFrame:
    path = data_dir / f"emotion_{name}.csv"
    return pd.read_csv(path)

def tokenize_texts(
    texts,
    max_vocab: int = 10000,
    max_len: int = 50,
    tokenizer: tf.keras.preprocessing.text.Tokenizer | None = None,
):
    if tokenizer is None:
        tokenizer = tf.keras.preprocessing.text.Tokenizer(
            num_words=max_vocab, oov_token="<OOV>"
        )
        tokenizer.fit_on_texts(texts)
    seqs = tokenizer.texts_to_sequences(texts)
    padded = tf.keras.preprocessing.sequence.pad_sequences(
        seqs, maxlen=max_len, padding="post", truncating="post"
    )
    return padded, tokenizer

def load_and_prepare_data(
    processed_dir: str = "data/processed",
    max_vocab: int = 10000,
    max_len: int = 50,
):
    processed_path = pathlib.Path(processed_dir)

    train_df = load_csv("train", processed_path)
    val_df = load_csv("val", processed_path)
    test_df = load_csv("test", processed_path)

    x_train, tokenizer = tokenize_texts(
        train_df["text"].values, max_vocab=max_vocab, max_len=max_len
    )
    x_val, _ = tokenize_texts(
        val_df["text"].values,
        max_vocab=max_vocab,
        max_len=max_len,
        tokenizer=tokenizer,
    )
    x_test, _ = tokenize_texts(
        test_df["text"].values,
        max_vocab=max_vocab,
        max_len=max_len,
        tokenizer=tokenizer,
    )

    y_train = train_df["label"].values
    y_val = val_df["label"].values
    y_test = test_df["label"].values

    num_classes = len(LABEL_MAP)
    return (
        x_train,
        y_train,
        x_val,
        y_val,
        x_test,
        y_test,
        tokenizer,
        num_classes,
        list(LABEL_MAP.values()),
    )
