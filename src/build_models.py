from tensorflow.keras import Sequential
from tensorflow.keras.layers import (
    Embedding,
    SimpleRNN,
    LSTM,
    GRU,
    Bidirectional,
    Dense,
    Dropout,
)

def build_rnn_model(vocab_size: int, max_len: int, num_classes: int) -> Sequential:
    model = Sequential(
        [
            Embedding(vocab_size, 64, input_length=max_len),
            SimpleRNN(64),
            Dropout(0.3),
            Dense(64, activation="relu"),
            Dense(num_classes, activation="softmax"),
        ]
    )
    return model

def build_lstm_model(vocab_size: int, max_len: int, num_classes: int) -> Sequential:
    model = Sequential(
        [
            Embedding(vocab_size, 64, input_length=max_len),
            Bidirectional(LSTM(64)),
            Dropout(0.3),
            Dense(64, activation="relu"),
            Dense(num_classes, activation="softmax"),
        ]
    )
    return model

def build_gru_model(vocab_size: int, max_len: int, num_classes: int) -> Sequential:
    model = Sequential(
        [
            Embedding(vocab_size, 64, input_length=max_len),
            Bidirectional(GRU(64)),
            Dropout(0.3),
            Dense(64, activation="relu"),
            Dense(num_classes, activation="softmax"),
        ]
    )
    return model
