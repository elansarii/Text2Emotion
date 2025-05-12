import pandas as pd

# Load train/val/test
train = pd.read_parquet("data/raw/train-00000-of-00001.parquet")
val = pd.read_parquet("data/raw/validation-00000-of-00001.parquet")
test = pd.read_parquet("data/raw/test-00000-of-00001.parquet")

# Show sample
print("Train sample:\n", train.head())

# Save as CSV
train.to_csv("data/processed/emotion_train.csv", index=False)
val.to_csv("data/processed/emotion_val.csv", index=False)
test.to_csv("data/processed/emotion_test.csv", index=False)
