# Emotion Classification with RNNs

A simple project to compare three recurrent neural network architectures—SimpleRNN, BiGRU, and BiLSTM—for emotion classification using the DAIR‑AI emotion dataset.

## Requirements

* Python 3.6+
* TensorFlow
* pandas
* numpy
* matplotlib
* seaborn
* scikit-learn

## Data

Unzip `data.zip` in the project root. It should contain:

* **Raw data** from Hugging Face:
  [https://huggingface.co/datasets/dair-ai/emotion](https://huggingface.co/datasets/dair-ai/emotion)
* **Processed CSV files** ready for immediate use

## Usage

1. Unzip the data archive:

   ```bash
   unzip data.zip
   ```
2. (Optional) Create and activate a virtual environment.
3. Install the dependencies:

   ```bash
   pip install -r requirements.txt
   ```
4. Launch Jupyter and open `code.ipynb`:

   ```bash
   jupyter notebook code.ipynb
   ```
5. Run all cells in order. The final cell trains each model and saves performance metrics to `results.csv`.

## Project Structure

* `code.ipynb`
  Notebook with sections for:

  * Data loading and tokenization
  * Model builders (`build_rnn_model`, `build_gru_model`, `build_lstm_model`)
  * Training/compilation helpers and plotting functions
  * Model comparison loop over SimpleRNN, BiGRU, and BiLSTM
* `results.csv`
  Generated file containing accuracy, loss, and other metrics for each model

## Results

After running the notebook, inspect `results.csv` for a tabular summary of each model's performance. You can also review the training/validation curves plotted in the notebook.

---

*Enjoy experimenting with different RNN architectures!*
