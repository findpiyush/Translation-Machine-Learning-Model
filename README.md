# Hindi-English Neural Machine Translation (Seq2Seq RNN)

This project implements a **Neural Machine Translation (NMT) system** for **Hindi to English translation** using a **Sequence-to-Sequence (Seq2Seq) Recurrent Neural Network (RNN)** architecture with **Long Short-Term Memory (LSTM)** units. It's a foundational example of building an NMT model from scratch using TensorFlow/Keras.

## Overview

The "Hindi-English Neural Machine Translation" project demonstrates how to build a model that can learn to translate text between two languages at a character level. By training on pairs of Hindi and English sentences, the model learns the complex mappings required for translation, showcasing the power of recurrent neural networks in natural language processing tasks. This project also includes a research paper detailing the methodology and findings.

## Features

* **Sequence-to-Sequence (Seq2Seq) Architecture:** Employs the standard encoder-decoder model for machine translation.
* **Recurrent Neural Networks (RNNs) with LSTMs:** Uses LSTM layers in both the encoder and decoder for effective handling of sequence data and capturing long-range dependencies.
* **Character-Level Translation:** Processes and generates text at the character level, allowing it to handle out-of-vocabulary words more robustly than word-level models (though potentially at the cost of longer training).
* **Data Preprocessing Pipeline:** Includes steps for tokenizing text, building character vocabularies, and one-hot encoding sequences for model input.
* **Model Training & Evaluation:** Trains the NMT model and evaluates its performance (accuracy) during the training phase.
* **Model Saving and Loading:** The trained model is saved (`s2s_model.keras`) and can be reloaded for inference.
* **Inference Model:** Separate encoder and decoder models are set up specifically for generating translations given new input.
* **Greedy Decoding:** Uses a greedy search strategy to generate the translated output character by character.

## Technologies Used

* Python
* TensorFlow / Keras (for building and training the neural network)
* NumPy (for numerical operations and array manipulation)
* Pandas (though not extensively used in the provided snippet, often helpful for data handling)
* `pathlib` and `os` (for file path management)

## How It Works

The NMT system functions based on the principles of Sequence-to-Sequence models:

1.  **Data Loading and Preparation:**
    * Hindi and English sentence pairs are loaded from `train.hi` and `train.en` text files, respectively.
    * A subset of these samples (`num_samples`) is used for training.
    * All unique characters present in both input (English) and target (Hindi) languages are identified to create character-level vocabularies.
    * Target sentences are prefixed with a "start of sequence" token (`\t`) and implicitly end with a "end of sequence" token (`\n`).

2.  **Data Vectorization (One-Hot Encoding):**
    * Input, decoder input, and decoder target sequences are converted into 3D NumPy arrays using one-hot encoding. This means each character is represented as a binary vector in a high-dimensional space.
    * Sequences are padded to their maximum lengths to ensure uniform input dimensions for the neural network.

3.  **Model Architecture (Encoder-Decoder):**
    * **Encoder:** An LSTM layer reads the input (English) sequence and compresses its information into a fixed-size context vector, represented by its internal "states" (hidden state `h` and cell state `c`).
    * **Decoder:** Another LSTM layer takes the encoder's final states as its initial states. It then learns to generate the output (Hindi) sequence character by character, conditioned on these states and the previously generated character.
    * A `Dense` layer with `softmax` activation follows the decoder LSTM to output probability distributions over the target vocabulary for each time step, predicting the next character.

4.  **Training:**
    * The complete Seq2Seq model is compiled with the `adam` optimizer and `categorical_crossentropy` loss function (suitable for one-hot encoded targets).
    * The model is trained using the prepared one-hot encoded input, decoder input, and decoder target data.

5.  **Inference (Translation):**
    * After training, separate encoder and decoder models are created for inference.
    * The `encoder_model` takes an input sequence and returns its internal states.
    * The `decoder_model` takes the encoder's states and a single "start of sequence" token, then iteratively predicts the next character while updating its own internal states, until an "end of sequence" token is predicted or a maximum length is reached. This is a greedy decoding approach.

## Data Requirements

* **`train.hi`:** A plain text file containing Hindi sentences, with one sentence per line.
* **`train.en`:** A plain text file containing English sentences, with one sentence per line, corresponding to the translations in `train.hi`.

Ensure these files are present in the same directory as the script. The script is configured to use a `num_samples` (defaulting to 4500) for training, but you can adjust this based on the size of your dataset and computational resources.

## Research Paper

This project is accompanied by a research paper that delves deeper into the methodology, experimental setup, and results of this Hindi-English Neural Machine Translation model. You can access it here:

**[Research Paper on Hindi-English Translation](https://drive.google.com/file/d/1vSt5D6ayqXuj4qssV3YS5U5437_i4HML/view?usp=drive_link)**

## Usage and Further Experimentation

After training, the script includes a loop to print example translations from the training set. You can uncomment the `while True` loop at the end of the script to enable interactive translation:

```python
# '''
# while True:
#     user_input = input(
#         "Enter a Hindi sentence to translate (or 'q' to quit): ")
#     if user_input.lower() == 'q':
#         break
#     # You would need to add preprocessing (e.g., tokenization, one-hot encoding)
#     # for the user input sentence before passing it to decode_sequence
#     translated_sentence = decode_sequence(user_input)
#     print("Translated sentence:", translated_sentence)
# '''
