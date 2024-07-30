import tensorflow as tf
import numpy as np
import pandas as pd
from pathlib import Path

batch_size=10
epochs=5
latent_dim=300
num_samples=2000
data_path= "Dataset_English_Hindi.txt"

input_texts=[]
target_texts=[]
input_characters=set()
target_characters=set()

nltk.download("stopwords")
from nltk.corpus import stopwords
english_stopwords= stopwords.words("english")
with open("final_stopwords.txt", 'r', encoding='utf-8') as file:
    hindi_stopwords=file.read().split("\n")

with open(data_path, 'r', encoding='utf-8') as file:
    lines=file.read().split("\n")
for line in lines[: min(num_samples, len(lines)-1)]:
    input_text, target_text= line.split(",", 1)
    input_text=input_text.strip()
    target_text="\t"+target_text+"\n"
    updated_text_eng = ""
    updated_text_hin = ""
    for text in input_text.split(' '):
        if text not in english_stopwords:
            if updated_text_eng!="":
                updated_text_eng = updated_text_eng + " " + text
            else:
                updated_text_eng = text
    input_texts.append(updated_text_eng)
    for text in target_text.split(' '):
        if text not in hindi_stopwords:
            updated_text_hin=updated_text_hin+" "+text
    target_texts.append(updated_text_hin)
    for char in input_text:
        if char not in input_characters:
            input_characters.add(char)
    for char in target_text:
        if char not in target_characters:
            target_characters.add(char)

input_characters=sorted(list(input_characters))
target_characters=sorted(list(target_characters))
num_encoder_tokens=len(input_characters)
num_decoder_tokens=len(target_characters)
max_encoder_seq_length=max([len(txt) for txt in input_texts])
max_decoder_seq_length=max([len(txt) for txt in target_texts])

print("Number of samples:", len(input_texts))
print("Number of unique input tokens:", num_encoder_tokens)
print("Number of unique output tokens:", num_decoder_tokens)
print("Max sequence length for inputs:", max_encoder_seq_length)
print("Max sequence length for outputs:", max_decoder_seq_length)

input_token_index=dict([(char,i) for i, char in enumerate(input_characters)])
target_token_index=dict([(char,i) for i, char in enumerate(target_characters)])

encoder_input_data=np.zeros((len(input_texts),max_encoder_seq_length,
                             num_encoder_tokens),dtype="float32",)
decoder_input_data = np.zeros((len(input_texts), max_decoder_seq_length,
                               num_decoder_tokens),dtype="float32",)
decoder_target_data=np.zeros((len(input_texts),max_decoder_seq_length,
                             num_decoder_tokens),dtype="float32",)

for i, (input_text, target_text) in enumerate(zip(input_texts,target_texts)):
    for t, char in enumerate(input_text):
        encoder_input_data[i, t, input_token_index[char]]=1.0
        encoder_input_data[i,t+1:, input_token_index[" "]]=1.0
        for t, char in enumerate(target_text):
            decoder_input_data[i,t,target_token_index[char]]=1.0
            if t>0:
                decoder_target_data[i,t-1,target_token_index[char]]=1.0
        decoder_input_data[i,t+1:,target_token_index[" "]]=1.0
        decoder_target_data[i,t:, target_token_index[" "]]=1.0

# Define an input sequence and process it.
encoder_inputs = tf.keras.Input(shape=(None, num_encoder_tokens))
encoder = tf.keras.layers.LSTM(latent_dim, return_state=True)
encoder_outputs, state_h, state_c = encoder(encoder_inputs)

# We discard `encoder_outputs` and only keep the states.
encoder_states = [state_h, state_c]

# Set up the decoder, using `encoder_states` as initial state.
decoder_inputs = tf.keras.Input(shape=(None, num_decoder_tokens))

# We set up our decoder to return full output sequences,
# and to return internal states as well. We don't use the
# return states in the training model, but we will use them in inference.
decoder_lstm = tf.keras.layers.LSTM(latent_dim, return_sequences=True,
                                    return_state=True)
decoder_outputs,_,_=decoder_lstm(decoder_inputs, initial_state=encoder_states)
decoder_dense = tf.keras.layers.Dense(num_decoder_tokens, activation="softmax")
decoder_outputs = decoder_dense(decoder_outputs)


model = tf.keras.Model([encoder_inputs, decoder_inputs], decoder_outputs)

model.compile(optimizer="adam", loss="categorical_crossentropy",
              metrics=["accuracy"])
model.fit([encoder_input_data, decoder_input_data], decoder_target_data,
            batch_size=batch_size, epochs=epochs, validation_split=0.2,)
model.save("s2s_model.keras")

model = tf.keras.models.load_model("s2s_model.keras")

encoder_inputs = model.input[0]  # input_1
encoder_outputs, state_h_enc, state_c_enc = model.layers[2].output  # lstm_1
encoder_states = [state_h_enc, state_c_enc]
encoder_model = tf.keras.Model(encoder_inputs, encoder_states)

decoder_inputs = model.input[1]  # input_2
decoder_state_input_h = tf.keras.Input(shape=(latent_dim,))
decoder_state_input_c = tf.keras.Input(shape=(latent_dim,))
decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]
decoder_lstm = model.layers[3]
decoder_outputs, state_h_dec, state_c_dec = decoder_lstm( decoder_inputs,
                                        initial_state=decoder_states_inputs)
decoder_states = [state_h_dec, state_c_dec]
decoder_dense = model.layers[4]
decoder_outputs = decoder_dense(decoder_outputs)
decoder_model = tf.keras.Model([decoder_inputs] + decoder_states_inputs,
                               [decoder_outputs] + decoder_states)

# Reverse-lookup token index to decode sequences back to
# something readable.
reverse_input_char_index = dict((i, char) for char, i in
                                input_token_index.items())
reverse_target_char_index = dict((i, char) for char, i in
                                 target_token_index.items())

def decode_sequence(input_seq):
    # Encode the input as state vectors.
    states_value = encoder_model.predict(input_seq, verbose=0)

    # Generate empty target sequence of length 1.
    target_seq = np.zeros((1, 1, num_decoder_tokens))
    # Populate the first character of target sequence with the start character.
    target_seq[0, 0, target_token_index["\t"]] = 1.0

    # Sampling loop for a batch of sequences
    # (to simplify, here we assume a batch of size 1).
    stop_condition = False
    decoded_sentence = ""
    while not stop_condition:
        output_tokens, h, c = decoder_model.predict([target_seq] +
                                                    states_value, verbose=0)

        # Sample a token
        sampled_token_index = np.argmax(output_tokens[0, -1, :])
        sampled_char = reverse_target_char_index[sampled_token_index]
        decoded_sentence += sampled_char

        # Exit condition: either hit max length
        # or find stop character.
        if sampled_char=="\n" or len(decoded_sentence)>max_decoder_seq_length:
            stop_condition = True

        # Update the target sequence (of length 1).
        target_seq = np.zeros((1, 1, num_decoder_tokens))
        target_seq[0, 0, sampled_token_index] = 1.0

        # Update states
        states_value = [h, c]
    return decoded_sentence

for seq_index in [4,16,85,102,402,824]:
    # Take one sequence (part of the training set)
    # for trying out decoding.
    input_seq = encoder_input_data[seq_index : seq_index + 1]
    decoded_sentence = decode_sequence(input_seq)
    print("-")
    print("Input sentence:", input_texts[seq_index])
    print("Decoded sentence:", decoded_sentence)
'''
while True:
    user_input = input(
        "Enter a Hindi sentence to translate (or 'q' to quit): ")
    if user_input.lower() == 'q':
        break
    translated_sentence = decode_sequence(user_input)
    print("Translated sentence:", translated_sentence)
'''