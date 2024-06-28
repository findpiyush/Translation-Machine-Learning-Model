#original model link: https://github.com/priyadarshiguha/English-to-Hindi-NMT-Model-Using-RNN

import numpy as np
from keras.models import Model
from keras.layers import Input, LSTM, Embedding, Dense
from keras.optimizers import Adam
from keras.preprocessing.sequence import pad_sequences

import time

start_time = time.time()

data = open(r"C:\Users\admin\CCPS Research\virtual_test\piyush\eng_to_hin.txt", 'r', encoding='utf8', errors='ignore').read()
lines = data.split('\n')
lines = lines[:]

#=--------------------------
english = [l.split('\t')[0] for l in lines]
hindi = [l.split('\t')[1] for l in lines]
hindi = ['<s> '+h+' <e>' for h in hindi]

english_vocab = sorted(list(set([w for e in english for w in e.split()])))
hindi_vocab = sorted(list(set([w for h in hindi for w in h.split()])))

english_vocab_size = min(8000, len(english_vocab))
hindi_vocab_size = min(8000, len(hindi_vocab))

english_vocab = english_vocab[:english_vocab_size]
hindi_vocab = hindi_vocab[:hindi_vocab_size]

max_english_len = max([len(e.split()) for e in english])
max_hindi_len = max([len(h.split()) for h in hindi])

english_w_to_i = {w: i for i, w in enumerate(english_vocab)}
hindi_w_to_i = {w: i for i, w in enumerate(hindi_vocab)}
english_i_to_w = {i: w for i, w in enumerate(english_vocab)}
hindi_i_to_w = {i: w for i, w in enumerate(hindi_vocab)}

#-----------------
def sentences_to_indices(sentences, word_to_index, max_len):
    indices = [[word_to_index.get(w, 0) for w in s.split()] for s in sentences]
    return pad_sequences(indices, maxlen=max_len, padding='post')

encoder_input_data = sentences_to_indices(english, english_w_to_i, max_english_len)
decoder_input_data = sentences_to_indices(hindi, hindi_w_to_i, max_hindi_len)

decoder_target_data = np.zeros((len(hindi), max_hindi_len, hindi_vocab_size), dtype='float32')
for i, seq in enumerate(decoder_input_data):
    for t in range(1, len(seq)):
        decoder_target_data[i, t - 1, seq[t]] = 1.0

units = 256
epochs = 100
batch_size = 32

encoder_inputs = Input(shape=(None,))
encoder_embedding = Embedding(english_vocab_size, units)(encoder_inputs)
encoder_LSTM = LSTM(units, return_sequences=False, return_state=True)
_, state_h, state_c = encoder_LSTM(encoder_embedding)
encoder_states = [state_h, state_c]

decoder_inputs = Input(shape=(None,))
decoder_embedding_layer = Embedding(hindi_vocab_size, units)
decoder_embedding = decoder_embedding_layer(decoder_inputs)
decoder_LSTM = LSTM(units, return_sequences=True, return_state=True)
decoder_outputs, _, _ = decoder_LSTM(decoder_embedding, initial_state=encoder_states)
decoder_dense = Dense(hindi_vocab_size, activation='softmax')
decoder_outputs = decoder_dense(decoder_outputs)

model = Model([encoder_inputs, decoder_inputs], decoder_outputs)

model.compile(optimizer='adam', loss='categorical_crossentropy')

model.fit([encoder_input_data, decoder_input_data], decoder_target_data,
          epochs=epochs, batch_size=batch_size, verbose=1)

weights = 'nmt_weights.weights.h5'
model.save_weights(weights)
model.load_weights(weights)

encoder_model = Model(encoder_inputs, encoder_states)

decoder_state_input_h = Input(shape=(units,))
decoder_state_input_c = Input(shape=(units,))
decoder_state_inputs = [decoder_state_input_h, decoder_state_input_c]

decoder_embedding_2 = decoder_embedding_layer(decoder_inputs)
decoder_output_2, state_h_2, state_c_2 = decoder_LSTM(decoder_embedding_2, initial_state=decoder_state_inputs)
decoder_output_2 = decoder_dense(decoder_output_2)
decoder_states_2 = [state_h_2, state_c_2]

decoder_model = Model([decoder_inputs]+decoder_state_inputs, [decoder_output_2]+decoder_states_2)

train_time = time.time()
print(f"Train time taken1: {train_time - start_time} seconds")

def translate(input_seq):
    input_seq = input_seq.reshape(1, -1)
    states = encoder_model.predict(input_seq)
    
    target_seq = np.zeros((1, 1))
    target_seq[0, 0] = hindi_w_to_i.get('<s>', 0)
    
    stop = False
    target_sentence = ''
    
    while not stop:
        output, h, c = decoder_model.predict([target_seq] + states)
        index = np.argmax(output[0, -1, :])
        char = hindi_i_to_w.get(index, '')
        target_sentence += ' ' + char
        
        if char == '<e>' or len(target_sentence.split()) > max_hindi_len:
            stop = True
        
        target_seq = np.zeros((1, 1))
        target_seq[0, 0] = index
        
        states = [h, c]
    
    return target_sentence.strip()

results = []
for i in range(len(english)):
    print("-----------------------------------------------------------------------------------------------")
    print('English Sentence: ', english[i])
    print('Target Hindi Sentence: ', hindi[i])

    input_seq1 = encoder_input_data[i, :]
    print('Predicted Hindi Sentence1: ', translate(input_seq1))

    input_seq2 = encoder_input_data[i, :].reshape(1, -1)
    predicted_sentence2 = translate(input_seq2)
    print('Predicted Hindi Sentence2: ', predicted_sentence2)

    results.append((english[i], hindi[i], predicted_sentence2))


train_time2 = time.time()
print(f"Train time taken2: {train_time2 - start_time} seconds")


#----------------------------better print
for i, (eng, target, pred) in enumerate(results):
    print(f"\nSample {i + 1}")
    print(f"English: {eng}")
    print(f"Target Hindi: {target}")
    print(f"Predicted Hindi: {pred}")
   
    
end_time = time.time()
print(f"Total time taken: {end_time - start_time} seconds")