import numpy as np
from keras.models import Model
from keras.layers import Input, LSTM, Embedding, Dense
from keras.optimizers import Adam

data = open(r"C:\Users\admin\CCPS Research\virtual_test\piyush\eng_to_hin.txt", 'r', encoding='utf8', errors='ignore').read()
lines = data.split('\n')
lines = lines[:]

english = [l.split('\t')[0] for l in lines]
hindi = [l.split('\t')[1] for l in lines]
hindi = ['<s> '+h+' <e>' for h in hindi]

english_vocab = sorted(list(set([w for e in english for w in e.split()])))
hindi_vocab = sorted(list(set([w for h in hindi for w in h.split()])))

english_vocab_size = min(8000, len(english_vocab)) #vocab size
hindi_vocab_size = min(8000, len(hindi_vocab))

english_vocab = english_vocab[:english_vocab_size]
hindi_vocab = hindi_vocab[:hindi_vocab_size]

max_english_len = max([len(e.split()) for e in english])
max_hindi_len = max([len(h.split()) for h in hindi])

english_w_to_i = {w: i for i, w in enumerate(english_vocab)}
hindi_w_to_i = {w: i for i, w in enumerate(hindi_vocab)}
english_i_to_w = {i: w for i, w in enumerate(english_vocab)}
hindi_i_to_w = {i: w for i, w in enumerate(hindi_vocab)}

encoder_input_data = np.zeros((len(english), max_english_len), dtype='float32') #specifying datatype 
decoder_input_data = np.zeros((len(hindi), max_hindi_len), dtype='float32')
decoder_target_data = np.zeros((len(hindi), max_hindi_len, hindi_vocab_size), dtype='float32')

for i in range(len(english)):
    for j in range(len(english[i].split())):
        encoder_input_data[i, j] = english_w_to_i.get(english[i].split()[j], 0)
for i in range(len(hindi)):
    for j in range(len(hindi[i].split())):
        decoder_input_data[i, j] = hindi_w_to_i.get(hindi[i].split()[j], 0)
        if j > 0:
            decoder_target_data[i, j - 1, hindi_w_to_i.get(hindi[i].split()[j], 0)] = 1.0

units = 256
epochs = 100
batch_size = 32

encoder_inputs = Input(shape=(None,))
encoder_embedding = Embedding(english_vocab_size, units)(encoder_inputs)
encoder_LSTM = LSTM(units, return_sequences=True, return_state=True)
_, state_h, state_c = encoder_LSTM(encoder_embedding)
encoder_states = [state_h, state_c]

decoder_inputs = Input(shape=(None,))
decoder_embedding_layer = Embedding(hindi_vocab_size, units)
decoder_embedding = decoder_embedding_layer(decoder_inputs)
decoder_LSTM = LSTM(units, return_state=True, return_sequences=True)
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


def translate(input_seq):
    #batch dimension
    input_seq = input_seq.reshape(1, -1)
    states = encoder_model.predict(input_seq) #encoder states
    
    target_seq = np.zeros((1, 1)) #input for decoder
    target_seq[0, 0] = hindi_w_to_i['<s>']
    
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
        target_seq[0, 0] = index #updating target sequence
        
        states = [h, c] #updating states
    
    return target_sentence

for i in range(len(english)):
    print("-----------------------------------------------------------------------------------------------")
    print('English Sentence: ', english[i])
    print('Target Hindi Sentence: ', hindi[i])

    input_seq1 = encoder_input_data[i, :]
    print('Predicted Hindi Sentence1: ', translate(input_seq1))

    input_seq2 = encoder_input_data[i, :].reshape(1, -1)
    predicted_sentence2 = translate(input_seq2)
    print('Predicted Hindi Sentence2: ', predicted_sentence2)
