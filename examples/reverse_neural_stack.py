
from __future__ import print_function
from keras.models import Sequential, slice_X
from keras.layers.core import Activation, Dropout, Dense
from keras.layers import NeuralStack, recurrent
import numpy as np

class CharacterTable(object):
    '''
    Given a set of characters:
    + Encode them to a one hot integer representation
    + Decode the one hot integer representation to their character output
    + Decode a vector of probabilties to their character output
    '''
    def __init__(self, chars, maxlen):
        self.chars = sorted(set(chars))
        self.char_indices = dict((c, i) for i, c in enumerate(self.chars))
        self.indices_char = dict((i, c) for i, c in enumerate(self.chars))
        self.maxlen = maxlen

    def encode(self, C, maxlen=None):
        maxlen = maxlen if maxlen else self.maxlen
        X = np.zeros((maxlen, len(self.chars)))
        for i, c in enumerate(C):
            X[i, self.char_indices[c]] = 1
        return X

    def decode(self, X, calc_argmax=True):
        if calc_argmax:
            X = X.argmax(axis=-1)
        return ''.join(self.indices_char[x] for x in X)


def generate_sequences(lookup_table, number_of_sequences, max_sequence_length):

    X = []
    for s in range(number_of_sequences):

        # have to take into account the start, stop and reverse chars, and divide the sequence by two so we can reverse
        sequence_length = np.random.randint(high=(max_sequence_length-3/2))
        sequence = np.random.randint(low=3, high=len(lookup_table.chars), size=sequence_length)

        # the start, stop and reverse characters
        start = 0 # {
        stop = 1 # }
        reverse = 2 # |
        full_sequence = np.concatenate([start, sequence, reverse, sequence[::-1], stop])

        x = np.zeros((max_sequence_length, lookup_table.maxlen))
        for i, c in enumerate(full_sequence):
            x[i, c] = 1

        X.append(x)

        print lookup_table.decode(X)

    return X



# Number of sequences in the test set to generate
NUMBER_OF_SEQUENCES = 10000
STACK_VECTOR_SIZE = 100
CONTROLLER_HIDDEN_SIZE = 100


# This is the list of characters to  we will learn to reverse
chars = '{}|ABCDEFGHIJKLMNOPQRSTUVWXYZ'
# { start character
# } stop character
# | reverse character

# this is the max sequence length plus the reversal
MAX_SEQUENCE_LENGTH = 500

RNN = recurrent.LSTM
HIDDEN_SIZE = 128
BATCH_SIZE = 128

# have to add the start, stop and reverse chars
lookup_table = CharacterTable(chars, MAX_SEQUENCE_LENGTH)

print('Build model...')
model = Sequential()

neural_stack_layer = NeuralStack(RNN, BATCH_SIZE, STACK_VECTOR_SIZE, MAX_SEQUENCE_LENGTH, len(chars), len(chars))

model.add(neural_stack_layer)
model.add(Dropout(0.2))
model.add(Dense(len(chars)))
model.add(Activation('softmax'))

model.compile(loss='categorical_crossentropy', optimizer='rmsprop')