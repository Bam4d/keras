
from keras.models import Sequential
from keras.layers.core import Activation
from keras.layers import NeuralStack, recurrent
from keras.optimizers import RMSprop, Adagrad
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

    def one_hot(self, indeces, maxlen=None):
        maxlen = maxlen if maxlen else self.maxlen
        X = np.zeros((maxlen, len(self.chars)))
        for i, c in enumerate(indeces):
            X[i, c] = 1
        return X

    def decode(self, X, calc_argmax=True):
        if calc_argmax:
            X = X.argmax(axis=-1)
        return ''.join(self.indices_char[x] for x in X)


def generate_sequences(lookup_table, number_of_sequences, min_sequence_length, max_sequence_length):

    padded_sequence_length = max_sequence_length*2+3
    X = np.zeros((number_of_sequences, padded_sequence_length, len(lookup_table.chars)))
    Y = np.zeros((number_of_sequences, padded_sequence_length, len(lookup_table.chars)))
    for s in range(number_of_sequences):

        # have to take into account the start, stop and reverse chars, and divide the sequence by two so we can reverse
        sequence_length = np.random.randint(low=min_sequence_length, high=max_sequence_length)
        sequence = np.random.randint(low=0, high=len(lookup_table.chars)-3, size=sequence_length)

        # the start, stop and reverse characters
        start = 26 # {
        reverse = 27 # |
        stop = 28 # }

        stop_seq = np.ones((sequence_length))*stop
        start_seq = np.ones((sequence_length))*start

        full_x_sequence = np.concatenate([[start],  sequence,  [reverse], sequence[::-1], [stop]])
        full_y_sequence = np.concatenate([[start],  sequence, [reverse], sequence[::-1], [stop]])

        x = lookup_table.one_hot(full_x_sequence)
        y = lookup_table.one_hot(full_y_sequence)

        # Pad x and y with stop symbol
        for k in range(len(full_x_sequence), padded_sequence_length):
            x[k, stop] = 1
            y[k, stop] = 1

        X[s] = x
        Y[s] = y

    return X, Y

# Number of sequences in the test set to generate
NUMBER_OF_SEQUENCES = 1000

# This is the list of characters to  we will learn to reverse
chars = '{}|ABCDEFGHIJKLMNOPQRSTUVWXYZ'
# { start character
# } stop character
# | reverse character

# This is the max sequence length plus the reversal, plus the start, stop and reverse characters
MAX_SEQUENCE_LENGTH = 10
MIN_SEQUENCE_LENGTH = 3

PADDED_SEQUENCE_LENGTH = MAX_SEQUENCE_LENGTH*2+3

RNN = recurrent.SimpleRNN
CONTROLLER_OUTPUT_SIZE = 50
STACK_VECTOR_SIZE = 50
OUTPUT_SIZE = len(chars)
BATCH_SIZE = 10

# Have to add the start, stop and reverse chars
lookup_table = CharacterTable(chars, PADDED_SEQUENCE_LENGTH)

print 'Generating training data...'
X, Y = generate_sequences(lookup_table, NUMBER_OF_SEQUENCES, MIN_SEQUENCE_LENGTH, MAX_SEQUENCE_LENGTH)

print 'Building model...'
model = Sequential()

neural_stack_layer = NeuralStack(RNN, CONTROLLER_OUTPUT_SIZE, OUTPUT_SIZE, STACK_VECTOR_SIZE, return_sequences=True,
                                 batch_input_shape=(BATCH_SIZE, PADDED_SEQUENCE_LENGTH, len(chars)))

model.add(neural_stack_layer)
model.add(Activation('softmax'))

print 'Compiling model..'
rmsprop = RMSprop(clipvalue=1.0)
adagrad = Adagrad()
model.compile(loss='categorical_crossentropy',
              optimizer=rmsprop,
              metrics=['accuracy'])

print 'Model compiled..'

print 'Fitting..'
res = model.fit(X, Y, batch_size=BATCH_SIZE, nb_epoch=1)

test_X, test_Y = generate_sequences(lookup_table, BATCH_SIZE, MIN_SEQUENCE_LENGTH, MAX_SEQUENCE_LENGTH)

pred = model.predict(test_X, batch_size=BATCH_SIZE)

#print('Test score:', score)
#print('Test accuracy:', acc)

for t in range(0, BATCH_SIZE):
    print np.argmax(test_Y[t], axis=1)
    print np.argmax(pred[t], axis=1)
    print "----"
