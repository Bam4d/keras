
import pytest
import numpy as np
from keras.models import Sequential
from keras.layers.recurrent import SimpleRNN
from numpy.testing import assert_allclose
import theano
from keras import backend as K
from keras.layers.recurrent import Recurrent, NeuralStack


# We're not testing the controller here so mock it
def _runner(layer_class, nb_samples, vector_size, timesteps, input_dim, controller_output_dim, output_dim):

    for ret_seq in [True, False]:
        layer = NeuralStack(layer_class, controller_output_dim, output_dim, vector_size, nb_samples, weights=None, return_sequences=ret_seq, input_shape=(timesteps, input_dim))
        layer.input = K.variable(np.ones((nb_samples, timesteps, input_dim)))
        layer.get_config()

        for train in [True, False]:
            out = K.eval(layer.get_output(train))
            # Make sure the output has the desired shape
            if ret_seq:
                assert(out.shape == (nb_samples, timesteps, output_dim))
            else:
                assert(out.shape == (nb_samples, output_dim))

            mask = layer.get_output_mask(train)

    # check statefulness
    layer = NeuralStack(layer_class, controller_output_dim, output_dim, vector_size, nb_samples, weights=None, return_sequences=False, batch_input_shape=(nb_samples, timesteps, input_dim))

    model = Sequential()
    model.add(layer)
    model.compile(optimizer='sgd', loss='mse')
    out1 = model.predict(np.ones((nb_samples, timesteps, input_dim)))
    assert(out1.shape == (nb_samples, output_dim))

    # train once so that the states change
    model.train_on_batch(np.ones((nb_samples, timesteps, input_dim)),
                         np.ones((nb_samples, output_dim)))
    out2 = model.predict(np.ones((nb_samples, timesteps, input_dim)))

    # if the state is not reset, output should be different
    assert(out1.max() != out2.max())

class MockController():

    def __init__(self, output_dim, **kwargs):
        self.output_dim = output_dim
        self.params = []

    def build(self):
        pass

    def step(self, x, states):
        pass

@pytest.mark.skipif(K._BACKEND == 'tensorflow',
                    reason='currently not working with TensorFlow')
def test_compute_read():
    '''
    This check is to show that this implementation gives the same result as https://github.com/PrajitR/NeuralStacksQueues/blob/master/test/memoryTest.lua
    '''

    import theano.tensor as T

    pop = K.variable(np.array([[0.0], [0.0]]).T)
    push = K.variable(np.array([[1.0], [0.8]]).T)
    vec = K.variable(np.array([[0.0, 0.0, 3.0], [0.0, 0.0, 3.0]]).T)

    batch_size = 2
    vector_size = 3
    time_steps = 10
    input_dim = 3
    output_dim = 5
    controller_output_dim = 6

    step_number = 2

    stack = NeuralStack(MockController, controller_output_dim, output_dim, vector_size, batch_size, input_shape=(time_steps, input_dim))

    #stack.build()

    stack.step_count = K.variable(step_number, dtype=np.int32)
    stack.vectors = K.variable(np.zeros([stack.stack_vector_size, time_steps, batch_size]))
    stack.strengths = K.variable(np.zeros([time_steps, batch_size]))
    stack.strengths = T.set_subtensor(stack.strengths[:step_number, :], K.variable(np.array([[0.5,0.4], [0.5,0.4]]).T))
    stack.vectors = T.set_subtensor(stack.vectors[:, :step_number, :], K.variable(np.array([[[1.0,0.0,0.0],[0.0,2.0,0.0]], [[1.0,0.0,0.0],[0.0,2.0,0.0]]]).T))

    vec, s, r = stack._step(pop, push, vec)

    assert np.allclose(K.eval(s), np.array([[0.5, 0.4, 1.0], [0.5, 0.4, 0.8]]).T, atol=0.001)
    assert np.allclose(K.eval(vec), np.array([[[1.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 3.0]],[[1.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 3.0]]]).T, atol=0.001)
    assert np.allclose(K.eval(r), np.array([[0.0, 0.0, 3.0], [0.0, 0.4, 2.4]]).T, atol=0.001)


@pytest.mark.skipif(K._BACKEND == 'tensorflow',
                    reason='currently not working with TensorFlow')
def test_neural_stack_step():
    '''
    numeric checks to test that the neural stack is acting as it should do in forward pass
    (assuming that this means it will work OK in backward pass)
    '''

    import theano.tensor as T

    pop = K.variable(np.array([[0.2], [0.2], [0.2]]).T)
    push = K.variable(np.array([[0.4], [1.0], [0.2]]).T)
    vec = K.variable(np.array([[1,0,1], [1,0,1], [1,0,1]]).T)

    batch_size = 3
    vector_size = 3
    time_steps = 10
    input_dim = 3
    output_dim = 5
    controller_output_dim = 6

    step_number = 3

    stack = NeuralStack(MockController, controller_output_dim, output_dim, vector_size, batch_size, input_shape=(time_steps, input_dim))

    stack.step_count = K.variable(step_number, dtype=np.int32)
    stack.vectors = K.variable(np.zeros([stack.stack_vector_size, time_steps, batch_size]))
    stack.strengths = K.variable(np.zeros([time_steps, batch_size]))
    stack.strengths = T.set_subtensor(stack.strengths[:step_number, :], K.variable(np.array([[0.4,0.1,0.3],[0.5,0.4,1.0],[0.3,0.3,0.3]]).T))
    stack.vectors = T.set_subtensor(stack.vectors[:, :step_number, :], K.variable(np.array([[[1,1,1],[2,2,2],[3,0,3]],[[1,0,0],[0,2,0],[0,0,3]],[[1,0,0],[0,2,0],[0,0,3]]]).T))

    vec, s, r = stack._step(pop, push, vec)

    assert np.allclose(K.eval(s), np.array([[0.4,0.1,0.1,0.4], [0.5,0.4,0.8,1.0],[0.3,0.3,0.1,0.2]]).T, atol=0.001)
    assert np.allclose(K.eval(vec), np.array([[[1,1,1], [2,2,2], [3,0,3], [1,0,1]], [[1,0,0], [0,2,0], [0,0,3], [1,0,1]], [[1,0,0], [0,2,0], [0,0,3], [1,0,1]], ]).T, atol=0.001)
    assert np.allclose(K.eval(r), np.array([[1.3,0.6,1.3], [1,0,1], [0.5,0.6,0.5]]).T, atol=0.001)

def test_reverse_cumalative_sum():

    stack = NeuralStack(MockController, 5, 6, 3, 1, input_shape=(10, 4))

    seq = K.variable(np.array([[0.4,0.1,0.3],[0.5,0.4,1.0],[0.3,0.3,0.3]]))
    sum = stack._rev_cumsum(seq)

    assert np.allclose(K.eval(sum), np.array([[1.2,0.8,1.6],[0.8,0.7,1.3],[0.3,0.3,0.3]]).T, atol=0.001)

def test_full_step():
    batch_size = 2
    vector_size = 3
    time_steps = 10
    input_dim = 4

    output_dim = 5
    controller_output_dim = 6

    stack = NeuralStack(SimpleRNN, controller_output_dim, output_dim, vector_size, batch_size, input_shape=(time_steps, input_dim))
    stack.step_count = K.variable(1, dtype=np.int32)
    stack.vectors = K.variable(np.zeros([stack.stack_vector_size, time_steps, batch_size]))
    stack.strengths = K.variable(np.zeros([time_steps, batch_size]))

    x = K.ones((batch_size, input_dim))
    states = [K.zeros((batch_size, vector_size))]+ stack._get_initial_controller_states(batch_size)

    output, states = stack.full_step(x, states)

    print K.eval(output)
    print K.eval(states[0])

def test_neural_stack_with_controller():

    batch_size = 2
    vector_size = 3
    time_steps = 2
    input_dim = 4

    output_dim = 5
    controller_output_dim = 6

    _runner(SimpleRNN, batch_size, vector_size, time_steps, input_dim, controller_output_dim, output_dim)

if __name__ == '__main__':
    pytest.main([__file__])