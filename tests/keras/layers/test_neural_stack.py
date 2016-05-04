
import pytest
import numpy as np
from keras.models import Sequential
from keras.layers.recurrent import SimpleRNN
from numpy.testing import assert_allclose
import theano
from keras import backend as K
from keras.layers.recurrent import Recurrent, NeuralStack
from keras.engine import Model, Input


# We're not testing the controller here so mock it
def layer_test(layer_cls, vector_size, controller_output_dim, output_dim, kwargs={}, input_shape=None, input_dtype=None,
               input_data=None, expected_output=None, expected_output_dtype=None):
    '''Test routine for a layer with a single input tensor
    and single output tensor.
    '''
    if input_data is None:
        assert input_shape
        if not input_dtype:
            input_dtype = K.floatx()
        input_data = (10 * np.random.random(input_shape)).astype(input_dtype)
    elif input_shape is None:
        input_shape = input_data.shape

    if expected_output_dtype is None:
        expected_output_dtype = input_dtype

    # instantiation
    layer = NeuralStack(layer_cls,  controller_output_dim, output_dim, vector_size, **kwargs)

    # test get_weights , set_weights
    weights = layer.get_weights()
    layer.set_weights(weights)

    # test and instantiation from weights
    # if 'weights' in inspect.getargspec(layer_cls.__init__):
    #     kwargs['weights'] = weights
    #     layer = layer_cls(**kwargs)

    # test in functional API
    x = Input(batch_shape=input_shape, dtype=input_dtype)
    y = layer(x)
    assert K.dtype(y) == expected_output_dtype

    model = Model(input=x, output=y)
    model.compile('rmsprop', 'mse')

    expected_output_shape = layer.get_output_shape_for(input_shape)
    actual_output = model.predict(input_data)
    actual_output_shape = actual_output.shape
    assert expected_output_shape == actual_output_shape
    if expected_output is not None:
        assert_allclose(actual_output, expected_output, rtol=1e-3)

    # test serialization
    #model_config = model.get_config()
    #model = Model.from_config(model_config)
    #model.compile('rmsprop', 'mse')

    # test as first layer in Sequential API
    # layer_config = layer.get_config()
    # layer_config['batch_input_shape'] = input_shape
    # layer = layer.__class__.from_config(layer_config)
    #
    # model = Sequential()
    # model.add(layer)
    # model.compile('rmsprop', 'mse')
    # actual_output = model.predict(input_data)
    # actual_output_shape = actual_output.shape
    # assert expected_output_shape == actual_output_shape
    # if expected_output is not None:
    #     assert_allclose(actual_output, expected_output, rtol=1e-3)

    # test JSON serialization
    #json_model = model.to_json()
    #model = model_from_json(json_model)

    # for further checks in the caller function
    return actual_output

class MockController():

    def __init__(self, output_dim, **kwargs):
        self.output_dim = output_dim
        self.trainable_weights = []

    def build(self, input_shape):
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

    stack = NeuralStack(MockController, controller_output_dim, output_dim, vector_size, input_shape=(batch_size ,time_steps, input_dim))

    stack.build((batch_size ,time_steps, input_dim))

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

    stack = NeuralStack(MockController, 5, 6, 3, input_shape=(10, 4))

    seq = K.variable(np.array([[0.4,0.1,0.3],[0.5,0.4,1.0],[0.3,0.3,0.3]]))
    sum = stack._rev_cumsum(seq)

    assert np.allclose(K.eval(sum), np.array([[1.2,0.8,1.6],[0.8,0.7,1.3],[0.3,0.3,0.3]]).T, atol=0.001)


def test_neural_stack_with_controller():

    batch_size = 2
    vector_size = 10
    time_steps = 50
    input_dim = 20

    output_dim = 5
    controller_output_dim = 4

    layer_test(SimpleRNN, vector_size, controller_output_dim, output_dim, input_shape=(batch_size, time_steps, input_dim))

if __name__ == '__main__':
    pytest.main([__file__])