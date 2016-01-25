
import pytest
import numpy as np
from keras.models import Sequential
from numpy.testing import assert_allclose

from keras import backend as K
from keras.layers import core

@pytest.mark.skipif(K._BACKEND == 'tensorflow',
                    reason='currently not working with TensorFlow')
def test_compute_read():
    import theano.tensor as T

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

    # We're not testing the controller here so mock it
    class MockController():
        def step(self, x, states):
            pass

    stack = core.NeuralStack(vector_size, time_steps, batch_size, MockController())

    stack.build()

    stack.step = 2
    stack.strengths = T.set_subtensor(stack.strengths[:stack.step, :], K.variable(np.array([[0.5,0.4], [0.5,0.4]]).T))
    stack.vectors = T.set_subtensor(stack.vectors[:, :stack.step, :], K.variable(np.array([[[1.0,0.0,0.0],[0.0,2.0,0.0]], [[1.0,0.0,0.0],[0.0,2.0,0.0]]]).T))

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

    # We're not testing the controller here so mock it
    class MockController():
        def step(self, x, states):
            pass

    stack = core.NeuralStack(vector_size, time_steps, batch_size, MockController())

    stack.build()

    stack.step = 3
    stack.strengths = T.set_subtensor(stack.strengths[:stack.step, :], K.variable(np.array([[0.4,0.1,0.3],[0.5,0.4,1.0],[0.3,0.3,0.3]]).T))
    stack.vectors = T.set_subtensor(stack.vectors[:, :stack.step, :], K.variable(np.array([[[1,1,1],[2,2,2],[3,0,3]],[[1,0,0],[0,2,0],[0,0,3]],[[1,0,0],[0,2,0],[0,0,3]]]).T))

    vec, s, r = stack._step(pop, push, vec)

    assert np.allclose(K.eval(s), np.array([[0.4,0.1,0.1,0.4], [0.5,0.4,0.8,1.0],[0.3,0.3,0.1,0.2]]).T, atol=0.001)
    assert np.allclose(K.eval(vec), np.array([[[1,1,1], [2,2,2], [3,0,3], [1,0,1]], [[1,0,0], [0,2,0], [0,0,3], [1,0,1]], [[1,0,0], [0,2,0], [0,0,3], [1,0,1]], ]).T, atol=0.001)
    assert np.allclose(K.eval(r), np.array([[1.3,0.6,1.3], [1,0,1], [0.5,0.6,0.5]]).T, atol=0.001)


if __name__ == '__main__':
    pytest.main([__file__])