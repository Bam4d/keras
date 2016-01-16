
import pytest
import numpy as np
from keras.models import Sequential
from numpy.testing import assert_allclose

from keras import backend as K
from keras.layers import core


@pytest.mark.skipif(K._BACKEND == 'tensorflow',
                    reason='currently not working with TensorFlow')
def test_neural_stack_step():
    '''
    numeric checks to test that the neural stack is acting as it should do in forward pass
    (assuming that this means it will work OK in backward pass)
    '''

    import theano.tensor as T

    pop = K.variable(np.array([[0.0], [0.2], [0.2]]).T)
    push = K.variable(np.array([[0.0], [1.0], [0.2]]).T)
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
    stack.strengths = T.set_subtensor(stack.strengths[:stack.step, :], K.variable(np.array([[0.5,0.4,1.0],[0.5,0.4,0.8],[0.3,0.3,0.3]]).T))
    stack.vectors = T.set_subtensor(stack.vectors[:, :stack.step, :], K.variable(np.array([[[1,1,1],[2,2,2],[3,0,3]],[[1,1,1],[2,2,2],[3,0,3]],[[1,0,0],[0,2,0],[0,0,3]]]).T))

    vec, s, r = stack._step(pop, push, vec)

    assert K.eval(vec) == 0


if __name__ == '__main__':
    pytest.main([__file__])