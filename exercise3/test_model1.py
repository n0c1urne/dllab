import tensorflow as tf
from test_utils import test_agent
from model1 import Model1

# seed tensorflow for stable results
tf.set_random_seed(42)

agent = Model1(name='model1')

# load specific snapshot
agent.load('models/model1.ckpt')

test_agent(agent, use_softmax = True)


