import tensorflow as tf
from test_utils import test_agent
from model4 import Model4

# seed tensorflow for stable results
tf.set_random_seed(42)

agent = Model4(name='model4', history_len=10)

# load specific snapshot
agent.load("./models/model4_40000.ckpt")

test_agent(agent, history_len=10, use_softmax = True, factors=[0.5, 0.5, 0.4], startup=True)


