import tensorflow as tf
from test_utils import test_agent
from model3 import Model3

# seed tensorflow for stable results
tf.set_random_seed(42)

agent = Model3(name='model3', history_len=5)

# load specific snapshot
agent.load("./models/model3_30000.ckpt")

test_agent(agent, history_len=5, use_softmax = True)


