import tensorflow as tf
from test_utils import test_agent
from model2 import Model2

# seed tensorflow for stable results
tf.set_random_seed(42)

agent = Model2(name='model2', dropout=1.0)

# load specific snapshot
agent.load("./models/model2_30000.ckpt")

test_agent(agent, use_softmax = True, factors=[0.5, 0.5, 0.2])


