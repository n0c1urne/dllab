import sys
sys.path.append("../")

import numpy as np
import matplotlib.pyplot as plt
import gym
from dqn.dqn_agent_car_racing import DQNAgent
from dqn.conv_networks import CNN, CNNTargetNetwork
from tensorboard_evaluation import *
import itertools as it
from utils import *
from model2 import Model2

def id_to_action(action_id):
    # determine final action
    a = [0,0,0]

    if action_id == 1:
        a[0] = -1
    elif action_id == 2:
        a[0] = 1
    elif action_id == 3:
        a[1] = 1
    elif action_id == 4:
        a[2] = 0.2

    return a

def lane_penalty(state):
    eps = 2.0

    ## check for green values (174.??? or 192.???)
    offlane_left = abs(state[68, 44] - 174.0) < eps or abs(state[68, 44] - 192.0) < eps
    offlane_right = abs(state[68, 51] - 174.0) < eps or abs(state[68, 51] - 192.0) < eps

    if offlane_left and offlane_right:
        return -2.0  # bad buggy: completely of track, high penalty
    elif offlane_left or offlane_right:
        return -0.5  # one side off track
    else:
        return 0.0


def run_episode(env, agent, deterministic, skip_frames=0,  do_training=True, rendering=False, max_timesteps=1000, history_length=0, expert_agent=None, apply_lane_penalty=False):
    """
    This methods runs one episode for a gym environment.
    deterministic == True => agent executes only greedy actions according the Q function approximator (no random actions).
    do_training == True => train agent
    """

    stats = EpisodeStats()

    # Save history
    image_hist = []

    step = 0
    state = env.reset()

    # fix bug of corrupted states without rendering in gym environment
    env.viewer.window.dispatch_events()

    # append image history to first state
    state = state_preprocessing(state)
    image_hist.extend([state] * (history_length + 1))
    state = np.array(image_hist).reshape(96, 96, history_length + 1)

    while True:

        # TODO: get action_id from agent
        # Hint: adapt the probabilities of the 5 actions for random sampling so that the agent explores properly.
        # action_id = agent.act(...)
        # action = your_id_to_action_method(...)


        if expert_agent:
            res = expert_agent.sess.run([expert_agent.softmax], feed_dict = { expert_agent.x: state.reshape((1, 96, 96, 1)) })[0][0]
            action_id = np.argmax(res)
        else:
            action_id = agent.act(state=state, deterministic=deterministic)

        action = id_to_action(action_id)

        # Hint: frame skipping might help you to get better results.
        reward = 0
        for _ in range(skip_frames + 1):
            next_state, r, terminal, info = env.step(action)
            reward += r

            # now we need the processing on every frame
            next_state = state_preprocessing(next_state)

            if step > 50 and apply_lane_penalty:   # only after zooming, apply lane penalty
                reward += lane_penalty(next_state) # add lane penalty to reward

            if rendering:
                env.render()

            if terminal:
                 break

        print(step, reward)

        if step > 50 and False: # remove false for debugging
            plt.figure()
            img = next_state.copy()
            img[68, 43] = 255
            img[68, 52] = 255
            plt.imshow(img.reshape((96, 96)), cmap='gray')
            plt.show()

        image_hist.append(next_state)
        image_hist.pop(0)
        next_state = np.array(image_hist).reshape(96, 96, history_length + 1)

        if do_training:
            agent.train(state, action_id, next_state, reward, terminal)

        stats.step(reward, action_id)

        state = next_state

        if terminal or (step * (skip_frames + 1)) > max_timesteps:
            break

        step += 1

    return stats


def train_online(env, agent, num_episodes, history_length=0, model_dir="./models_carracing", tensorboard_dir="./tensorboard", expert_agent=None):

    if not os.path.exists(model_dir):
        os.mkdir(model_dir)

    print("... train agent")
    tensorboard = Evaluation(os.path.join(tensorboard_dir, "train"), ["episode_reward", "straight", "left", "right", "accel", "brake"])

    for i in range(num_episodes):
        print("episode %d" % i)

        # Hint: you can keep the episodes short in the beginning by changing max_timesteps (otherwise the car will spend most of the time out of the track)
        #apply_lane_penalty = False
        #if i > 50:
        #    expert_agent = None
        #    apply_lane_penalty = True

        stats = run_episode(env, agent, max_timesteps=min(i*2 + 100, 1000), deterministic=False, do_training=True, rendering=True, skip_frames=2, expert_agent=expert_agent, apply_lane_penalty=True)

        tensorboard.write_episode_data(i, eval_dict={ "episode_reward" : stats.episode_reward,
                                                      "straight" : stats.get_action_usage(STRAIGHT),
                                                      "left" : stats.get_action_usage(LEFT),
                                                      "right" : stats.get_action_usage(RIGHT),
                                                      "accel" : stats.get_action_usage(ACCELERATE),
                                                      "brake" : stats.get_action_usage(BRAKE)
                                                      })

        print(stats.episode_reward)
        # TODO: evaluate agent with deterministic actions from time to time
        # ...

        if i % 100 == 0 or (i >= num_episodes - 1):
            agent.saver.save(agent.sess, os.path.join(model_dir, "dqn_agent.ckpt"))

    tensorboard.close_session()

def state_preprocessing(state):
    return rgb2gray(state).reshape(96, 96) #/ 255.0

if __name__ == "__main__":

    env = gym.make('CarRacing-v0').unwrapped

    # Define Q network, target network and DQN agent
    q = CNN(env.observation_space.shape[0], 5)
    q_target = CNNTargetNetwork(96*96, 5)

    # 2. init DQNAgent (see dqn/dqn_agent.py)
    agent = DQNAgent(q, q_target, 5)

    # expert agent from ex 3
    #expert_agent = Model2(name='model2', dropout=1.0)
    #expert_agent.load("./models_carracing/model2_30000.ckpt")


    train_online(env, agent, num_episodes=10000, history_length=0, model_dir="./models_carracing")

