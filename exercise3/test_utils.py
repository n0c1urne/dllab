from __future__ import print_function

import tensorflow as tf
from datetime import datetime
import numpy as np
import gym
import os
import json

from model1 import Model1
from utils import *

def run_episode(env, agent, history_len, use_softmax, rendering=True, factors=[0.5, 1.0, 0.2], max_timesteps=1000, startup=False):

    episode_reward = 0
    step = 0

    # reset env
    state = env.reset()
    state_hist = np.zeros((1, 96, 96, history_len))

    # fill up state history with first states
    for i in range(history_len):
        state, r, _, _ = env.step([0,0,0])
        episode_reward += r
        step += 1

        state_hist[0,:,:,i] = rgb2gray(state)[np.newaxis, ...]

    while True:
        # preprocess the state in the same way than in in your preprocessing in train_agent.py
        state = rgb2gray(state)[np.newaxis, ...]

        # update state history - move last hist-1 states and add new
        new_state_hist = state_hist.copy()
        for i in range(history_len-1):
            new_state_hist[0, :, :, i] = new_state_hist[0, :, :, i+1]
        new_state_hist[0,:,:,history_len-1] = state
        state_hist = new_state_hist

        # ask agent for action
        res = agent.sess.run([agent.softmax], feed_dict = { agent.x: state_hist })[0][0]

        # determine final action
        a = [0,0,0]

        if use_softmax:
            # some weights on the softmax values
            # to make driving more smooth
            a[0] = (-res[1] + res[2])*factors[0]
            a[1] = res[3]*factors[1]
            a[2] = res[4]*factors[2]
        else:
            # determine max action and perform discrete action
            n = np.argmax(res)

            if n == 1:
                a[0] = -1
            elif n == 2:
                a[0] = 1
            elif n == 3:
                a[1] = 1
            elif n == 4:
                a[2] = 0.2

        # some help for the lstm model to get started
        if startup and step < 40:
            a[1] = 1

        next_state, r, done, info = env.step(a)
        episode_reward += r
        state = next_state
        step += 1

        if rendering:
            env.render("human")

        if done or step > max_timesteps:
            break

    return episode_reward


def test_agent(agent, history_len=1, use_softmax = False, factors=[0.5, 1.0, 0.2], startup=False):
    rendering = True                      # set rendering=False if you want to evaluate faster

    n_test_episodes = 15                  # number of episodes to test

    env = gym.make('CarRacing-v0').unwrapped

    # seed everything for stable results - note: tensorflow must also get seeded, done in the
    # training files
    env.seed(42)
    np.random.seed(42)

    episode_rewards = []
    for i in range(n_test_episodes):
        episode_reward = run_episode(env, agent, history_len, use_softmax, rendering=rendering, factors=factors, startup=startup)
        print("Episode", str(i+1)+':', episode_reward)
        episode_rewards.append(episode_reward)

    # save results in a dictionary and write them into a .json file
    results = dict()
    results["episode_rewards"] = episode_rewards
    results["mean"] = np.array(episode_rewards).mean()
    results["std"] = np.array(episode_rewards).std()

    fname = "results/"+agent.name+".json"
    fh = open(fname, "w")
    json.dump(results, fh)

    env.close()

    print('Mean:', results['mean'])
    print('Std.:', results['std'])
