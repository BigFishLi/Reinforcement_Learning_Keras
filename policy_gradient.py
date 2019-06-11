#!/usr/bin/env python
# coding=utf-8

'''
__title__ = ''
__author__ = 'l00381098'
__mtime__ = '2019/6/10'
'''

import matplotlib.pyplot as plt
import gym
import tensorflow as tf
import numpy as np
from keras.layers import Input, Dense, Lambda
from keras.models import Model
from keras.optimizers import Adam
import keras.backend as K


class Policy_Gradient():
    def __init__(self, n_actions, n_features, learning_rate=0.01, reward_decay=0.9):
        # init some parameters
        self.n_actions = n_actions
        self.n_features = n_features
        self.lr = learning_rate
        self.gamma = reward_decay

        self.model = self.create_softmax_network()
        self.ep_obs, self.ep_as, self.ep_rs = [], [], []

    def my_loss(self, arg):
        action_pred, action_true, discount_episode_reward = arg
        action_true = K.cast(action_true, dtype=tf.int32)
        loss = K.sparse_categorical_crossentropy(action_true, action_pred)
        loss = loss * K.flatten(discount_episode_reward)
        return loss

    def create_softmax_network(self):
        state_input = Input(shape=(self.n_features,))
        action_input = Input(shape=(1,))
        discount_episode_reward_input = Input(shape=(1,))
        output = Dense(10, activation='relu', kernel_initializer='glorot_uniform')(state_input)
        output = Dense(2, activation='relu', kernel_initializer='glorot_uniform')(output)
        output = Dense(self.n_actions, activation='softmax', kernel_initializer='glorot_uniform')(output)
        output_layer = Lambda(self.my_loss)
        loss_output = output_layer([output, action_input, discount_episode_reward_input])
        model = Model(inputs=[state_input, action_input, discount_episode_reward_input], outputs=[loss_output, output])
        losses = [
            lambda y_true, y_pred: y_pred,  # loss is computed in Lambda layer
            lambda y_true, y_pred: K.zeros_like(y_pred),  # we only include this for the metrics
        ]
        model.compile(loss=losses, optimizer=Adam(lr=self.lr))
        print('----network summary----')
        print(model.summary())
        return model

    def choose_action(self, observation):
        state = observation.reshape(-1, 4)
        _, prob_weights = self.model.predict([state, np.zeros((1,1)), np.array([100]).reshape(-1, 1)])
        action = np.random.choice(range(prob_weights.shape[1]), p=prob_weights.ravel())
        return action

    def store_transition(self, s, a, r):
        self.ep_obs.append(s)
        self.ep_as.append(a)
        self.ep_rs.append(r)

    def learn(self):
        discounted_ep_rs = np.zeros_like(self.ep_rs)
        running_add = 0
        for t in reversed(range(0, len(self.ep_rs))):
            running_add = running_add * self.gamma + self.ep_rs[t]
            discounted_ep_rs[t] = running_add

        discounted_ep_rs -= np.mean(discounted_ep_rs)
        discounted_ep_rs /= np.std(discounted_ep_rs)

        # train on episode
        X = [self.ep_obs, self.ep_as, discounted_ep_rs]

        batch_size = len(self.ep_obs)
        y = [np.zeros(shape=(batch_size, 1)), np.zeros(shape=(batch_size, self.n_actions))]
        self.model.fit(X, y, batch_size=batch_size, verbose=0)
        self.ep_obs, self.ep_as, self.ep_rs = [], [], []  # empty episode data

def main():
    # initialize OpenAI Gym env and dqn agent
    env = gym.make(ENV_NAME)
    action_dim = env.action_space.n
    state_dim = env.observation_space.shape[0]
    agent = Policy_Gradient(action_dim,state_dim)
    reward_lst = []
    for episode in range(EPISODE):
        # initialize task
        state = env.reset()
        # Train
        for step in range(STEP):
            action = agent.choose_action(state)  # e-greedy action for train
            next_state, reward, done, _ = env.step(action)
            agent.store_transition(state, action, reward)
            state = next_state
            if done:
                agent.learn()
                break

        # Test every 100 episodes
        if episode % 10 == 0:
            total_reward = 0
            for i in range(TEST):
                state = env.reset()
                for j in range(STEP):
                    # env.render()
                    action = agent.choose_action(state)  # direct action for test
                    state, reward, done, _ = env.step(action)
                    total_reward += reward
                    if done:
                        break
            ave_reward = total_reward / TEST
            reward_lst.append(ave_reward)
            print('episode: ', episode, 'Evaluation Average Reward:', ave_reward)

    plt.figure()
    x = range(0, EPISODE, TEST)
    plt.plot(x, reward_lst)
    plt.xlabel('Epoch')
    plt.ylabel('Reward')
    plt.savefig('images/policy_gradient.jpg')

if __name__ == '__main__':
    # Hyper Parameters
    ENV_NAME = 'CartPole-v0'
    EPISODE = 300  # Episode limitation
    STEP = 300  # Step limitation in an episode
    TEST = 10  # The number of experiment test every 100 episode
    main()
