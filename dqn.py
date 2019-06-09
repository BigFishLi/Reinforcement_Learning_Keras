#! /usr/bin/env python
#-*- coding:utf-8 -*-
"""
@time: 2019/06/02
"""

"""
NIPS 2013 DQN
Deep Q-Learning算法的基本思路来源于Q-Learning。
但是和Q-Learning不同的地方在于，它的Q值的计算不是直接通过状态值s和动作来计算，而是通过Q网络来计算的
算法參考:https://www.cnblogs.com/pinard/p/9714655.html
"""

import random
from collections import deque

import gym
import numpy as np
from keras.layers import Dense
from keras.models import Sequential
from keras.optimizers import Adam
import matplotlib.pyplot as plt

#用keras重写
class DQN_K(object):
    # DQN Agent
    def __init__(self, n_actions, n_features, learning_rate=0.0001,reward_decay=0.9,e_greedy=0.5,memory_size=10000,batch_size=32):
        """
        :param n_actions: int，动作的数量
        :param n_features: int，状态的数量
        """
        self.n_actions = n_actions
        self.n_features = n_features
        self.lr = learning_rate
        self.gamma = reward_decay
        self.epsilon = e_greedy
        self.memory_size = memory_size
        self.batch_size = batch_size
        # init experience replay
        self.replay_buffer = deque()
        self.target_model = self.create_Q_network()

    def create_Q_network(self):
        model = Sequential()
        model.add(Dense(32, activation='relu', input_dim=self.n_features))
        model.add(Dense(self.n_actions))
        optimizer = Adam(lr=self.lr)
        model.compile(loss='mean_squared_error', optimizer=optimizer, metrics=['accuracy'])
        print('Q_network')
        print(model.summary())
        return model

    def perceive(self,state,action,reward,next_state,done):
        self.replay_buffer.append((state,action,reward,next_state,done))
        if len(self.replay_buffer) > self.memory_size:
          self.replay_buffer.popleft()

        if len(self.replay_buffer) > self.batch_size:
          self.train_Q_network()

    def train_Q_network(self):
        # Step 1: obtain random minibatch from replay memory
        minibatch = np.array(random.sample(self.replay_buffer, self.batch_size))
        state_batch = np.array([data for data in minibatch[:,0]])
        action_batch = np.array([data for data in minibatch[:,1]])
        reward_batch = np.array([data for data in minibatch[:,2]])
        next_state_batch = np.array([data for data in minibatch[:,3]])

        # Step 2: calculate y
        q_target = self.target_model.predict(state_batch)
        q_value_batch = self.target_model.predict(next_state_batch)
        done_batch=np.array([data for data in minibatch[:,4]])

        re_lst=reward_batch.copy()
        notdone_index = list(np.argwhere(done_batch == False).ravel())
        re_lst[notdone_index] += self.gamma * np.max(q_value_batch[notdone_index], axis=1)
        for i in range(self.batch_size):
            q_target[i, action_batch[i]] = re_lst[i]
        self.target_model.fit(state_batch, q_target, epochs=10, verbose=0)

    def egreedy_action(self,state):
        """
        贪心策略+随机策略,根据状态选择行为
        :param state: ndarray,shape是(self.n_features,)
        :return:int,选择出来的行为
        """
        observation = state[np.newaxis, :]
        if np.random.uniform() < self.epsilon:
          actions_value = self.target_model.predict(observation)
          action = np.argmax(actions_value)
        else:
          action = np.random.randint(0, self.n_actions)
        return action

    def action(self, state):
        """
        贪心策略根据状态选择行为
        :param state: ndarray,shape是(self.n_features,)
        :return:int,选择出来的行为
        """
        observation = state[np.newaxis, :]
        actions_value = self.target_model.predict(observation)
        action = np.argmax(actions_value)
        return action

def main():
    # initialize OpenAI Gym env and dqn agent
    env = gym.make(ENV_NAME)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    agent = DQN_K(action_dim, state_dim)

    reward_lst=[]
    for episode in range(EPISODE):
        # initialize task
        state = env.reset()
        # Train
        for step in range(STEP):
            action = agent.egreedy_action(state)  # e-greedy action for train
            next_state, reward, done, _ = env.step(action)  # 向前走一步
            # Define reward for agent
            reward = -1 if done else 0.1
            agent.perceive(state, action, reward, next_state, done)  # 存储记忆
            state = next_state
            if done:
                break
        # Test every 100 episodes
        if episode % TEST == 0:
            total_reward = 0
            for i in range(TEST):
                state = env.reset()
                for j in range(STEP):
                    env.render()  # 用于渲染出当前的智能体以及环境的状态
                    action = agent.action(state)  # direct action for test
                    state, reward, done, _ = env.step(action)
                    total_reward += reward
                    if done:
                        break
            ave_reward = total_reward / TEST
            reward_lst.append(ave_reward)
            print('episode: ', episode, 'Evaluation Average Reward:', ave_reward)

    plt.figure()
    x=range(0,EPISODE,TEST)
    plt.plot(x,reward_lst)
    plt.xticks(x)
    plt.xlabel('Epoch')
    plt.ylabel('Reward')
    plt.savefig('images/dqn.jpg')

if __name__ == '__main__':
    # ---------------------------------------------------------
    # Hyper Parameters
    ENV_NAME = 'CartPole-v0'
    EPISODE = 300  # Episode limitation
    STEP = 300  # Step limitation in an episode
    TEST = 10  # The number of experiment test every 100 episode
    main()