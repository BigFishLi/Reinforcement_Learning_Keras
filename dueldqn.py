#! /usr/bin/env python
#-*- coding:utf-8 -*-
"""
@time: 2019/06/09
"""

"""
Dueling DQN,ICML 2016
在NativeDQN的基础上，修改了神经网络的结构
算法参考：https://www.cnblogs.com/pinard/p/9923859.html
"""
from keras.models import Sequential,Model
from keras.layers import Dense,Input
from keras import backend as K
from keras.optimizers import Adam
import matplotlib.pyplot as plt
import gym

from ndqn import NDQN_K

class DuelingDQN(NDQN_K):
    def __init__(self, n_actions, n_features, learning_rate=0.0001, reward_decay=0.9, e_greedy=0.5, memory_size=10000,
                 batch_size=32,replace_target_iter=100):
        NDQN_K.__init__(self, n_actions, n_features, learning_rate, reward_decay, e_greedy, memory_size, batch_size,replace_target_iter)

    def create_Q_network(self):
        input=Input(shape=(self.n_features,))
        x=Dense(32, activation='relu')(input)
        value=Dense(1, activation='relu')(x) #Value
        advantage=Dense(self.n_actions, activation='relu')(x) #Action
        out= value + (advantage -K.mean(advantage, axis=1, keepdims=True))
        model=Model(inputs=[input],outputs=[out])
        optimizer = Adam(lr=self.lr)
        model.compile(loss='mean_squared_error', optimizer=optimizer, metrics=['accuracy'])
        print('Q_network')
        print(model.summary())
        return model

def main():
    # initialize OpenAI Gym env and dqn agent
    env = gym.make(ENV_NAME)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    agent = NDQN_K(action_dim,state_dim,replace_target_iter=TEST)

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
    plt.xlabel('Epoch')
    plt.ylabel('Reward')
    plt.savefig('images/duel_dqn.jpg')

if __name__ == '__main__':
    # ---------------------------------------------------------
    # Hyper Parameters
    ENV_NAME = 'CartPole-v0'
    EPISODE = 300  # Episode limitation
    STEP = 300  # Step limitation in an episode
    TEST = 10  # The number of experiment test every 100 episode
    main()