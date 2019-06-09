#! /usr/bin/env python
#-*- coding:utf-8 -*-
"""
@time: 2019/06/08
"""
"""
Double DQN (DDQN),ICML 2016
和Nature DQN一样,通过解耦目标Q值动作的选择和目标Q值的计算这两步，来消除过度估计的问题

算法參考:https://www.cnblogs.com/pinard/p/9778063.html
"""
import random

import numpy as np
import matplotlib.pyplot as plt
import gym

from ndqn import NDQN_K

class DDQN_K(NDQN_K):
    def __init__(self, n_actions, n_features, learning_rate=0.0001, reward_decay=0.9, e_greedy=0.5, memory_size=10000,
                 batch_size=32,replace_target_iter=100):
        NDQN_K.__init__(self, n_actions, n_features, learning_rate, reward_decay, e_greedy, memory_size, batch_size,replace_target_iter)

    def train_Q_network(self):
        if self.learn_step_counter % self.replace_target_iter == 0:
            self.target_replace_op()

        # Step 1: obtain random minibatch from replay memory
        minibatch = np.array(random.sample(self.replay_buffer, self.batch_size))
        state_batch = np.array([data for data in minibatch[:,0]])
        action_batch = np.array([data for data in minibatch[:,1]])
        reward_batch = np.array([data for data in minibatch[:,2]])
        next_state_batch = np.array([data for data in minibatch[:,3]])

        # Step 2: calculate y
        q_eval =self.eval_model.predict(state_batch)
        q_target = q_eval.copy()
        max_action_next = np.argmax(q_eval,axis=1)
        target_q_batch=self.target_model.predict(next_state_batch)

        re_lst = reward_batch.copy()
        for i in range(self.batch_size):
            done = minibatch[i][4]
            if not done:
                re_lst[i] += self.gamma * target_q_batch[i, max_action_next[i]]
            q_target[i, action_batch[i]] = re_lst[i]
        self.eval_model.fit(state_batch, q_target, epochs=10, verbose=0)
        self.learn_step_counter+=1



def main():
    # initialize OpenAI Gym env and dqn agent
    env = gym.make(ENV_NAME)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    agent = DDQN_K(action_dim,state_dim,replace_target_iter=TEST)

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
    plt.savefig('images/ddqn.jpg')

if __name__ == '__main__':
    # ---------------------------------------------------------
    # Hyper Parameters
    ENV_NAME = 'CartPole-v0'
    EPISODE = 300  # Episode limitation
    STEP = 300  # Step limitation in an episode
    TEST = 10  # The number of experiment test every 100 episode
    main()