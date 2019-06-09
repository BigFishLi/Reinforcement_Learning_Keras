#! /usr/bin/env python
#-*- coding:utf-8 -*-
"""
@time: 2019/06/01
"""

"""
Nature DQN,ICML 2016
Nature DQN使用了两个Q网络，一个当前Q网络Q用来选择动作，更新模型参数，另一个目标Q网络Q′用于计算目标Q值。
目标Q网络的网络参数不需要迭代更新，而是每隔一段时间从当前Q网络Q复制过来，即延时更新，这样可以减少目标Q值和当前的Q值相关性.
两个Q网络的结构是一模一样的。这样才可以复制网络参数。
Nature DQN和DQN相比，除了用一个新的相同结构的目标Q网络来计算目标Q值以外，其余部分基本是完全相同的
算法參考:https://www.cnblogs.com/pinard/p/9756075.html
"""
import random

import numpy as np
import matplotlib.pyplot as plt
import gym

from dqn import DQN_K


class NDQN_K(DQN_K):
    def __init__(self, n_actions, n_features, learning_rate=0.0001, reward_decay=0.9, e_greedy=0.5, memory_size=10000,
                 batch_size=32,replace_target_iter=100):
        """
        :param n_actions: int，动作的数量
        :param n_features: int，状态的数量
        :param replace_target_iter:int,set target_model.weights=eval_model.weights的批次
        """
        DQN_K.__init__(self,n_actions=n_actions, n_features=n_features,learning_rate=learning_rate,reward_decay=reward_decay,e_greedy=e_greedy,memory_size=memory_size,batch_size=batch_size)
        self.replace_target_iter=replace_target_iter
        self.eval_model=self.create_Q_network()
        self.learn_step_counter = 0

    def target_replace_op(self):
        v1 = self.eval_model.get_weights()
        self.target_model.set_weights(v1)

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
        q_eval = self.eval_model.predict(state_batch)
        q_target = q_eval.copy()
        q_value_batch = self.target_model.predict(next_state_batch)
        done_batch = np.array([data for data in minibatch[:, 4]])

        re_lst = reward_batch.copy()
        notdone_index = list(np.argwhere(done_batch == False).ravel())
        re_lst[notdone_index] += self.gamma * np.max(q_value_batch[notdone_index], axis=1)
        for i in range(self.batch_size):
            q_target[i, action_batch[i]] = re_lst[i]

        self.eval_model.fit(state_batch, q_target, epochs=10, verbose=0)

        self.learn_step_counter+=1

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
    plt.savefig('images/ndqn.jpg')

if __name__ == '__main__':
    # ---------------------------------------------------------
    # Hyper Parameters
    ENV_NAME = 'CartPole-v0'
    EPISODE = 300  # Episode limitation
    STEP = 300  # Step limitation in an episode
    TEST = 10  # The number of experiment test every 100 episode
    main()