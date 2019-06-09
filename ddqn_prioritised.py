#! /usr/bin/env python
#-*- coding:utf-8 -*-
"""
@time: 2019/06/08
"""
"""
Prioritized Replay DDQN,ICML 2016
在DDQN的基础上，对经验回放部分的逻辑做优化,设定每次经验回放的优先级
算法參考:https://www.cnblogs.com/pinard/p/9797695.html
"""
import numpy as np
import matplotlib.pyplot as plt
import gym

from ddqn import DDQN_K

class SumTree(object):
    """
    This SumTree code is a modified version and the original code is from:
    https://github.com/jaara/AI-blog/blob/master/SumTree.py
    Story data with its priority in the tree.
    """
    data_pointer = 0
    def __init__(self, capacity):
        """
        :param capacity: 叶子节点的数量，记忆中样本数量。叶子节点除了保存数据data以外，还要保存该样本的优先级p
        """
        self.capacity = capacity  # for all priority values
        self.tree = np.zeros(2 * capacity - 1) #二叉树所有的树节点
        self.data = np.zeros(capacity, dtype=object)  # for all transitions

    def add(self, p, data):
        """
        Tree增加新叶子节点
        :param p: 优先级p
        :param data:叶子结点data
        :return:
        """
        tree_idx = self.data_pointer + self.capacity - 1
        self.data[self.data_pointer] = data  #更新叶子节点的data
        self.update(tree_idx, p)  #更新树上相关节点的p，叶子节点和所有根节点都需要更新

        self.data_pointer += 1
        if self.data_pointer >= self.capacity:  # replace when exceed the capacity
            self.data_pointer = 0

    def update(self, tree_idx, p):
        """
        依次更新叶子节点和树上所有根节点的p
        :param tree_idx: 索引
        :param p: 优先级p
        :return:
        """
        change = p - self.tree[tree_idx]
        self.tree[tree_idx] = p
        # then propagate the change through tree
        while tree_idx != 0:    # this method is faster than the recursive loop in the reference code
            tree_idx = (tree_idx - 1) // 2
            self.tree[tree_idx] += change

    def get_leaf(self, v):
        """
         Tree structure and array storage:
        Tree index:
             0         -> storing priority sum
            / \
          1     2
         / \   / \
        3   4 5   6    -> storing priority for transitions
        Array type for storing:
        [0,1,2,3,4,5,6]
        :param v:
        :return:
        """
        parent_idx = 0
        while True:  # the while loop is faster than the method in the reference code
            cl_idx = 2 * parent_idx + 1  # this leaf's left and right kids
            cr_idx = cl_idx + 1
            if cl_idx >= len(self.tree):  # reach bottom, end search
                leaf_idx = parent_idx
                break
            else:  # downward search, always search for a higher priority node
                if v <= self.tree[cl_idx]:
                    parent_idx = cl_idx
                else:
                    v -= self.tree[cl_idx]
                    parent_idx = cr_idx

        data_idx = leaf_idx - self.capacity + 1
        return leaf_idx, self.tree[leaf_idx], self.data[data_idx]

    @property
    def total_p(self):
        return self.tree[0]  # the root

class Memory(object):  # stored as ( s, a, r, s_ ) in SumTree
    """
    This Memory class is modified based on the original code from:
    https://github.com/jaara/AI-blog/blob/master/Seaquest-DDQN-PER.py
    """
    epsilon = 0.01  # small amount to avoid zero priority
    alpha = 0.6  # [0~1] convert the importance of TD error to priority
    beta = 0.4  # importance-sampling, from initial value increasing to 1
    beta_increment_per_sampling = 0.001
    abs_err_upper = 1.  # clipped abs error

    def __init__(self, capacity):
        """
        :param capacity: 记忆中样本数量
        """
        self.tree = SumTree(capacity)

    def store(self, transition):
        max_p = np.max(self.tree.tree[-self.tree.capacity:])
        if max_p == 0:
            max_p = self.abs_err_upper
        self.tree.add(max_p, transition)   # set the max p for new p

    def sample(self, n):
        b_idx= np.empty((n,), dtype=np.int32)
        b_memory = np.empty((n, len(self.tree.data[0])),dtype=object)
        ISWeights=np.empty((n, 1))
        pri_seg = self.tree.total_p / n       # priority segment
        self.beta = np.min([1., self.beta + self.beta_increment_per_sampling])  # max = 1

        min_prob = np.min(self.tree.tree[-self.tree.capacity:]) / self.tree.total_p     # for later calculate ISweight
        if min_prob == 0:
            min_prob = 0.00001
        for i in range(n):
            a, b = pri_seg * i, pri_seg * (i + 1)
            v = np.random.uniform(a, b)
            idx, p, data = self.tree.get_leaf(v)
            prob = p / self.tree.total_p
            ISWeights[i, 0] = np.power(prob/min_prob, -self.beta)
            b_idx[i] = idx
            b_memory[i, :]=data
        return b_idx, b_memory, ISWeights

    def batch_update(self, tree_idx, abs_errors):
        abs_errors += self.epsilon  # convert to abs and avoid 0
        clipped_errors = np.minimum(abs_errors, self.abs_err_upper)
        ps = np.power(clipped_errors, self.alpha)
        for ti, p in zip(tree_idx, ps):
            self.tree.update(ti, p)

class DDQN_prio(DDQN_K):
    def __init__(self, n_actions, n_features, learning_rate=0.0001, reward_decay=0.9, e_greedy=0.5, memory_size=10000,
                 batch_size=32,replace_target_iter=100):
        DDQN_K.__init__(self, n_actions, n_features, learning_rate, reward_decay, e_greedy, memory_size, batch_size,replace_target_iter)
        self.memory = Memory(capacity=memory_size)
        self.replay_total = 0

    def store_transition(self, s, a, r, s_, done):
        transition = [s, a, r, s_, done]
        self.memory.store(transition)  # have high priority for newly arrived transition

    def perceive(self,state,action,reward,next_state,done):
        self.store_transition(state, action, reward, next_state, done)

        self.replay_total += 1
        if self.replay_total > self.batch_size:
            self.train_Q_network()

    def train_Q_network(self):
        if self.learn_step_counter % self.replace_target_iter == 0:
            self.target_replace_op()

        tree_idx, minibatch, ISWeights = self.memory.sample(self.batch_size)
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

        #his=self.eval_model.fit(state_batch, q_target, epochs=10, verbose=0,sample_weight=ISWeights.ravel())
        weight=ISWeights.ravel()
        his = self.eval_model.fit(state_batch, q_target, epochs=10, verbose=0, sample_weight=weight)
        abs_errors=np.array(his.history['loss'])
        self.memory.batch_update(tree_idx, abs_errors)
        self.learn_step_counter += 1

def main():
    # initialize OpenAI Gym env and dqn agent
    env = gym.make(ENV_NAME)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    agent = DDQN_prio(action_dim,state_dim,replace_target_iter=TEST)

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
    plt.savefig('images/ddqn_prior.jpg')

if __name__ == '__main__':
    # ---------------------------------------------------------
    # Hyper Parameters
    ENV_NAME = 'CartPole-v0'
    EPISODE = 300  # Episode limitation
    STEP = 300  # Step limitation in an episode
    TEST = 10  # The number of experiment test every 100 episode
    main()