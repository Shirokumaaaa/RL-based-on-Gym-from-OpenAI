import math
import gym
import numpy as np
from collections import defaultdict
import time
import pickle
import matplotlib.pyplot as plt

class Agent:
    def __init__(self, n_actions, lr=0.1, gamma=0.9):
        self.epsilon_start = 0.95
        self.epsilon_end = 0.01
        self.epsilon_decay = 200
        self.epsilon = self.epsilon_start
        self.Q_table = defaultdict(lambda: np.zeros(n_actions))
        self.lr = lr
        self.gamma = gamma
        self.sample_count = 0
        self.n_actions = n_actions

    # Based on the On-Policy Strategy
    #################################
    def update(self, state, action, reward, next_state, next_action, terminated):
        Q_predict = self.Q_table[str(state)][action]
        if terminated:
            Q_target = reward
        else:
            Q_target = reward + self.gamma * self.Q_table[str(next_state)][next_action]
        self.Q_table[str(state)][action] += self.lr * (Q_target - Q_predict)
    #################################

    def predict_action(self, state):
        action = np.argmax(self.Q_table[str(state)])
        return action

    def sample_action(self, state):
        self.sample_count += 1
        self.epsilon = self.epsilon_end + (self.epsilon_start - self.epsilon_end) * math.exp(-self.sample_count / self.epsilon_decay)
        if np.random.uniform(0, 1) > self.epsilon:
            action = np.argmax(self.Q_table[str(state)])
        else:
            action = np.random.choice(self.n_actions)
        return action

# 示例环境和Agent的初始化
env = gym.make('CliffWalking-v0', render_mode="human")

# 加载Q-table
with open('q_table.pkl', 'rb') as f:
    q_table = pickle.load(f)

n_states = env.observation_space.n
n_actions = env.action_space.n
agent = Agent(n_actions)
agent.Q_table = q_table

print(f"状态数：{n_states}， 动作数：{n_actions}")

train_eps = 11
steps_per_episode = []  # 步数统计应该在所有回合外部初始化

for i_ep in range(train_eps):  # 遍历每个回合
    state = env.reset()  # 重置环境,即开始新的回合
    env.render()
    steps = 0  # 初始化步数计数器
    action = agent.sample_action(state)  # 根据算法采样一个初始动作
    while True:
        try:
            result = env.step(action)
            next_state, reward, terminated, _ = result[:4]  # 与环境进行一次交互
            print(f"Step: Action={action}, Reward={reward}, Terminated={terminated}")
            next_action = agent.sample_action(next_state) if not terminated else None  # 如果回合未结束，则采样下一个动作
            agent.update(state, action, reward, next_state, next_action, terminated)  # 更新策略
            state, action = next_state, next_action  # 更新状态和动作
            if terminated:
                break
            #time.sleep(3)  # 减慢执行速度以便观察
            steps += 1  # 每执行一步，步数计数器加1
        except Exception as e:
            print(f"Error: {e}")
            break
    steps_per_episode.append(steps)  # 将这一次运行的步数添加到列表中

# 训练循环结束后
with open('q_table2.pkl', 'wb') as f:
    pickle.dump(dict(agent.Q_table), f)

env.close()

# 绘制折线图
plt.plot(steps_per_episode)
plt.title('Steps per Episode')
plt.xlabel('Episode')
plt.ylabel('Steps')
plt.show()