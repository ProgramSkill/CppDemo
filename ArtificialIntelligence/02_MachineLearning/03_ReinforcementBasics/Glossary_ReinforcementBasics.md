# Reinforcement Learning Glossary (强化学习词汇表)

## English-Chinese Technical Terms

---

## 1. 基础概念 (Fundamental Concepts)

| English | 中文 | 说明 |
|---------|------|------|
| **Reinforcement Learning** | 强化学习 | 通过奖励学习的ML方法 |
| **Agent** | 智能体 | 学习和决策的主体 |
| **Environment** | 环境 | 智能体交互的对象 |
| **State** | 状态 | 环境的当前情况 |
| **Action** | 动作 | 智能体的决策 |
| **Reward** | 奖励 | 环境的反馈信号 |
| **Policy** | 策略 | 状态到动作的映射 |
| **Episode** | 回合 | 从开始到终止的序列 |
| **Trajectory** | 轨迹 | 状态-动作-奖励序列 |

---

## 2. MDP相关 (MDP Related)

| English | 中文 | 说明 |
|---------|------|------|
| **Markov Decision Process** | 马尔可夫决策过程 | RL的数学框架 |
| **MDP** | MDP | 马尔可夫决策过程缩写 |
| **Markov Property** | 马尔可夫性质 | 未来只依赖当前状态 |
| **Transition Probability** | 转移概率 | P(s'|s,a) |
| **Reward Function** | 奖励函数 | R(s,a,s') |
| **Discount Factor** | 折扣因子 | γ, 未来奖励的权重 |
| **Return** | 回报 | 累积折扣奖励 |
| **Terminal State** | 终止状态 | 回合结束的状态 |

---

## 3. 价值函数 (Value Functions)

| English | 中文 | 说明 |
|---------|------|------|
| **Value Function** | 价值函数 | 状态的长期价值 |
| **State-Value Function** | 状态价值函数 | V(s) |
| **Action-Value Function** | 动作价值函数 | Q(s,a) |
| **Q-Function** | Q函数 | 动作价值函数 |
| **Optimal Value Function** | 最优价值函数 | V*, Q* |
| **Bellman Equation** | 贝尔曼方程 | 价值函数的递归关系 |
| **Bellman Optimality** | 贝尔曼最优性 | 最优价值的条件 |
| **Advantage Function** | 优势函数 | A(s,a) = Q(s,a) - V(s) |

---

## 4. 算法类型 (Algorithm Types)

| English | 中文 | 说明 |
|---------|------|------|
| **Model-Based** | 基于模型 | 学习环境模型 |
| **Model-Free** | 无模型 | 直接学习策略或价值 |
| **On-Policy** | 同策略 | 评估和改进同一策略 |
| **Off-Policy** | 异策略 | 评估和改进不同策略 |
| **Value-Based** | 基于价值 | 学习价值函数 |
| **Policy-Based** | 基于策略 | 直接学习策略 |
| **Actor-Critic** | 演员-评论家 | 结合价值和策略 |

---

## 5. 经典算法 (Classic Algorithms)

| English | 中文 | 说明 |
|---------|------|------|
| **Dynamic Programming** | 动态规划 | 基于模型的规划 |
| **Policy Evaluation** | 策略评估 | 计算V^π |
| **Policy Improvement** | 策略改进 | 改进策略 |
| **Policy Iteration** | 策略迭代 | 评估+改进循环 |
| **Value Iteration** | 价值迭代 | 直接计算V* |
| **Monte Carlo** | 蒙特卡洛 | 基于采样的方法 |
| **Temporal Difference** | 时序差分 | TD学习 |
| **Q-Learning** | Q学习 | 异策略TD控制 |
| **SARSA** | SARSA | 同策略TD控制 |

---

## 6. 探索与利用 (Exploration & Exploitation)

| English | 中文 | 说明 |
|---------|------|------|
| **Exploration** | 探索 | 尝试新动作 |
| **Exploitation** | 利用 | 选择已知最优 |
| **Exploration-Exploitation Tradeoff** | 探索-利用权衡 | 平衡探索和利用 |
| **ε-Greedy** | ε贪婪 | 以ε概率随机探索 |
| **Softmax** | Softmax | 基于概率的探索 |
| **UCB** | 置信上界 | Upper Confidence Bound |
| **Thompson Sampling** | 汤普森采样 | 贝叶斯探索方法 |

---

## 7. 深度强化学习 (Deep RL)

| English | 中文 | 说明 |
|---------|------|------|
| **Deep Q-Network** | 深度Q网络 | 神经网络Q函数 |
| **DQN** | DQN | Deep Q-Network缩写 |
| **Experience Replay** | 经验回放 | 存储和重用经验 |
| **Target Network** | 目标网络 | 稳定训练的网络 |
| **Double DQN** | 双重DQN | 减少过估计 |
| **Dueling DQN** | 对决DQN | 分离V和A |
| **Policy Gradient** | 策略梯度 | 直接优化策略 |
| **REINFORCE** | REINFORCE | 基础策略梯度算法 |
| **A3C** | A3C | 异步优势演员评论家 |
| **PPO** | PPO | 近端策略优化 |
| **SAC** | SAC | 软演员评论家 |

---

## 常用缩写 (Common Abbreviations)

| 缩写 | 英文全称 | 中文 |
|------|----------|------|
| **RL** | Reinforcement Learning | 强化学习 |
| **MDP** | Markov Decision Process | 马尔可夫决策过程 |
| **DP** | Dynamic Programming | 动态规划 |
| **MC** | Monte Carlo | 蒙特卡洛 |
| **TD** | Temporal Difference | 时序差分 |
| **DQN** | Deep Q-Network | 深度Q网络 |
| **PPO** | Proximal Policy Optimization | 近端策略优化 |
| **A3C** | Asynchronous Advantage Actor-Critic | 异步优势演员评论家 |
| **SAC** | Soft Actor-Critic | 软演员评论家 |

---

**最后更新**: 2024-01-29
