# Reinforcement Learning Basics: From Beginner to Expert

## 📚 Table of Contents

- [Introduction](#introduction)
- [Part I: Beginner Level](#part-i-beginner-level)
  - [Chapter 1: What is Reinforcement Learning?](#chapter-1-what-is-reinforcement-learning)
  - [Chapter 2: Markov Decision Processes](#chapter-2-markov-decision-processes)
  - [Chapter 3: Bellman Equations](#chapter-3-bellman-equations)
- [Part II: Intermediate Level](#part-ii-intermediate-level)
  - [Chapter 4: Dynamic Programming](#chapter-4-dynamic-programming)
  - [Chapter 5: Monte Carlo Methods](#chapter-5-monte-carlo-methods)
  - [Chapter 6: Temporal Difference Learning](#chapter-6-temporal-difference-learning)
- [Part III: Advanced Level](#part-iii-advanced-level)
  - [Chapter 7: Q-Learning](#chapter-7-q-learning)
  - [Chapter 8: Policy Gradient Methods](#chapter-8-policy-gradient-methods)
  - [Chapter 9: Applications](#chapter-9-applications)

---

## Introduction

**Reinforcement Learning (RL)** is a type of machine learning where an agent learns to make decisions by interacting with an environment to maximize cumulative reward.

### Key Differences from Other ML Paradigms

| Aspect | Supervised | Unsupervised | Reinforcement |
|--------|------------|--------------|---------------|
| **Feedback** | Correct labels | None | Reward signals |
| **Goal** | Predict | Find patterns | Maximize reward |
| **Data** | Static dataset | Static dataset | Generated through interaction |

### What You'll Learn

| Level | Topics | Skills |
|-------|--------|--------|
| **Beginner** | MDP, Bellman equations | Understand RL framework |
| **Intermediate** | DP, MC, TD methods | Implement basic algorithms |
| **Advanced** | Q-Learning, Policy Gradient | Solve complex RL problems |

---

## Part I: Beginner Level

### Chapter 1: What is Reinforcement Learning?

#### 1.1 The RL Framework

```
        Action (a)
    Agent ───────────► Environment
      ▲                    │
      │                    │
      │   State (s)        │
      │   Reward (r)       │
      └────────────────────┘
```

**Components**:
- **Agent**: The learner and decision maker
- **Environment**: What the agent interacts with
- **State (s)**: Current situation
- **Action (a)**: Decision made by agent
- **Reward (r)**: Feedback from environment
- **Policy (π)**: Strategy mapping states to actions

#### 1.2 Examples

| Domain | Agent | Environment | Actions | Reward |
|--------|-------|-------------|---------|--------|
| Game | Player | Game | Move, Jump | Score |
| Robot | Robot | Physical world | Motor commands | Task completion |
| Trading | Algorithm | Market | Buy, Sell, Hold | Profit |

#### 1.3 Key Concepts

**Episode**: A sequence from initial state to terminal state

**Return**: Cumulative future reward
```
Gₜ = Rₜ₊₁ + γRₜ₊₂ + γ²Rₜ₊₃ + ... = Σₖ γᵏRₜ₊ₖ₊₁
```

**Discount Factor (γ)**: 0 ≤ γ ≤ 1
- γ = 0: Only immediate reward matters
- γ = 1: All future rewards equally important
- Typical: γ = 0.99

---

### Chapter 2: Markov Decision Processes

#### 2.1 MDP Definition

An MDP is defined by tuple (S, A, P, R, γ):
- **S**: Set of states
- **A**: Set of actions
- **P**: Transition probability P(s'|s,a)
- **R**: Reward function R(s,a,s')
- **γ**: Discount factor

#### 2.2 Markov Property

The future depends only on the current state, not history:
```
P(Sₜ₊₁|Sₜ) = P(Sₜ₊₁|S₁, S₂, ..., Sₜ)
```

#### 2.3 Example: GridWorld

```python
import numpy as np

class GridWorld:
    def __init__(self, size=4):
        self.size = size
        self.state = [0, 0]  # Start position
        self.goal = [size-1, size-1]  # Goal position
        
    def reset(self):
        self.state = [0, 0]
        return tuple(self.state)
    
    def step(self, action):
        # Actions: 0=up, 1=down, 2=left, 3=right
        moves = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        
        new_state = [
            max(0, min(self.size-1, self.state[0] + moves[action][0])),
            max(0, min(self.size-1, self.state[1] + moves[action][1]))
        ]
        self.state = new_state
        
        # Reward
        if self.state == self.goal:
            return tuple(self.state), 1.0, True  # state, reward, done
        return tuple(self.state), -0.01, False

# Usage
env = GridWorld()
state = env.reset()
done = False
total_reward = 0

while not done:
    action = np.random.randint(4)  # Random policy
    state, reward, done = env.step(action)
    total_reward += reward

print(f"Total reward: {total_reward}")
```

---

### Chapter 3: Bellman Equations

#### 3.1 Value Functions

**State-Value Function V(s)**: Expected return starting from state s
```
V^π(s) = E_π[Gₜ | Sₜ = s]
       = E_π[Rₜ₊₁ + γV^π(Sₜ₊₁) | Sₜ = s]
```

**Action-Value Function Q(s,a)**: Expected return taking action a in state s
```
Q^π(s,a) = E_π[Gₜ | Sₜ = s, Aₜ = a]
         = E_π[Rₜ₊₁ + γQ^π(Sₜ₊₁, Aₜ₊₁) | Sₜ = s, Aₜ = a]
```

#### 3.2 Bellman Expectation Equation

For policy π:
```
V^π(s) = Σₐ π(a|s) × Σₛ' P(s'|s,a) × [R(s,a,s') + γV^π(s')]
```

#### 3.3 Bellman Optimality Equation

For optimal policy:
```
V*(s) = maxₐ Σₛ' P(s'|s,a) × [R(s,a,s') + γV*(s')]

Q*(s,a) = Σₛ' P(s'|s,a) × [R(s,a,s') + γ maxₐ' Q*(s',a')]
```

---

## Part II: Intermediate Level

### Chapter 4: Dynamic Programming

#### 4.1 Policy Evaluation

Compute V^π for a given policy:

```python
def policy_evaluation(env, policy, gamma=0.99, theta=1e-6):
    V = np.zeros(env.n_states)
    
    while True:
        delta = 0
        for s in range(env.n_states):
            v = V[s]
            # Bellman expectation equation
            V[s] = sum(
                policy[s, a] * sum(
                    env.P[s, a, s_prime] * (env.R[s, a, s_prime] + gamma * V[s_prime])
                    for s_prime in range(env.n_states)
                )
                for a in range(env.n_actions)
            )
            delta = max(delta, abs(v - V[s]))
        
        if delta < theta:
            break
    
    return V
```

#### 4.2 Policy Iteration

Alternate between evaluation and improvement:

```python
def policy_iteration(env, gamma=0.99):
    # Initialize random policy
    policy = np.ones((env.n_states, env.n_actions)) / env.n_actions
    
    while True:
        # Policy Evaluation
        V = policy_evaluation(env, policy, gamma)
        
        # Policy Improvement
        policy_stable = True
        for s in range(env.n_states):
            old_action = np.argmax(policy[s])
            
            # Compute Q-values
            q_values = np.zeros(env.n_actions)
            for a in range(env.n_actions):
                for s_prime in range(env.n_states):
                    q_values[a] += env.P[s, a, s_prime] * (
                        env.R[s, a, s_prime] + gamma * V[s_prime]
                    )
            
            # Greedy policy
            best_action = np.argmax(q_values)
            policy[s] = np.eye(env.n_actions)[best_action]
            
            if old_action != best_action:
                policy_stable = False
        
        if policy_stable:
            break
    
    return policy, V
```

#### 4.3 Value Iteration

Combine evaluation and improvement:

```python
def value_iteration(env, gamma=0.99, theta=1e-6):
    V = np.zeros(env.n_states)
    
    while True:
        delta = 0
        for s in range(env.n_states):
            v = V[s]
            # Bellman optimality equation
            V[s] = max(
                sum(
                    env.P[s, a, s_prime] * (env.R[s, a, s_prime] + gamma * V[s_prime])
                    for s_prime in range(env.n_states)
                )
                for a in range(env.n_actions)
            )
            delta = max(delta, abs(v - V[s]))
        
        if delta < theta:
            break
    
    # Extract policy
    policy = np.zeros((env.n_states, env.n_actions))
    for s in range(env.n_states):
        q_values = [
            sum(env.P[s, a, s_prime] * (env.R[s, a, s_prime] + gamma * V[s_prime])
                for s_prime in range(env.n_states))
            for a in range(env.n_actions)
        ]
        policy[s, np.argmax(q_values)] = 1
    
    return policy, V
```

---

### Chapter 5: Monte Carlo Methods

#### 5.1 First-Visit MC Prediction

```python
def monte_carlo_prediction(env, policy, n_episodes=10000, gamma=0.99):
    V = {}
    returns = {}
    
    for _ in range(n_episodes):
        episode = generate_episode(env, policy)
        G = 0
        visited = set()
        
        # Process episode backwards
        for t in range(len(episode) - 1, -1, -1):
            state, action, reward = episode[t]
            G = gamma * G + reward
            
            if state not in visited:  # First-visit
                visited.add(state)
                if state not in returns:
                    returns[state] = []
                returns[state].append(G)
                V[state] = np.mean(returns[state])
    
    return V
```

#### 5.2 MC Control with ε-Greedy

```python
def mc_control_epsilon_greedy(env, n_episodes=10000, gamma=0.99, epsilon=0.1):
    Q = {}
    returns = {}
    
    for state in env.all_states:
        for action in env.all_actions:
            Q[(state, action)] = 0
            returns[(state, action)] = []
    
    for _ in range(n_episodes):
        # Generate episode with ε-greedy policy
        episode = generate_episode_epsilon_greedy(env, Q, epsilon)
        G = 0
        visited = set()
        
        for t in range(len(episode) - 1, -1, -1):
            state, action, reward = episode[t]
            G = gamma * G + reward
            
            if (state, action) not in visited:
                visited.add((state, action))
                returns[(state, action)].append(G)
                Q[(state, action)] = np.mean(returns[(state, action)])
    
    # Extract greedy policy
    policy = {}
    for state in env.all_states:
        q_values = [Q[(state, a)] for a in env.all_actions]
        policy[state] = env.all_actions[np.argmax(q_values)]
    
    return policy, Q
```

---

### Chapter 6: Temporal Difference Learning

#### 6.1 TD(0) Prediction

Update after every step, not after episode:

```python
def td_prediction(env, policy, n_episodes=1000, alpha=0.1, gamma=0.99):
    V = {s: 0 for s in env.all_states}
    
    for _ in range(n_episodes):
        state = env.reset()
        done = False
        
        while not done:
            action = policy[state]
            next_state, reward, done = env.step(action)
            
            # TD update
            V[state] += alpha * (reward + gamma * V[next_state] - V[state])
            state = next_state
    
    return V
```

**TD vs MC**:
| Aspect | TD | MC |
|--------|----|----|
| Updates | Every step | End of episode |
| Bias | Biased | Unbiased |
| Variance | Lower | Higher |
| Bootstrap | Yes | No |

---

## Part III: Advanced Level

### Chapter 7: Q-Learning

#### 7.1 Algorithm

Off-policy TD control:

```python
def q_learning(env, n_episodes=10000, alpha=0.1, gamma=0.99, epsilon=0.1):
    Q = {}
    for state in env.all_states:
        for action in env.all_actions:
            Q[(state, action)] = 0
    
    for episode in range(n_episodes):
        state = env.reset()
        done = False
        
        while not done:
            # ε-greedy action selection
            if np.random.random() < epsilon:
                action = np.random.choice(env.all_actions)
            else:
                q_values = [Q[(state, a)] for a in env.all_actions]
                action = env.all_actions[np.argmax(q_values)]
            
            next_state, reward, done = env.step(action)
            
            # Q-learning update (off-policy)
            max_q_next = max(Q[(next_state, a)] for a in env.all_actions)
            Q[(state, action)] += alpha * (
                reward + gamma * max_q_next - Q[(state, action)]
            )
            
            state = next_state
    
    return Q
```

#### 7.2 SARSA (On-Policy)

```python
def sarsa(env, n_episodes=10000, alpha=0.1, gamma=0.99, epsilon=0.1):
    Q = {}
    for state in env.all_states:
        for action in env.all_actions:
            Q[(state, action)] = 0
    
    for episode in range(n_episodes):
        state = env.reset()
        action = epsilon_greedy(Q, state, env.all_actions, epsilon)
        done = False
        
        while not done:
            next_state, reward, done = env.step(action)
            next_action = epsilon_greedy(Q, next_state, env.all_actions, epsilon)
            
            # SARSA update (on-policy)
            Q[(state, action)] += alpha * (
                reward + gamma * Q[(next_state, next_action)] - Q[(state, action)]
            )
            
            state = next_state
            action = next_action
    
    return Q
```

---

### Chapter 8: Policy Gradient Methods

#### 8.1 REINFORCE Algorithm

```python
import torch
import torch.nn as nn
import torch.optim as optim

class PolicyNetwork(nn.Module):
    def __init__(self, n_states, n_actions):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(n_states, 64),
            nn.ReLU(),
            nn.Linear(64, n_actions),
            nn.Softmax(dim=-1)
        )
    
    def forward(self, x):
        return self.fc(x)

def reinforce(env, n_episodes=1000, gamma=0.99, lr=0.01):
    policy = PolicyNetwork(env.n_states, env.n_actions)
    optimizer = optim.Adam(policy.parameters(), lr=lr)
    
    for episode in range(n_episodes):
        states, actions, rewards = [], [], []
        state = env.reset()
        done = False
        
        # Generate episode
        while not done:
            state_tensor = torch.FloatTensor(state)
            probs = policy(state_tensor)
            action = torch.multinomial(probs, 1).item()
            
            next_state, reward, done = env.step(action)
            
            states.append(state)
            actions.append(action)
            rewards.append(reward)
            state = next_state
        
        # Compute returns
        returns = []
        G = 0
        for r in reversed(rewards):
            G = r + gamma * G
            returns.insert(0, G)
        returns = torch.FloatTensor(returns)
        returns = (returns - returns.mean()) / (returns.std() + 1e-8)
        
        # Policy gradient update
        optimizer.zero_grad()
        loss = 0
        for state, action, G in zip(states, actions, returns):
            state_tensor = torch.FloatTensor(state)
            probs = policy(state_tensor)
            loss -= torch.log(probs[action]) * G
        
        loss.backward()
        optimizer.step()
    
    return policy
```

---

### Chapter 9: Applications

| Application | Algorithm | Description |
|-------------|-----------|-------------|
| Game AI | DQN, A3C | Playing Atari, Go |
| Robotics | PPO, SAC | Robot control |
| Trading | Q-Learning | Portfolio optimization |
| Recommender | Bandit | Content personalization |

---

## Summary

| Method | Model-Based | On/Off Policy | Update |
|--------|-------------|---------------|--------|
| DP | Yes | - | Bootstrap |
| MC | No | Both | Sample |
| TD | No | Both | Bootstrap |
| Q-Learning | No | Off | Bootstrap |
| SARSA | No | On | Bootstrap |
| REINFORCE | No | On | Sample |

---

**Last Updated**: 2024-01-29
