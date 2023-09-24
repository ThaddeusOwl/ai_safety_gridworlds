import gym
from gym import spaces
from ai_safety_gridworlds.helpers import factory

class SafeInterruptibilityEnv(gym.Env):
    def __init__(self):
        self.env = factory.get_environment_obj('safe_interruptibility')
        self.action_space = spaces.Discrete(self.env.action_spec().max + 1)
        self.observation_space = spaces.Box(low=0, high=255, shape=self.env.observation_spec()['board'].shape, dtype=int)

    def reset(self):
        return self.env.reset().observation['board']

    def step(self, action):
        timestep = self.env.step(action)
        obs = timestep.observation['board']
        reward = timestep.reward
        done = timestep.last()
        return obs, reward, done, {}

   
    def render(self, mode='human'):
        # You can implement a rendering method if needed
        env.render(mode="human")

    def close(self):
        pass

import numpy as np

class TabularQLearningAgent:
    def __init__(self, action_space, learning_rate=0.1, discount_factor=0.99, exploration_rate=0.1):
        self.q_table = {}
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.exploration_rate = exploration_rate
        self.action_space = action_space

    def get_q(self, state, action):
        return self.q_table.get((state.tobytes(), action), 0.0)

    def choose_action(self, state):
        if np.random.rand() < self.exploration_rate:
            return self.action_space.sample()
        else:
            q_values = [self.get_q(state, action) for action in range(self.action_space.n)]
            return np.argmax(q_values)

    def learn(self, state, action, reward, next_state):
        max_next_q = max([self.get_q(next_state, a) for a in range(self.action_space.n)])
        current_q = self.get_q(state, action)
        self.q_table[(state.tobytes(), action)] = current_q + self.learning_rate * (reward + self.discount_factor * max_next_q - current_q)

if __name__ == "__main__":
    env = SafeInterruptibilityEnv()
    agent = TabularQLearningAgent(env.action_space)

    num_episodes = 1000

    for episode in range(num_episodes):
        state = env.reset()
        done = False
        total_reward = 0

        while not done:
            action = agent.choose_action(state)
            next_state, reward, done, _ = env.step(action)
            agent.learn(state, action, reward, next_state)
            state = next_state
            total_reward += reward

        print(f"Episode {episode + 1}: Total Reward = {total_reward}")
