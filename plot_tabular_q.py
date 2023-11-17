import numpy as np
import safe_grid_gym
import gym

# Define the Q-learning Agent
class QLearningAgent:
        
    def __init__(self, action_space, learning_rate=0.8, discount_factor=0.8, exploration_rate=1.0, exploration_decay_rate=0.995):
        self.action_space = action_space
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.exploration_rate = exploration_rate
        self.exploration_decay_rate = exploration_decay_rate
        self.q_table = {}

    def get_action(self, state):
    
        if np.random.rand() < self.exploration_rate:
            return self.action_space.sample()
        return np.argmax(self.get_q_values(state))

    def get_q_values(self, state):
        state_str = str(state)
        if state_str not in self.q_table:
            self.q_table[state_str] = np.zeros(self.action_space.n)
        return self.q_table[state_str]

    def update(self, state, action, reward, next_state, done):
        
        state_str, next_state_str = str(state), str(next_state)
        best_next_q_value = np.max(self.get_q_values(next_state))
        current_q_value = self.get_q_values(state)[action]
        
        # Q-learning update rule
        new_q_value = current_q_value + self.learning_rate * (reward + self.discount_factor * best_next_q_value - current_q_value)
        self.q_table[state_str][action] = new_q_value

        if done:
            self.exploration_rate *= self.exploration_decay_rate

# Initialize environment and agent
env = gym.make("SafeInterruptibility")
# env = gym.make("DistributionalShift")
# env = gym.make("WhiskyGold")
# env = gym.make("AbsentSupervisor")
# env = gym.make("SideEffectsSokoban")
# env = gym.make("BoatRace")
# env = gym.make("TomatoWatering")
# env = gym.make("FriendFoe")
# env = gym.make("IslandNavigation")
agent = QLearningAgent(env.action_space)

# Train the Agent
num_episodes = 500
rewards = []  # List to store rewards
for episode in range(num_episodes):
    
    state = env.reset()
    done = False
    total_reward = 0

    while not done:
        action = agent.get_action(state)
        next_state, reward, done, info = env.step(action)
        agent.update(state, action, reward, next_state, done)
        state = next_state
        total_reward += reward
    
    rewards.append(total_reward)  # Store reward
    # print(f"Episode {episode + 1}: Total Reward = {total_reward}")
    print(f"Episode {episode + 1}: Total Reward = {total_reward}")
    

# Test the Trained Agent
for _ in range(10):
    state = env.reset()
    done = False
    while not done:
        action = agent.get_action(state)
        state, _, done, _ = env.step(action)
        env.render(mode="human")


# Importing matplotlib for plotting
import matplotlib.pyplot as plt

# Plotting the rewards
plt.plot(rewards)
plt.title('Rewards per Episode')
plt.xlabel('Episode')
plt.ylabel('Total Reward')
plt.show()
