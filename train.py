import gymnasium as gym
from agent.dqn_agent import DQNAgent
from agent.replay_buffer import ReplayBuffer
from config import config
import matplotlib.pyplot as plt
import torch
import os

def train():
    env = gym.make("CartPole-v1")

    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n

    agent = DQNAgent(state_size, action_size, config)
    memory = ReplayBuffer(config.MEMORY_SIZE)

    rewards = []

    for episode in range(config.EPISODES):
        state = env.reset()[0]
        total_reward = 0
        done = False

        # 🔥 update target network
        if episode % config.TARGET_UPDATE == 0:
            agent.target_model.load_state_dict(agent.model.state_dict())

        while not done:
            action = agent.act(state, env)

            next_state, reward, done, _, _ = env.step(action)

            # 🔥 reward shaping (critical)
            if done:
                reward = -10

            memory.add((state, action, reward, next_state, done))

            state = next_state
            total_reward += reward

            agent.train(memory, config.BATCH_SIZE)

        agent.update_epsilon()
        rewards.append(total_reward)

        print(f"Episode {episode}, Reward: {total_reward}, Epsilon: {agent.epsilon:.3f}")

    # 🔥 auto create folders
    os.makedirs("outputs/models", exist_ok=True)
    os.makedirs("outputs/plots", exist_ok=True)

    torch.save(agent.model.state_dict(), "outputs/models/dqn.pth")

    plt.plot(rewards)
    plt.xlabel("Episodes")
    plt.ylabel("Reward")
    plt.title("Training Progress")
    plt.savefig("outputs/plots/training.png")
    plt.show()

if __name__ == "__main__":
    train()