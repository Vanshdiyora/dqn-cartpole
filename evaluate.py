import gymnasium as gym
import torch
import time
from model.dqn_model import DQN

def evaluate():
    env = gym.make("CartPole-v1", render_mode="human")

    model = DQN(4, 2)
    model.load_state_dict(torch.load("outputs/models/dqn.pth"))
    model.eval()

    state = env.reset()[0]
    done = False
    total_reward = 0

    while not done:
        with torch.no_grad():
            action = torch.argmax(model(torch.FloatTensor(state))).item()

        state, _, done, _, _ = env.step(action)
        total_reward += 1
        time.sleep(0.02)

    print("Total Reward:", total_reward)
    env.close()

if __name__ == "__main__":
    evaluate()