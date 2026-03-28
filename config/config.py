GAMMA = 0.99
LR = 0.0005   # reduced for stability
BATCH_SIZE = 64
MEMORY_SIZE = 10000

EPSILON_START = 1.0
EPSILON_END = 0.01
EPSILON_DECAY = 0.992   # 🔥 FIXED (critical)

EPISODES = 800          # more training

TARGET_UPDATE = 5       # more frequent updates