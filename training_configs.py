
# Just a simple configs file with all the hyperparameters you need to set

# Note: 0 sets the seed to torch's initial seed
RANDOM_SEED = 10

# Training-specific hyperparameters
BATCH_SIZE = 8
EPSILON = 1e-6
LEARNING_RATE = 1e-4
MOMENTUM = 0.9
NUM_EXAMPLES = 20000
CHECKPOINT = NUM_EXAMPLES // 200

# Controller configurations
HIDDEN_SIZE = 64
NUM_LAYERS = 1

# Memory configurations
MEMORY_SIZE = 32
WORD_SIZE = 8
NUM_WRITES = 1
NUM_READS = 4
