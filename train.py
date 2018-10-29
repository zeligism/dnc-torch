
import time

import torch
import torch.nn.functional as F

from training_configs import *
from dnc import DNC
from repeat_copy import RepeatCopy

# Define controller and memory configurations
controller_config = {
    "hidden_size": HIDDEN_SIZE,
    "num_layers": NUM_LAYERS,
}
memory_config = {
    "memory_size": MEMORY_SIZE,
    "word_size": WORD_SIZE,
    "num_writes": NUM_WRITES,
    "num_reads": NUM_READS,
}

def train(dnc, dataset):
	# Initialize optimizer and loss function
	optimizer = torch.optim.SGD(dnc.parameters(),
		lr=LEARNING_RATE, momentum=MOMENTUM)
	# Adam seems to be faster (maybe)
	optimizer = torch.optim.Adam(dnc.parameters())

	# Define input and its true output
	start_time = time.time()
	for i, data in enumerate(dataset.generate(NUM_EXAMPLES)):
		# Zero gradients
		optimizer.zero_grad()

		# Unpack input/output and turn them into variables
		inputs, true_outputs = data

		# Do a forward pass, compute loss, then do a backward pass
		pred_outputs = dnc(inputs)
		loss = dataset.loss(pred_outputs, true_outputs)
		loss.backward()

		# Update parameters using the optimizer
		optimizer.step()

		# Print report when we reach a checkpoint
		if (i + 1) % CHECKPOINT == 0:
			dataset.report(data, pred_outputs.data)
			#dnc.debug()
			print("[%d/%d] Loss = %.3f" % (i+1, NUM_EXAMPLES, loss.item()) )
			print("Time elapsed = %ds" % (time.time() - start_time))

def main():
	# Set random seed if given
	torch.manual_seed(RANDOM_SEED or torch.initial_seed())

	# Choose dataset and initialize size of data's input and output
	dataset = RepeatCopy()  # default parameters

	# Initialize DNC
	dnc = DNC(dataset.input_size, dataset.output_size,
		controller_config, memory_config)

	train(dnc, dataset)

if __name__ == '__main__':
	main()

