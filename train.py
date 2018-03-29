
import time

import torch
import torch.nn.functional as F
from torch.autograd import Variable

from training_configs import *
from dnc import DNC
from repeat_copy import RepeatCopy

# Define controller and memory configurations
controller_config = {
    "hidden_size": 64,
    "num_layers": 1,
}
memory_config = {
    "memory_size": 32,
    "word_size": 8,
    "num_writes": 1,
    "num_reads": 4,
}

def train(dnc, dataset):
	# Initialize optimizer and loss function
	optimizer = torch.optim.SGD(dnc.parameters(),
		lr=LEARNING_RATE, momentum=MOMENTUM)
	loss_func = torch.nn.MSELoss()
	
	last_loss = 0

	# Define input and its true output
	start_time = time.time()
	for i, data in enumerate(dataset(NUM_EXAMPLES)):
		# Unpack data
		inputs, true_outputs = data

		# Zero gradients
		optimizer.zero_grad()

		# Turn input/output to variable
		inputs, true_outputs = Variable(inputs), Variable(true_outputs)

		# Do a forward pass, compute loss, then do a backward pass
		pred_outputs = dnc(inputs)
		# Skip first output (prediction from onehot vector)
		loss = loss_func(pred_outputs[1:], true_outputs)
		loss.backward()

		# Update parameters using the optimizer
		optimizer.step()

		# Print report when we reach a checkpoint
		if (i + 1) % CHECKPOINT == 0:
			pred_bits = pred_outputs.data.clamp(0,1).round()
			dataset.report(inputs.data, true_outputs.data, pred_bits)
			print('[%d/%d] Loss = %.3f' % (i+1, NUM_EXAMPLES, loss.data.mean()))
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

