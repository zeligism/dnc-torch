
import torch
from torch.autograd import Variable

from training_configs import *
from dnc import DNC

def main():

	# Set random seed if given
	torch.manual_seed(RANDOM_SEED or torch.initial_seed())
	dataset = None

	# Controller configurations
	controller_config = {
	    "hidden_size": 64,
	    "num_layers": 1,
	}
	# Memory configurations
	memory_config = {
	    "memory_size": 16,
	    "word_size": 16,
	    "num_writes": 1,
	    "num_reads": 4,
	}

	# Initialize DNC
	input_size = 8
	output_size = 8
	dnc = DNC(input_size, output_size, controller_config, memory_config)

	# Initialize optimizer and loss function
	optimizer = torch.optim.SGD(dnc.parameters(),
		lr=LEARNING_RATE, momentum=MOMENTUM)
	loss_func = torch.nn.MSELoss()
	
	# Define input and its true output
	sequence_length = 2
	inputs = Variable(torch.rand(sequence_length, BATCH_SIZE, input_size))
	true_outputs = Variable(torch.rand(sequence_length, BATCH_SIZE, output_size))

	# Zero grads and do a forward pass
	optimizer.zero_grad()
	pred_outputs = dnc(inputs)
	print(pred_outputs)

	# Compute loss and do a backward pass
	loss = loss_func(pred_outputs, true_outputs)
	loss.backward()

	# Update parameters using our optimizer
	optimizer.step()

if __name__ == '__main__':
	main()

