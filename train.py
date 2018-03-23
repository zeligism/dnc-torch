
import torch
from torch.autograd import Variable

from training_configs import *
from dnc import DNC

def main():

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
	
	# Test forward
	sequence_length = 100
	input_ = torch.rand(sequence_length, BATCH_SIZE, input_size)
	result = dnc(Variable(input_))
	print(result)


if __name__ == '__main__':
	main()

