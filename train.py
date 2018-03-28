
import torch
from torch.autograd import Variable

from training_configs import *
from dnc import DNC
from repeat_copy import *

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

def train(dnc, data_loader):
	# Initialize optimizer and loss function
	optimizer = torch.optim.SGD(dnc.parameters(),
		lr=LEARNING_RATE, momentum=MOMENTUM)
	loss_func = torch.nn.MSELoss()
	
	# Define input and its true output
	num_examples = 20
	data = data_loader(num_examples)
	for inputs, true_outputs in data:
		#print(inputs, true_outputs)
		# Zero gradients
		optimizer.zero_grad()

		# Turn input/output to variable
		inputs, true_outputs = Variable(inputs), Variable(true_outputs)

		# Do a forward pass, compute loss, then do a backward pass
		pred_outputs = dnc(inputs)
		loss = loss_func(pred_outputs, true_outputs)
		loss.backward()

		# Update parameters using the optimizer
		optimizer.step()

def main():
	# Set random seed if given
	torch.manual_seed(RANDOM_SEED or torch.initial_seed())

	# Choose dataset and initialize size of data's input and output
	data_loader, input_size, output_size = create_repeat_copy()  # default parameters

	# Initialize DNC
	dnc = DNC(input_size, output_size, controller_config, memory_config)

	train(dnc, data_loader)

if __name__ == '__main__':
	main()

