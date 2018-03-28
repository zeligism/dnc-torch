import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from training_configs import *

def prediction_report(inputs, true_outputs, pred_outputs):
	...

"""
This is my own version of repeat copy.
"""
def create_repeat_copy(**configs):
	def repeat_copy_generator(n):
		return repeat_copy(**configs, num_examples=n)
	inputs, outputs = next(repeat_copy(**configs, num_examples=1))
	input_size, output_size = inputs.size()[-1], outputs.size()[-1]
	return repeat_copy_generator, input_size, output_size

def repeat_copy(num_bits=4,
	min_seq_length=4, max_seq_length=6,
	min_repeats=0, max_repeats=3,
	num_examples=1000):

	assert num_bits > 0
	assert 0 < min_seq_length and min_seq_length <= max_seq_length
	assert 0 <= min_repeats and min_repeats <= max_repeats

	# Input and output sizes are fixed for the given parameters
	onehot_length = max_repeats - min_repeats + 1
	input_size = onehot_length + num_bits
	output_size = max_repeats * num_bits

	for _ in range(num_examples):
		# Get the sequence length of examples before repeat copy query
		seq_length = torch.IntTensor(1).random_(
			min_seq_length, max_seq_length + 1)[0]

		# Create random one hot vectors of the number of repeats
		repeats = torch.LongTensor(BATCH_SIZE, 1)
		repeats = repeats.random_(min_repeats, max_repeats + 1)
		# Scatter 1s along dimension 1 on indices specified by `repeats_idx`
		repeats_onehot = torch.zeros(BATCH_SIZE, onehot_length)
		repeats_onehot = repeats_onehot.scatter_(
			dim=1, index=repeats-min_repeats, value=1)

		# Create random bits of length `num_bits` each
		bits = torch.bernoulli(0.5 * torch.ones(seq_length, BATCH_SIZE, num_bits))

		# Input sequence include a last input of the one-hot vector
		inputs = torch.zeros(seq_length + 1, BATCH_SIZE, input_size)
		inputs[-1, ..., -onehot_length:] = repeats_onehot
		inputs[:seq_length, ..., :-onehot_length] = bits

		# Expected output is `bits` repeated `repeats` times
		outputs = torch.zeros(seq_length + 1, BATCH_SIZE, output_size)
		for i in range(BATCH_SIZE):
		    num_repeats = repeats[i][0]
		    if num_repeats == 0: continue  # Keep outputs as it is
		    outputs[:seq_length, i, :num_repeats * num_bits] =\
		    	bits[:, i, :].repeat(1, num_repeats)
		# Add the same onehot sequence step at the end of output
		outputs[-1, ..., -onehot_length:] = repeats_onehot

		yield inputs, outputs
