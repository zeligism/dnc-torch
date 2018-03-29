import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from training_configs import *

"""
This is my own version of repeat copy.
"""
class RepeatCopy:

	def __init__(self, num_bits=4,
		min_seq_length=4, max_seq_length=6,
		min_repeats=0, max_repeats=3):
		# Check for obvious errors and save configs
		assert num_bits > 0
		assert 0 < min_seq_length and min_seq_length <= max_seq_length
		assert 0 <= min_repeats and min_repeats <= max_repeats
		self.num_bits = num_bits
		self.min_seq_length = min_seq_length
		self.max_seq_length = max_seq_length
		self.min_repeats = min_repeats
		self.max_repeats = max_repeats
		
		# Input and output sizes are fixed for the given parameters
		self.onehot_length = self.max_repeats - self.min_repeats + 1
		self.input_size = max_repeats - min_repeats + 1 + num_bits
		self.output_size = max_repeats * num_bits

	def __call__(self, num_examples):
		"""
		By calling the object, we create a generator, so in order to get the
		next example, we should either do that in a loop or by calling `next()`.
		For example,
			>>> dataset = RepeatCopy()
			>>> data_generator = dataset(5)
			>>> inputs, outputs = next(data_generator)
			>>> i = 0
			>>> for inputs, outputs in data_generator:
			>>>		i += 1
			>>> print(i)
			4
		"""
		for _ in range(num_examples):
			yield self.example()

	def example(self):
		"""
		Fetches/creates the next repeat-copy example.
		"""

		# Get the sequence length of examples before repeat copy query
		seq_length = torch.IntTensor(1).random_(
			self.min_seq_length, self.max_seq_length + 1)[0]

		# Create random one hot vectors of the number of repeats
		repeats = torch.LongTensor(BATCH_SIZE, 1)
		repeats = repeats.random_(self.min_repeats, self.max_repeats + 1)
		# Scatter 1s along dimension 1 on indices specified by `repeats_idx`
		repeats_onehot = torch.zeros(BATCH_SIZE, self.onehot_length)
		repeats_idx = repeats - self.min_repeats
		repeats_onehot = repeats_onehot.scatter_(
			dim=1, index=repeats_idx, value=1)

		# Create random bits of length `num_bits` each
		bits = torch.bernoulli(0.5 * torch.ones(seq_length, BATCH_SIZE, self.num_bits))

		# Input sequence include one-hot vector as well
		inputs = torch.zeros(seq_length + 1, BATCH_SIZE, self.input_size)
		inputs[0 , ...,  -self.onehot_length:] = repeats_onehot
		inputs[1:, ..., :-self.onehot_length ] = bits

		# Expected output is `bits` repeated `repeats` times
		outputs = torch.zeros(seq_length, BATCH_SIZE, self.output_size)
		for i in range(BATCH_SIZE):
		    num_repeats = repeats[i][0]
		    # If repeats is 0, then keep outputs as it is
		    if num_repeats == 0:
		    	continue
		    # Fill output up to repeats of bits
		    filled_length = num_repeats * self.num_bits
		    outputs[:, i, :filled_length] = bits[:, i, :].repeat(1, num_repeats)

		return inputs, outputs

	def report(self, inputs, true_outputs, pred_outputs):
		# Get the data from any batch (pick the first w.l.o.g.)
		example_input = inputs[1:, 0, :-self.onehot_length]
		expected = true_outputs[:, 0, :]
		got = pred_outputs[:, 0, :]
		# Get the number of repeats
		onehot = inputs[0, 0, -self.onehot_length:]
		repeat = self.min_repeats + onehot.nonzero()[0][0]

		print()
		print("-----------------------------------")
		print("Print each row %d out of %d times:" % (repeat, self.max_repeats))
		print(self.readable(example_input))
		print()
		print("Expected:")
		print(self.readable(expected))
		print()
		print("Got:")
		print(self.readable(got))
		print()

	def readable(self, tensor):
		col_strings = []
		rows, cols = tensor.size()
		for i in range(rows):
			row_strings = []
			for j_next in range(0, cols, self.num_bits):
				# Take bits one at a time, join them with no seperation
				row_strings.append("".join(map(lambda x: str(int(x)),
					tensor[i, j_next : j_next + self.num_bits])))
			# Seperate bits with a vertical line for readability
			col_strings.append(" | ".join(row_strings))
		# Add a new line between rows and return
		return "\n".join(col_strings)




