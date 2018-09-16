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
		min_length=1, max_length=3,
		min_repeats=1, max_repeats=3):
		# Check for obvious errors and save configs
		assert num_bits > 0
		assert 1 <= min_length and min_length <= max_length
		assert 0 <= min_repeats and min_repeats <= max_repeats

		self.num_bits = num_bits
		self.min_length = min_length
		self.max_length = max_length
		self.min_repeats = min_repeats
		self.max_repeats = max_repeats
		
		# Input and output sizes are fixed for the given parameters
		self.input_size = num_bits + 2 + max_repeats
		self.output_size = num_bits + 1

		self.bits_lengths = None


	def generate(self, num_examples):
		"""
		Generate `num_examples` examples.
		"""
		for _ in range(num_examples):
			yield self.example()


	def example(self):
		"""
		Fetches/creates the next repeat-copy example.
		Also returns the lengths of the bits in each batch element.
		"""

		# Index for the start marker
		start_channel = self.num_bits

		# Get the length of observations and repeats
		bits_lengths = torch.IntTensor(BATCH_SIZE).random_(
			self.min_length, self.max_length + 1)
		repeats = torch.IntTensor(BATCH_SIZE).random_(
			self.min_repeats, self.max_repeats + 1)

		# Total sequence length is input bits + repeats + channels
		seq_length = torch.max(bits_lengths + repeats * bits_lengths + 3)
		# Fill inputs and outputs with zeros
		inputs = torch.zeros(seq_length, BATCH_SIZE, self.input_size)
		outputs = torch.zeros(seq_length, BATCH_SIZE, self.output_size)

		for i in range(BATCH_SIZE):
			# Handy sequence indices to improve readability
			obs_end = bits_lengths[i] + 1
			target_start = bits_lengths[i] + 2
			target_end = target_start + repeats[i] * bits_lengths[i]

			# Create `num_bits` random binary bits of length `obs_length`
			bits = torch.bernoulli(0.5 * torch.ones(bits_lengths[i], self.num_bits))

			# Inputs starts with a marker at `start_channel`
			inputs[0, i, start_channel] = 1
			# Then the observation bits follow (from idx 0 up to start channel)
			inputs[1:obs_end, i, :start_channel] = bits
			# Finally, we activate the appropriate repeat channel
			repeats_active_channel = start_channel + 1 + repeats[i]
			inputs[obs_end, i, repeats_active_channel] = 1

			# Fill output up to repeats of bits
			outputs[target_start:target_end, i, 1:] = bits.repeat(repeats[i], 1)
			outputs[target_end, i, 0] = 1

		self.bits_lengths = bits_lengths

		return inputs, outputs

	def loss(self, pred_outputs, true_outputs):
		"""
		TODO
		"""
		bits_lengths = self.bits_lengths

		# Clean predictions made during input
		for i in range(BATCH_SIZE):
			pred_outputs[:bits_lengths[i]+2, i, :] = 0

		# Calculate the accumulated MSE Loss for all time steps
		loss = 0
		for t in range(true_outputs.size()[0]):
			loss += F.mse_loss(pred_outputs[t, ...], true_outputs[t, ...])

		return loss


	def report(self, data, pred_outputs):
		"""
		TODO
		"""
		inputs, true_outputs = data
		bits_lengths = self.bits_lengths

		# Pick a random batch number
		i = torch.IntTensor(1).random_(1, BATCH_SIZE).item()

		# Show the true outputs and the (rounded) predictions
		print()
		print("-----------------------------------")
		print(inputs[:bits_lengths[i]+2, i, :])
		print()
		print("Expected:")
		print(true_outputs[bits_lengths[i]+2:, i, :])
		print()
		print("Got:")
		print(pred_outputs[bits_lengths[i]+2:, i, :].round().abs())
		print()
		print("Got (without rounding):")
		print(pred_outputs[bits_lengths[i]+2:, i, :])
		print()

		# Print the number of mispredicted bits
		bits_miss = (pred_outputs.round() - true_outputs).abs().sum().item()
		bits_total = self.output_size * (bits_lengths + 2).sum().item()
		print("Bits mispredicted =", int(bits_miss),
			"out of", int(bits_total))




