
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


class Memory:
    def __init__(self, memory_size=16,
        word_size=16, num_writes=1, num_reads=4):
        self.memory_size = memory_size
        self.word_size = word_size
        self.num_writes = num_writes
        self.num_reads = num_reads

    def read(self, interface):
        pass



