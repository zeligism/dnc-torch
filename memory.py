
import torch.nn as nn
from torch.autograd import Variable

BATCH_SIZE = 16
EPSILON = 1e-6

class Memory:
    def __init__(self, memory_size=128,
        word_size=20, num_writes=1, num_reads=1):
        # Hyperparameters of the memory
        self.memory_size = memory_size
        self.word_size = word_size
        self.num_writes = num_writes
        self.num_reads = num_reads
        # Initialize state of memory
        self.init_state()

    def init_state(self):
        zero_variable = lambda *dim: Variable(torch.zeros(*dim))
        self.memory_data = zero_variable(BATCH_SIZE, self.memory_size, self.memory_size)
        self.read_weights = zero_variable(BATCH_SIZE, self.num_reads, self.memory_size)
        self.write_weights = zero_variable(BATCH_SIZE, self.num_writes, self.memory_size)
        self.link = zero_variable(BATCH_SIZE, self.num_writes, self.memory_size, self.memory_size)
        self.usage = zero_variable(BATCH_SIZE, self.memory_size)

    """
    Content-based addressing. TODO
    """
    def content_based_address(self):
        pass

    """
    Updates the current state of the memory, and returns the words read by memory.

    Interface is a dictionary of of tensors that describe how the memory
    should be updated and how the data should be retrieved and written.

    The names of each tensor in the interface is
    the following (batch dimension not included):
         names                 dim
    1) read_keys          (num_reads, word_size)
    2) read_strengths     (num_reads)
    3) write_keys         (num_writes, word_size)
    4) write_strengths    (num_writes)
    5) erase_vectors      (num_writes, word_size)
    6) write_vectors      (num_writes, word_size)
    7) free_gate          (num_reads)
    8) allocation_gate    (num_writes)
    9) write_gate         (num_writes)
    T) read_modes         (num_reads, num_read_modes)

    A memory update also updates the internal state of the memory.
    The state of the memory contains (batch dimension not included):
            names         dim
    1) memory_data    (memory_size, word_size)
    2) read_weights   (num_reads, memory_size)
    3) write_weights  (num_writes, memory_size)
    4) link           (num_writes, memory_size, memory_size)
    5) usage          (memory_size)

    A memory update can be divided into these steps:
    1) Read interface vector (the tensors are passed by DNC).
    2) Update current usage using free_gate.
    3) Find allocation weighting by using the usage vector.
    4) Calculate write_weights by finding a write-content weighting first,
        by using content-based addressing function C for the weight_keys.
        Next, we use write_gate, allocation_gate, allocation weightings,
        and write-content weightings to find write_weights.
    5) Update memory by using erase_vectors and write_vectors.
    6) Update link matrix
    """
    def update(self, interface):

        self.usage = self.update_usage(interface["free_gate"])
        self.write_weights = self.update_write_weights(interface)
        self.memory_data = self.update_memory_data(write_weights,
            interface["erase_vectors"], interface["write_vectors"])
        #self.link
        #self.read_weights

        #return read_words

    """
    Calculates and returns the next/current `usage`.

    Takes `free_gate` from the `interface` vector as an input, and also uses
    previous `write_weights`, previous `read_weights`, and previous `usage`.
    Assumes that the memory has the previous states stored in `self` directly.
    Note that all the mutliplications here are element-wise.
    """
    def update_usage(self, free_gate):
        # First find the aggregate write weights of all write heads per memory cell.
        # This is in case there are more than one write head (i.e. num_writes > 1).
        cell_write_weights = 1 - torch.prod(1 - self.write_weights, dim=1)

        # Usage is retained, and in addition, memory cells that are being used for
        # writing (i.e. have high cell_memory_weights) with low usage should have
        # their usage increased, which is exactly what is done here.
        usage_after_writes = self.usage + (1 - self.usage) * cell_write_weights

        # First, recall that there is a free_gate for each read-head.
        # Here, we multiply the free_gate of each read-head with all of its read_weights,
        # which gives us new read_weights scaled by the free_gate per read-head.
        free_read_weights = free_gate.unsqueeze(dim=-1) * self.read_weights

        # Next, we calculate psi, which is interpreted as a memory retention vector.
        psi = torch.prod(1 - free_read_weights, dim=1)

        # Finally, we calculate the next usage as defined in the paper.
        usage = usage_after_writes * psi

        return usage

    """
    Calculates and returns the next/current `write_weights`.
    It's very similar to the one in DeepMind's code.

    Takes the interface as an input (there are many interface variables to unpack).
    Also, the updated usage is used here to find `phi` and allocation weightings.
    """
    def update_write_weights(self, interface):

        # Find content-based weights
        write_content_weights = self.content_based_address()  # TODO

        # Find the allocation weights
        write_allocation_weights = self.write_allocation_weights(
            interface["write_gate"] * interface["allocation_gate"])

        # Add a dimension to gates for scalar multiplication along memory cells
        write_gate = interface["write_gate"].unsqueeze(dim=-1)
        allocation_gate =  interface["allocation_gate"].unsqueeze(dim=-1)

        # Calculate `write_weights` using allocation and content-based weights
        write_weights = write_gate * (allocation_gate * write_allocation_weights +
                            (1 - allocation_gate) * write_content_weights)

        return write_weights


    """
    Calculates and returns the write weights due to allocation.
    The returned tensor will have size of (BATCH_SIZE, num_writes, memory_size).
    This function is pretty identical to the one in DeepMind's code.
    `write_alloc_gates` is simply the product of `write_gate` and `allocation_gate`.
    It is used, along with `usage`, in case there is more than one write head.

    For more than one write head, the code from DeepMind does what they call a
    "simulated new usage", where it takes into account where the previous write
    heads are writing, and update its own usage based on that. This implies that
    there is some sort of precedence or ordering among the write heads.
    """
    def write_allocation_weights(self, write_alloc_gates)
        usage_per_write = self.usage

        # Add a dimension so that when we index the write head, we get
        # a tensor of size (BATCH_SIZE, 1) to multiply it with allocation weights.
        write_alloc_gates = write_alloc_gates.unsqueeze(dim=-1)

        write_allocation_weights = []
        for i in range(self.num_writes):
            # Get allocation weights per write head and add it to the big list
            write_allocation_weights.append(self.allocation(usage_per_write))
            # This is the "simulated new usage" thing. Note that usage can only
            # further increase due to the ith (and previous) write head activity.
            usage_per_write += (1 - usage_per_write) * 
                    write_alloc_gates[:, i, :] * write_allocation_weights[i]

      # Stack allocation weights into one tensor and return
      return torch.stack(allocation_weights, dim=1)

    """
    Sort of a subroutine that runs in `update_write_weights(...)`.
    Returns the allocation weightings for one write head given the usage.
    Note that `allocation_weights_per_write` has the same size as `usage`.
    """
    def allocation(self, usage):

        usage = EPSILON + (1 - EPSILON) * usage  # TODO: Explain yourself?

        # Sort `usage` and get keep its original indices in `phi`.
        sorted_usage, phi = self.usage.sort()

        # We will add this `one` before the `sorted_usage`.
        one = torch.ones(BATCH_SIZE, 1)
        padded_sorted_usage = torch.cat([one, sorted_usage], dim=1)
        # Now we can take the "exclusive" cumprod of the `sorted_usage` by taking
        # the cumprod of `padded_sorted_usage` and dropping the last column.
        cumprod_sorted_usage = padded_sorted_usage.cumprod(dim=1)[:, :-1]

        # Next we find the allocation weights.
        sorted_allocation = (1 - sorted_usage) * cumprod_sorted_usage
        # And unsort them using the original indices in `phi`.
        allocation_weights = sorted_allocation.gather(dim=1, index=phi)

        return allocation_weights

    """
    Update the data of the memory. Returns the updated memory.
    TODO: not sure about this...
    M_t(i) = M_t-1 o (1 - w_t^T * e_t) + (w_t^T * v_t)
    where `o` is element-wise product and `*` is matrix product.
    Of course, transpose is taken without batch dimension in mind.

    The arguments have dimensions as follows:
    `write_weights`: (BATCH_SIZE, num_writes, memory_size)
    `erase_vectors`: (BATCH_SIZE, num_writes, word_size)
    `write_vectors`: (BATCH_SIZE, num_writes, word_size)
    """
    def update_memory_data(self, write_weights, erase_vectors, write_vectors):
        pass





