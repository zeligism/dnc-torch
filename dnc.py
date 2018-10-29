
import torch
import torch.nn as nn
import torch.nn.functional as F

from training_configs import *
from memory import Memory

class DNC(nn.Module):

    def __init__(self, input_size, output_size,
        controller_config, memory_config, Controller=nn.LSTM):
        super().__init__()

        # Initialize memory
        self.memory = Memory(**memory_config)

        # First add read vectors' size to controller's input_size
        self.input_size = input_size + \
            self.memory.num_reads * self.memory.word_size
        # Now initialize controller
        self.controller = Controller(self.input_size, **controller_config)

        # Initialize state of DNC
        self.init_state()

        # Define interface layers
        self.interface_layer = DNC_InterfaceLayer(
            self.controller.hidden_size,
            self.memory.num_writes,
            self.memory.num_reads,
            self.memory.word_size,
            )

        # Define output layer
        self.output_size = output_size
        pre_output_size = self.controller.hidden_size + \
            self.memory.num_reads * self.memory.word_size
        self.output_layer = nn.Linear(pre_output_size, self.output_size)


    def init_state(self):
        """
        Initialize the state of the DNC.
        """
        # Initialize controller's state
        num_layers = self.controller.num_layers
        hidden_size = self.controller.hidden_size
        self.controller_state = (
            torch.zeros(num_layers, BATCH_SIZE, hidden_size),
            torch.zeros(num_layers, BATCH_SIZE, hidden_size))
        # Initialize read_words state
        self.read_words = torch.zeros(BATCH_SIZE,
            self.memory.num_reads, self.memory.word_size)


    def detach_state(self):
        """
        Detach the state of the DNC from the graph.
        """
        self.controller_state = (
            self.controller_state[0].detach(),
            self.controller_state[1].detach())
        self.read_words.detach_()
        self.memory.detach_state()


    def debug(self):
        """
        Prints helpful information about the DNC for debugging.
        """
        self.memory.debug()


    def forward(self, inputs):
        """
        Makes one forward pass one the inputs.
        `inputs` should have dimension:
            (sequence_size, batch_size, input_size)
        `read_words` should have dimension:
            (batch_size, num_reads * word_size)
        """

        self.detach_state()

        outputs = []
        for i in range(inputs.size()[0]):
            # We go through the inputs in the sequence one by one.

            # X_t = input ++ read_vectors/read_words
            controller_input = torch.cat([
                inputs[i].view(BATCH_SIZE, -1),
                self.read_words.view(BATCH_SIZE, -1)], dim=1)
            # Add sequence dimension
            controller_input = controller_input.unsqueeze(dim=0)
            # Run one step of controller
            controller_output, self.controller_state = self.controller(
                controller_input, self.controller_state)
            # Remove sequence dimension
            controller_output = controller_output.squeeze(dim=0)

            """ Compute all the interface tensors by passing
            the controller's output to all the layers, and
            then passing the result as an input to memory. """
            interface = self.interface_layer(controller_output)
            self.read_words = self.memory.update(interface)

            pre_output = torch.cat([controller_output,
                self.read_words.view(BATCH_SIZE, -1)], dim=1)
            output = self.output_layer(pre_output)

            outputs.append(output)

        return torch.stack(outputs, dim=0)


class DNC_InterfaceLayer(nn.Module):
    """
    The interface layer of the DNC.
    Simply applies linear layers to the hidden state of the controller.
    Each linear layer is associated with an interface vector,
    as described in the paper. The output is reshaped accordingly in LinearView,
    and activations are applied depending on the type of interface vector.
    """
    def __init__(self, input_size, num_writes, num_reads, word_size):
        super().__init__()

        # Read and write keys and their strengths.
        self.read_keys       = LinearView(input_size, [num_reads, word_size])
        self.read_strengths  = LinearView(input_size, [num_reads])
        self.write_keys      = LinearView(input_size, [num_writes, word_size])
        self.write_strengths = LinearView(input_size, [num_writes])
        # Erase and write (i.e. overwrite) vectors.
        self.erase_vectors   = LinearView(input_size, [num_writes, word_size])
        self.write_vectors   = LinearView(input_size, [num_writes, word_size])
        # Free, allocation, and write gates.
        self.free_gate       = LinearView(input_size, [num_reads])
        self.allocation_gate = LinearView(input_size, [num_writes])
        self.write_gate      = LinearView(input_size, [num_writes])
        # Read modes (forward + backward for each write head,
        # and one for content-based addressing).
        num_read_modes = 1 + 2 * num_writes
        self.read_modes = LinearView(input_size, [num_reads, num_read_modes])


    def forward(self, x):
        return {
            "read_keys":       self.read_keys(x),
            "read_strengths":  self.read_strengths(x),
            "write_keys":      self.write_keys(x),
            "write_strengths": self.write_strengths(x),
            "erase_vectors":   torch.sigmoid(self.erase_vectors(x)),
            "write_vectors":   torch.sigmoid(self.write_vectors(x)),
            "free_gate":       torch.sigmoid(self.free_gate(x)),
            "allocation_gate": torch.sigmoid(self.allocation_gate(x)),
            "write_gate":      torch.sigmoid(self.write_gate(x)),
            "read_modes":      F.softmax(self.read_modes(x), dim=2),
        }


class LinearView(nn.Module):
    """
    Similar to linear, except that it outputs a tensor with size `dim`.
    It is assumed that the first dimension is the batch dimension.
    """
    def __init__(self, input_size, output_view):
        super().__init__()
        # Calculate output size (just the product of dims in output_view)
        output_size = 1
        for dim in output_view: output_size *= dim
        # Define the layer and the desired view of the output
        self.layer = nn.Linear(input_size, output_size)
        self.output_view = output_view


    def forward(self, x):
        # -1 because we assume batch dimension exists
        return self.layer(x).view(-1, *self.output_view)





