
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from training_configs import *
from memory import Memory

class DNC(nn.Module):

    def __init__(self, input_size, output_size,
        controller_config, memory_config, Controller=nn.LSTM):
        super(DNC, self).__init__()

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
        self.interface_layers = self.init_interface_layers()
        # Define output layer
        self.output_size = output_size
        self.output_layer = self.init_output_layer()


    def init_state(self):
        """
        Initialize the state of the DNC.
        """
        zero_hidden = lambda: Variable(torch.zeros(
            self.controller.num_layers, BATCH_SIZE, self.controller.hidden_size))
        self.controller_state = (zero_hidden(), zero_hidden())
        self.read_words = Variable(torch.zeros(BATCH_SIZE,
            self.memory.num_reads, self.memory.word_size))


    def detach_state(self):
        """
        Detach the state of the DNC from the graph.
        """
        self.controller_state = (Variable(self.controller_state[0].data),
            Variable(self.controller_state[1].data))
        self.read_words = Variable(self.read_words.data)
        self.memory.detach_state()


    def debug(self):
        """
        Prints helpful information about the DNC for debugging.
        """
        self.memory.debug()


    def init_interface_layers(self):
        """
        Initialize all layers connected to the interface
        vector (the controller's output).
        """

        # The following functions help decorate an affine operation,
        # i.e. a linear layer, with a reshape and an activation.

        # `reshape_output` changes `f` to reshape its output to `dim`.
        reshape_output = lambda f, *dim: ( lambda x: f(x).view(-1, *dim) )
        # `add_activation` changes `f` by activating its output with `sigma`.
        add_activation = lambda f, sigma: ( lambda x: sigma(f(x)) )
        # Returns a modified linear layer transformation
        def linear(activation, *dim):
            dim_prod = dim[0] if len(dim) == 1 else dim[0] * dim[1]
            # Input size to all layers is interface (hidden) size
            layer = nn.Linear(self.controller.hidden_size, dim_prod)
            layer = reshape_output(layer, *dim)
            if activation is not None:
                layer = add_activation(layer, activation)
            return layer
        
        # This structure will hold all layers from the interface vector.
        layers = {}

        # Dimensions used
        num_writes = self.memory.num_writes
        num_reads = self.memory.num_reads
        word_size = self.memory.word_size
        num_read_modes = 1 + 2 * num_writes

        # Activations used
        sigmoid = nn.Sigmoid()
        softmax_mode = nn.Softmax(dim=2)

        # Read and write keys and their strengths.
        layers["read_keys"]       = linear(None, num_reads, word_size)
        layers["read_strengths"]  = linear(None, num_reads)
        layers["write_keys"]      = linear(None, num_writes, word_size)
        layers["write_strengths"] = linear(None, num_writes)
        # Erase and write (i.e. overwrite) vectors.
        layers["erase_vectors"] = linear(sigmoid, num_writes, word_size)
        layers["write_vectors"] = linear(sigmoid, num_writes, word_size)
        # Free, allocation, and write gates.
        layers["free_gate"]       = linear(sigmoid, num_reads)
        layers["allocation_gate"] = linear(sigmoid, num_writes)
        layers["write_gate"]      = linear(sigmoid, num_writes)
        # Read modes (forward + backward for each write head,
        # and one for content-based addressing).
        layers["read_modes"] = linear(softmax_mode, num_reads, num_read_modes)

        return layers


    def init_output_layer(self):
        """
        Initialize output layer that links the interface's outputs
        to the actual output of the DNC.
        """
        pre_output_size = self.controller.hidden_size + \
            self.memory.num_reads * self.memory.word_size

        output_linear = nn.Linear(pre_output_size, self.output_size)

        return output_linear


    def forward(self, inputs):
        """
        TODO
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
            interface = {name: layer(controller_output)
                for name, layer in self.interface_layers.items()}
            self.read_words = self.memory.update(interface)

            pre_output = torch.cat([controller_output,
                self.read_words.view(BATCH_SIZE, -1)], dim=1)
            output = self.output_layer(pre_output)

            outputs.append(output)

        return torch.stack(outputs, dim=0)





