from collections import namedtuple

import torch.nn as nn
from torch.autograd import Variable

from .memory import Memory

# Hyper parameters
BATCH_SIZE = 16
EPOCHS = 100000
LEARNING_RATE = 1e-4

# Controller configurations
controller_config = {
    "input_size": 8,    # Depends on input? 
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


Controller = nn.lstm

class DNC(nn.Module):

    def __init__(self, controller_config, memory_config, output_size):
        super(DNC, self).__init__()

        # Assign controller and memory
        self.controller = Controller(**controller_config)
        self.memory = Memory(self.hidden_size, **memory_config)
        # Initialize state of DNC
        self.state = self.init_state()
        # Define interface layers
        self.layers = self.init_interface_layers()
        self.layers["output_linear"] = nn.Linear(
            self.controller.hidden_size, output_size)

    """
    Initialize the state of the DNC.
    """
    def init_state(self):
        zero_init = lambda: Variable(torch.zeros(self.controller.num_layers,
            BATCH_SIZE, self.controller.hidden_size))
        controller_state = (zero_init(), zero_init())
        memory_state = None  # TODO: Call memory's init()?
        read_vectors = None  # TODO: What is this?
        return controller_state, memory_state, read_vectors

    """
    Initialize all layers connected to the interface
    vector (the controller's output).
    """
    def init_interface_layers(self):

        """
        The following functions help decorate an affine operation,
        i.e. a linear layer, with a reshape and an activation.
        """
        # `reshape_output` changes `f` to reshape its output to `dim`.
        reshape_output = lambda f, *dim: ( lambda x: f(x).view(-1, *dim) )
        # `add_activation` changes `f` by activating its output with `sigma`.
        add_activation = lambda f, sigma: ( lambda x: sigma(f(x)) )
        # Returns a modified linear layer transformation
        def linear(activation, *dim):
            dim_prod = dim[0] if len(dim) == 1 else dim[0] * dim[1]
            # Input size to all layers is interface (hidden) size
            layer = nn.Linear(self.hidden_size, dim_prod)
            layer = reshape_output(layer, *dim)
            if activation is not None:
                layer = add_activation(layer, activation)
            return layer
        
        # This structure will hold all layers from the interface vector.
        layers = {}

        # Dimensions used
        num_writes = self.memory.num_writes
        num_writes = self.memory.num_reads
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
        layers["read_mode"] = linear(softmax_mode, num_reads, num_read_modes)

        return layers

    """
    Marching forward in the computation graph.
    """
    def forward(self, inputs):
        """
        Note: TODO
        `inputs` should have dimension:
            (sequence_size, batch_size, input_size)
        `read_vectors` should have dimension:
            (batch_size, num_read_heads * word_size)
        """
        assert len(inputs.size()) == 3:

        # Unpack prev state
        controller_state, memory_state, read_vectors = self.state

        for i in range(inputs.size()[0]):
            """ We go through the inputs one by one. """

            # X_t = x_t ++ [(r^R)_t-1]
            controller_input = torch.cat([inputs[i], read_vectors], dim=1)
            # Add sequence dimension
            controller_input = controller_input.view(1, -1)
            # Run one step of controller
            controller_output, controller_state = self.controller(
                controller_input, controller_state)
            # Remove sequence dimension
            controller_output = controller_output.squeeze(dim=0)

            """ The memory access just takes the interface
            vector from the controller as an input. """
            memory_inputs = {name: layer(controller_output)
                                for name, layer in self.layers.items()}
            read_vectors, memory_state = self.memory.read(
                memory_inputs, memory_state)

            # y_t = v_t + W_r * [(r^R)_t] TODO ??
            output = self.layers["output_linear"](
                torch.cat([controller_output, read_vectors], dim=1))

            # TODO: accumulate outputs in a tensor

        # Pack state
        self.state = (controller_state, memory_state, read_vectors)
        return output, self.state





