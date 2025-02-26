import random
import torch
import torch.nn as nn
import torch.nn.functional as F

class LearnableGate16Array(nn.Module):
    def __init__(self, number_of_gates, number_of_inputs, name):
        super(LearnableGate16Array, self).__init__()
        self.number_of_gates = number_of_gates
        self.number_of_inputs = number_of_inputs
        self.name = name
        self.w = nn.Parameter(torch.zeros((16, number_of_gates), dtype=torch.float32)) # [16, W]
        self.zeros = torch.empty(0)
        self.ones = torch.empty(0)
        self.binarized = False
        self.frozen = False
        self.c = nn.Parameter(torch.zeros((number_of_inputs, number_of_gates, 2), dtype=torch.float32)) # connectome       
        # Only Gaussian inits supported for now
        nn.init.normal_(self.w, mean=0.0, std=1)
        nn.init.normal_(self.c, mean=0.0, std=1)


    def forward(self, x):
        batch_size = x.shape[0]
        connections = F.softmax(self.c, dim=0) if not self.binarized else self.c
        # [batch_size, number_of_inputs] x [number_of_inputs, number_of_gates*2] -> [batch_size, number_of_gates*2]
        x = torch.matmul(x, connections.view(self.number_of_inputs, self.number_of_gates*2))
        x = x.view(batch_size, self.number_of_gates, 2)

        A = x[:,:,0]
        A = A.transpose(0,1)
        B = x[:,:,1]
        B = B.transpose(0,1)

        if self.zeros.shape != A.shape:
            self.zeros = torch.zeros_like(A)
        if self.ones.shape != A.shape:
            self.ones = torch.ones_like(A)
            
        # Numbered according to https://arxiv.org/pdf/2210.08277 table
        AB = A*B

        g0  = self.zeros
        g1  = AB
        g2  = A - AB
        g3  = A
        g4  = B - AB
        g5  = B
        g7  = A + B - AB
        g6  = g7    - AB            # A + B - 2 * A * B
        g8  = self.ones - g7
        g9  = self.ones - g6
        g10 = self.ones - B
        g11 = self.ones - g4
        g12 = self.ones - A
        g13 = self.ones - g2
        g14 = self.ones - AB
        g15 = self.ones

        weights = F.softmax(self.w, dim=0) if not self.binarized else self.w
        gates = torch.stack([
            g0, g1, g2, g3, g4, g5, g6, g7,
            g8, g9, g10, g11, g12, g13, g14, g15
            ], dim=0)
        assert gates.dim() > 1
        if gates.dim() == 2:
            gates = gates.unsqueeze(dim=1)                    # broadcast [C,N] -> [C,1,N]; C=16, N=batch_size
        x = (gates * weights.unsqueeze(dim=-1)).sum(dim=0) # [C,W,N] .* (broadcast [C,W] -> [C,W,1]) =[sum-over-C]=> [W,N]
        return x.transpose(0,1)
    

class Model(nn.Module):
    def __init__(self, seed, net_architecture, number_of_categories, input_size):
        super(Model, self).__init__()
        self.net_architecture = net_architecture
        self.first_layer_gates = self.net_architecture[0]
        self.last_layer_gates = self.net_architecture[-1]
        self.number_of_categories = number_of_categories
        self.input_size = input_size
        self.seed = seed
        random.seed(self.seed)
        torch.manual_seed(self.seed)
        
        self.outputs_per_category = self.last_layer_gates // self.number_of_categories
        assert self.last_layer_gates == self.number_of_categories * self.outputs_per_category

        layers_ = []
        for layer_idx, layer_gates in enumerate(net_architecture):
            if layer_idx==0:
                layers_.append(LearnableGate16Array(number_of_gates=layer_gates,number_of_inputs=input_size, name=layer_idx))
            else:
                layers_.append(LearnableGate16Array(number_of_gates=layer_gates,number_of_inputs=prev_gates, name=layer_idx))
            prev_gates = layer_gates
        self.layers = nn.ModuleList(layers_)

    def forward(self, X):
        for layer_idx in range(0, len(self.layers)):
            X = self.layers[layer_idx](X)

        X = X.view(X.size(0), self.number_of_categories, self.outputs_per_category).sum(dim=-1)
        X = F.softmax(X, dim=-1)
        return X

    def get_passthrough_fraction(self):
        pass_fraction_array = torch.zeros(len(self.layers), dtype=torch.float32, device=device)
        indices = torch.tensor([3, 5, 10, 12], dtype=torch.long)
        for layer_ix, layer in enumerate(self.layers):
            weights_after_softmax = F.softmax(layer.w, dim=0)
            pass_weight = (weights_after_softmax[indices, :]).sum()
            total_weight = weights_after_softmax.sum()
            pass_fraction_array[layer_ix] = pass_weight / total_weight
        return pass_fraction_array
    
    def state_dict(self, *args, **kwargs):
        state_dict = super(Model, self).state_dict(*args, **kwargs)
        state_dict['net_architecture'] = self.net_architecture
        state_dict['seed'] = self.seed
        if hasattr(self, 'dataset_input'):
            state_dict['dataset_input'] = self.dataset_input
        if hasattr(self, 'dataset_output'):
            state_dict['dataset_output'] = self.dataset_output
        return state_dict

    def load_state_dict(self, state_dict, strict=True):
        if 'net_architecture' in state_dict:
            self.net_architecture = state_dict.pop('net_architecture')
        if 'seed' in state_dict:
            self.seed = state_dict.pop('seed')
        super(Model, self).load_state_dict(state_dict, strict=strict)