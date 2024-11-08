import os
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class R_Sparse_Linear(nn.Module):
    def __init__(self, in_features, out_features, sparsity=0, bias=True, device=None, dtype=None):
        super(R_Sparse_Linear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features))
        else:
            self.register_parameter('bias', None)

        self.sparsity = sparsity

    def forward(self, input):
        # Apply R_sparse forward pass
        bs, token, hidden = input.size()

        threshold = torch.kthvalue(input.abs().view(-1), int(input.numel() * self.sparsity)).values
        s_mask = input.abs().gt(threshold).to(input.dtype)
        sparse_input = input * s_mask
        sparse_output = F.linear(sparse_input, self.weight, self.bias)

        return sparse_output


class LlamaForCausalLM_Sparse_Aware(LlamaForCausalLM):
    def __init__(self, config):
        super().__init__(config)
        num_layers = len(self.model.layers)
        print('Sparsity: ', config.sparsity)
        for layer_idx in range(num_layers):
            original_linear_layer = self.model.layers[layer_idx].mlp.gate_proj
            gate_proj = R_Sparse_Linear(original_linear_layer.in_features, original_linear_layer.out_features, sparsity=config.sparsity, bias=False)
            gate_proj.weight.data = original_linear_layer.weight.data
            self.model.layers[layer_idx].mlp.gate_proj = gate_proj

            original_linear_layer = self.model.layers[layer_idx].mlp.up_proj
            up_proj = R_Sparse_Linear(original_linear_layer.in_features, original_linear_layer.out_features, sparsity=config.sparsity, bias=False)
            up_proj.weight.data = original_linear_layer.weight.data
            self.model.layers[layer_idx].mlp.up_proj = up_proj

            original_linear_layer = self.model.layers[layer_idx].mlp.down_proj
            down_proj = R_Sparse_Linear(original_linear_layer.in_features, original_linear_layer.out_features, sparsity=config.sparsity, bias=False)
            down_proj.weight.data = original_linear_layer.weight.data
            self.model.layers[layer_idx].mlp.down_proj = down_proj

            original_linear_layer = self.model.layers[layer_idx].self_attn.q_proj
            q_proj = R_Sparse_Linear(original_linear_layer.in_features, original_linear_layer.out_features, sparsity=config.sparsity, bias=False)
            q_proj.weight.data = original_linear_layer.weight.data
            self.model.layers[layer_idx].self_attn.q_proj = q_proj

            original_linear_layer = self.model.layers[layer_idx].self_attn.k_proj
            k_proj = R_Sparse_Linear(original_linear_layer.in_features, original_linear_layer.out_features, sparsity=config.sparsity, bias=False)
            k_proj.weight.data = original_linear_layer.weight.data
            self.model.layers[layer_idx].self_attn.k_proj = k_proj

            original_linear_layer = self.model.layers[layer_idx].self_attn.v_proj
            v_proj = R_Sparse_Linear(original_linear_layer.in_features, original_linear_layer.out_features, sparsity=config.sparsity, bias=False)
            v_proj.weight.data = original_linear_layer.weight.data
            self.model.layers[layer_idx].self_attn.v_proj = v_proj

            original_linear_layer = self.model.layers[layer_idx].self_attn.o_proj
            o_proj = R_Sparse_Linear(original_linear_layer.in_features, original_linear_layer.out_features, sparsity=config.sparsity, bias=False)
            o_proj.weight.data = original_linear_layer.weight.data
            self.model.layers[layer_idx].self_attn.o_proj = o_proj
