import os, sys
import numpy as np
import torch
import torch.nn as nn
from scipy.sparse import linalg
import scipy.sparse as sp


def calculate_normalized_laplacian(adj):
    adj = sp.coo_matrix(adj)
    d = np.array(adj.sum(1))
    d_inv_sqrt = np.power(d, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    normalized_laplacian = sp.eye(adj.shape[0]) - adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()
    return normalized_laplacian


def calculate_random_walk_matrix(adj_mx):
    adj_mx = sp.coo_matrix(adj_mx)
    d = np.array(adj_mx.sum(1))
    d_inv = np.power(d, -1).flatten()
    d_inv[np.isinf(d_inv)] = 0.
    d_mat_inv = sp.diags(d_inv)
    random_walk_mx = d_mat_inv.dot(adj_mx).tocoo()
    return random_walk_mx


def calculate_reverse_random_walk_matrix(adj_mx):
    return calculate_random_walk_matrix(np.transpose(adj_mx))


def calculate_scaled_laplacian(adj_mx, lambda_max=2, undirected=True):
    if undirected:
        adj_mx = np.maximum.reduce([adj_mx, adj_mx.T])
    L = calculate_normalized_laplacian(adj_mx)
    if lambda_max is None:
        lambda_max, _ = linalg.eigsh(L, 1, which='LM')
        lambda_max = lambda_max[0]
    L = sp.csr_matrix(L)
    M, _ = L.shape
    I = sp.identity(M, format='csr', dtype=L.dtype)
    L = (2 / lambda_max * L) - I
    return L.astype(np.float32)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


class LayerParams:
    def __init__(self, rnn_network: torch.nn.Module, layer_type: str):
        self._rnn_network = rnn_network
        self._layer_type = layer_type

    def get_weights(self, shape):
        return torch.nn.Parameter(torch.empty(*shape))

    def get_biases(self, length, bias_start=0.0):
        return torch.nn.Parameter(torch.empty(length))


class DCGRUCell(torch.nn.Module):
    def __init__(self, num_units, adj_mx, max_diffusion_step, num_nodes, nonlinearity='tanh',
                 filter_type="laplacian", use_gc_for_ru=True, device=None):
        super().__init__()
        self._activation = torch.tanh if nonlinearity == 'tanh' else torch.relu
        self._num_nodes = num_nodes
        self._num_units = num_units
        self._max_diffusion_step = max_diffusion_step
        self._supports = []
        self._use_gc_for_ru = use_gc_for_ru
        self.device = device
        self._layer_params = LayerParams(self, 'DCGRU')

        supports = []
        if filter_type == "laplacian":
            supports.append(calculate_scaled_laplacian(adj_mx, lambda_max=None))
        elif filter_type == "random_walk":
            supports.append(calculate_random_walk_matrix(adj_mx).T)
        elif filter_type == "dual_random_walk":
            supports.append(calculate_random_walk_matrix(adj_mx).T)
            supports.append(calculate_reverse_random_walk_matrix(adj_mx).T)
        else:
            supports.append(calculate_scaled_laplacian(adj_mx))

        for support in supports:
            self._supports.append(self._build_sparse_matrix(support))

        self.input_dim = num_nodes
        self.num_matrices = len(self._supports) * self._max_diffusion_step + 1
        
        self.w_ru = nn.Parameter(torch.empty(self.input_dim + self._num_units, 2 * self._num_units))
        self.b_ru = nn.Parameter(torch.zeros(2 * self._num_units))
        self.w_c = nn.Parameter(torch.empty(self.input_dim + self._num_units, self._num_units))
        self.b_c = nn.Parameter(torch.zeros(self._num_units))

        nn.init.xavier_uniform_(self.w_ru)
        nn.init.xavier_uniform_(self.w_c)

    def _build_sparse_matrix(self, L):
        L = L.tocoo()
        indices = np.column_stack((L.row, L.col))
        indices = indices[np.lexsort((indices[:, 1], indices[:, 0]))]
        L = torch.sparse_coo_tensor(indices.T, L.data, L.shape, device=self.device)
        return L

    def forward(self, inputs, hx):
        output_size = 2 * self._num_units
        fn = self._gconv

        value = torch.sigmoid(fn(inputs, hx, output_size, bias_start=1.0))
        value = torch.reshape(value, (-1, self._num_nodes, output_size))
        r, u = torch.split(tensor=value, split_size_or_sections=self._num_units, dim=-1)
        r = torch.reshape(r, (-1, self._num_nodes * self._num_units))
        u = torch.reshape(u, (-1, self._num_nodes * self._num_units))

        c = self._gconv(inputs, r * hx, self._num_units)
        if self._activation is not None:
            c = self._activation(c)

        new_state = u * hx + (1.0 - u) * c
        return new_state

    def _gconv(self, inputs, state, output_size, bias_start=0.0):
        batch_size = inputs.shape[0]
        inputs = torch.reshape(inputs, (batch_size, self._num_nodes, -1))
        state = torch.reshape(state, (batch_size, self._num_nodes, -1))
        inputs_and_state = torch.cat([inputs, state], dim=2)
        input_size = inputs_and_state.size(2)

        x = inputs_and_state
        x0 = x.permute(1, 2, 0)
        x0 = torch.reshape(x0, shape=[self._num_nodes, input_size * batch_size])
        x = torch.unsqueeze(x0, 0)

        if self._max_diffusion_step == 0:
            pass
        else:
            for support in self._supports:
                x1 = torch.sparse.mm(support, x0)
                x = self._concat(x, x1)

                for k in range(2, self._max_diffusion_step + 1):
                    x2 = 2 * torch.sparse.mm(support, x1) - x0
                    x = self._concat(x, x2)
                    x0, x1 = x1, x2

        num_matrices = len(self._supports) * self._max_diffusion_step + 1
        x = torch.reshape(x, shape=[num_matrices, self._num_nodes, input_size, batch_size])
        x = x.permute(3, 1, 2, 0)
        x = torch.reshape(x, shape=[batch_size * self._num_nodes, input_size * num_matrices])

        if output_size == 2 * self._num_units:
            weights = self.w_ru
            biases = self.b_ru
        else:
            weights = self.w_c
            biases = self.b_c

        x = torch.matmul(x, weights)
        x = x + biases
        return torch.reshape(x, [batch_size, self._num_nodes * output_size])

    def _concat(self, x, x_):
        x_ = torch.unsqueeze(x_, 0)
        return torch.cat([x, x_], dim=0)
