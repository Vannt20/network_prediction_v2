import numpy as np
import torch
import torch.nn as nn
import os, sys

try:
    from dcrnn_cell import DCGRUCell
except ImportError:
    from Graph_models.dcrnn_cell import DCGRUCell


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


class Seq2SeqAttrs:
    def __init__(self, adj_mx, nodes, num_rnn_layers=2, rnn_units=64):
        self.adj_mx = adj_mx
        self.max_diffusion_step = 2
        self.cl_decay_steps = 1000
        self.filter_type = 'laplacian'
        self.num_nodes = nodes
        self.num_rnn_layers = num_rnn_layers
        self.rnn_units = rnn_units
        self.hidden_state_size = self.num_nodes * self.rnn_units


class EncoderModel(nn.Module, Seq2SeqAttrs):
    def __init__(self, adj_mx, seq_len, nodes, device, num_rnn_layers=2, rnn_units=64, **model_kwargs):
        nn.Module.__init__(self)
        Seq2SeqAttrs.__init__(self, adj_mx, nodes, num_rnn_layers, rnn_units)
        self.device = device
        self.input_dim = nodes
        self.seq_len = seq_len
        self.dcgru_layers = nn.ModuleList(
            [DCGRUCell(self.rnn_units, adj_mx, self.max_diffusion_step, self.num_nodes,
                       filter_type=self.filter_type, device=device) for _ in range(self.num_rnn_layers)])

    def forward(self, inputs, hidden_state=None):
        batch_size, _ = inputs.size()
        if hidden_state is None:
            hidden_state = torch.zeros((self.num_rnn_layers, batch_size, self.hidden_state_size),
                                       device=self.device)
        hidden_states = []
        output = inputs
        for layer_num, dcgru_layer in enumerate(self.dcgru_layers):
            next_hidden_state = dcgru_layer(output, hidden_state[layer_num])
            hidden_states.append(next_hidden_state)
            output = next_hidden_state

        return output, torch.stack(hidden_states)


class DecoderModel(nn.Module, Seq2SeqAttrs):
    def __init__(self, adj_mx, nodes, out_seq_len, device, num_rnn_layers=2, rnn_units=64, **model_kwargs):
        nn.Module.__init__(self)
        Seq2SeqAttrs.__init__(self, adj_mx, nodes, num_rnn_layers, rnn_units)
        self.device = device
        self.output_dim = nodes
        self.horizon = out_seq_len
        self.projection_layer = nn.Linear(self.rnn_units, self.output_dim)
        self.dcgru_layers = nn.ModuleList(
            [DCGRUCell(self.rnn_units, adj_mx, self.max_diffusion_step, self.num_nodes,
                       filter_type=self.filter_type, device=device) for _ in range(self.num_rnn_layers)])

    def forward(self, inputs, hidden_state=None):
        hidden_states = []
        output = inputs
        for layer_num, dcgru_layer in enumerate(self.dcgru_layers):
            next_hidden_state = dcgru_layer(output, hidden_state[layer_num])
            hidden_states.append(next_hidden_state)
            output = next_hidden_state

        projected = self.projection_layer(output.view(-1, self.rnn_units))
        output = projected.view(-1, self.num_nodes * self.output_dim)

        return output, torch.stack(hidden_states)


class DCRNNModel(nn.Module, Seq2SeqAttrs):
    def __init__(self, adj_mx, nodes, batch_size, seq_len, out_seq_len, device, **model_kwargs):
        super().__init__()
        Seq2SeqAttrs.__init__(self, adj_mx, nodes)
        self.encoder_model = EncoderModel(adj_mx, seq_len, nodes, device, **model_kwargs)
        self.decoder_model = DecoderModel(adj_mx, nodes, out_seq_len, device, **model_kwargs)
        self.device = device
        self.nodes = nodes
        self.out_seq_len = out_seq_len
        self.cl_decay_steps = 1000
        self.use_curriculum_learning = False

    def _compute_sampling_threshold(self, batches_seen):
        return self.cl_decay_steps / (
                self.cl_decay_steps + np.exp(batches_seen / self.cl_decay_steps))

    def encoder(self, inputs):
        encoder_hidden_state = None
        for t in range(self.encoder_model.seq_len):
            _, encoder_hidden_state = self.encoder_model(inputs[t], encoder_hidden_state)

        return encoder_hidden_state

    def decoder(self, encoder_hidden_state, labels=None, batches_seen=None):
        batch_size = encoder_hidden_state.size(1)
        go_symbol = torch.zeros((batch_size, self.num_nodes * self.decoder_model.output_dim),
                                device=self.device)
        decoder_hidden_state = encoder_hidden_state
        decoder_input = go_symbol

        outputs = []

        for t in range(self.decoder_model.horizon):
            decoder_output, decoder_hidden_state = self.decoder_model(decoder_input,
                                                                      decoder_hidden_state)
            decoder_input = decoder_output
            outputs.append(decoder_output)
            if self.training and self.use_curriculum_learning:
                c = np.random.uniform(0, 1)
                if c < self._compute_sampling_threshold(batches_seen):
                    decoder_input = labels[t]

        outputs = torch.stack(outputs)
        return outputs

    def forward(self, inputs, labels=None, batches_seen=None):
        if inputs.dim() == 3:
            inputs = inputs.permute(1, 0, 2)
        elif inputs.dim() == 4:
            inputs = inputs[:, :, :, 0].permute(1, 0, 2)
        encoder_hidden_state = self.encoder(inputs)
        outputs = self.decoder(encoder_hidden_state, labels, batches_seen=batches_seen)
        return outputs.permute(1, 0, 2)
