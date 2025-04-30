import math, os
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch_sparse import SparseTensor, matmul
from torch_geometric.utils import degree


class AttentionMechanism(nn.Module):
    """Base attention mechanism class for computing attention operations"""

    def __init__(self):
        super(AttentionMechanism, self).__init__()

    def forward(self, qs, ks, vs, output_attn=False):
        """
        qs: query tensor [N, H, M]
        ks: key tensor [L, H, M]
        vs: value tensor [L, H, D]
        return output [N, H, D]
        """
        # Normalize input
        qs = qs / torch.norm(qs, p=2)  # [N, H, M]
        ks = ks / torch.norm(ks, p=2)  # [L, H, M]
        N = qs.shape[0]

        # Numerator
        kvs = torch.einsum("lhm,lhd->hmd", ks, vs)
        attention_num = torch.einsum("nhm,hmd->nhd", qs, kvs)  # [N, H, D]
        all_ones = torch.ones([vs.shape[0]]).to(vs.device)
        vs_sum = torch.einsum("l,lhd->hd", all_ones, vs)  # [H, D]
        attention_num += vs_sum.unsqueeze(0).repeat(vs.shape[0], 1, 1)  # [N, H, D]

        # Denominator
        all_ones = torch.ones([ks.shape[0]]).to(ks.device)
        ks_sum = torch.einsum("lhm,l->hm", ks, all_ones)
        attention_normalizer = torch.einsum("nhm,hm->nh", qs, ks_sum)  # [N, H]

        # Attentive aggregated results
        attention_normalizer = torch.unsqueeze(attention_normalizer, len(attention_normalizer.shape))  # [N, H, 1]
        attention_normalizer += torch.ones_like(attention_normalizer) * N
        attn_output = attention_num / attention_normalizer  # [N, H, D]

        # Compute attention for visualization if needed
        if output_attn:
            attention = torch.einsum("nhm,lhm->nlh", qs, ks) / attention_normalizer.unsqueeze(2)  # [N, L, H]
            return attn_output, attention
        else:
            return attn_output


class GraphConvolution(nn.Module):
    """Graph convolution module for processing graph-structured data"""

    def __init__(self):
        super(GraphConvolution, self).__init__()

    def forward(self, x, edge_index, edge_weight=None):
        """
        x: node features [N, H, D] or [N, D]
        edge_index: edge indices [2, E]
        edge_weight: edge weights [E]
        """
        # Handle input dimensions
        if len(x.shape) == 2:
            x = x.unsqueeze(1)
            squeeze_output = True
        else:
            squeeze_output = False

        N, H = x.shape[0], x.shape[1]
        row, col = edge_index

        # Calculate degree normalization
        d = degree(col, N).float()
        d_norm_in = (1. / d[col]).sqrt()
        d_norm_out = (1. / d[row]).sqrt()

        # Set edge weights
        if edge_weight is None:
            value = torch.ones_like(row) * d_norm_in * d_norm_out
        else:
            value = edge_weight * d_norm_in * d_norm_out

        # Handle anomalous values
        value = torch.nan_to_num(value, nan=0.0, posinf=0.0, neginf=0.0)

        # Build sparse adjacency matrix
        adj = SparseTensor(row=col, col=row, value=value, sparse_sizes=(N, N))

        # Apply convolution to each head
        gcn_conv_output = []
        for i in range(H):
            gcn_conv_output.append(matmul(adj, x[:, i]))  # [N, D]

        # Combine results
        gcn_conv_output = torch.stack(gcn_conv_output, dim=1)  # [N, H, D]

        # If input is 2D, output should also be 2D
        if squeeze_output:
            gcn_conv_output = gcn_conv_output.squeeze(1)

        return gcn_conv_output


class EWGNNConv(nn.Module):
    """EWGNN convolution layer that combines attention with graph convolution"""

    def __init__(self, in_channels,
                 out_channels,
                 num_heads,
                 use_graph=True,
                 use_weight=True):
        super(EWGNNConv, self).__init__()
        self.Wk = nn.Linear(in_channels, out_channels * num_heads)
        self.Wq = nn.Linear(in_channels, out_channels * num_heads)
        if use_weight:
            self.Wv = nn.Linear(in_channels, out_channels * num_heads)

        self.out_channels = out_channels
        self.num_heads = num_heads
        self.use_graph = use_graph
        self.use_weight = use_weight

        # Instantiate submodules
        self.attention = AttentionMechanism()
        self.graph_conv = GraphConvolution()

    def reset_parameters(self):
        self.Wk.reset_parameters()
        self.Wq.reset_parameters()
        if self.use_weight:
            self.Wv.reset_parameters()

    def forward(self, query_input, source_input, edge_index=None, edge_weight=None, output_attn=False):
        # Feature transformation
        query = self.Wq(query_input).reshape(-1, self.num_heads, self.out_channels)
        key = self.Wk(source_input).reshape(-1, self.num_heads, self.out_channels)
        if self.use_weight:
            value = self.Wv(source_input).reshape(-1, self.num_heads, self.out_channels)
        else:
            value = source_input.reshape(-1, 1, self.out_channels)

        # Compute attention aggregation
        if output_attn:
            attention_output, attn = self.attention(query, key, value, output_attn=True)
        else:
            attention_output = self.attention(query, key, value)

        # Use input graph for GCN convolution if enabled
        if self.use_graph:
            final_output = attention_output + self.graph_conv(value, edge_index, edge_weight)
        else:
            final_output = attention_output

        final_output = final_output.mean(dim=1)

        if output_attn:
            return final_output, attn
        else:
            return final_output


class AdaptiveLayer(nn.Module):
    """Energy-based adaptive layer that adjusts the model's wave parameter eta"""

    def __init__(self, target_ratio=1.0, hidden_dim=16, initial_eta=1.0):
        super(AdaptiveLayer, self).__init__()
        self.target_ratio = target_ratio
        self.initial_eta = initial_eta
        self.fc = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )

    def forward(self, H_prev, H_curr, G):
        row, col = G
        # Calculate kinetic energy
        E_kin = 0.5 * torch.norm(H_curr - H_prev, p='fro') ** 2

        # Calculate potential energy
        diff = H_curr[row] - H_curr[col]  # [E, D]
        E_pot = 0.5 * torch.sum(G * torch.norm(diff, dim=1) ** 2)

        # Calculate ratio
        r = E_kin / (E_pot + 1e-8)  # Avoid division by zero
        delta_r = r - self.target_ratio

        # Learn eta coefficient
        scale = self.fc(delta_r.unsqueeze(0)).squeeze()
        eta = self.initial_eta * scale

        return eta


class EWGNN(nn.Module):
    """
    Energy-Wave Graph Neural Network (EWGNN) model

    This model combines attention mechanisms with graph convolution
    and implements wave dynamics for feature propagation.
    """

    def __init__(self, in_channels, hidden_channels, out_channels,
                 num_layers=2, num_heads=1, alpha=0.5, sigma=0.5,
                 dropout=0.5, use_bn=True, use_residual=True,
                 use_weight=True, use_graph=True,
                 eta_mode='adaptive', fixed_eta=0.1):
        super(EWGNN, self).__init__()
        self.eta_mode = eta_mode
        self.fixed_eta = fixed_eta  # Default fixed eta value

        # Input layer
        self.input_fc = nn.Linear(in_channels, hidden_channels)

        # EWGNN convolution layers and batch normalization
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()

        # Add batch norm for input if requested
        self.bns.append(nn.LayerNorm(hidden_channels) if use_bn else nn.Identity())

        for i in range(num_layers):
            self.convs.append(
                EWGNNConv(hidden_channels, hidden_channels, num_heads=num_heads,
                          use_graph=use_graph, use_weight=use_weight))
            self.bns.append(nn.LayerNorm(hidden_channels) if use_bn else nn.Identity())

        # Output layer
        self.output_fc = nn.Linear(hidden_channels, out_channels)

        # Adaptive layer
        if eta_mode == 'adaptive':
            self.energy_layer = AdaptiveLayer()  # Initialize AdaptiveLayer if eta_mode is 'adaptive'

        # Model parameters
        self.dropout = dropout
        self.activation = F.relu
        self.use_bn = use_bn
        self.residual = use_residual
        self.alpha = alpha
        self.sigma = sigma

    def reset_parameters(self):
        self.input_fc.reset_parameters()
        for bn in self.bns:
            if isinstance(bn, nn.LayerNorm):
                bn.reset_parameters()
        for conv in self.convs:
            conv.reset_parameters()
        self.output_fc.reset_parameters()

    def forward(self, x, edge_index, edge_weight=None, return_embeddings=False):
        layer_ = []
        embeddings = []

        # Input MLP layer
        x = self.input_fc(x)
        if self.use_bn:
            x = self.bns[0](x)
        x = self.activation(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        # Store for residual connection
        layer_.append(x)
        embeddings.append(x)

        # Define prev_x for wave dynamics
        prev_x = torch.zeros_like(x)

        # Process through EWGNN layers
        for i, conv in enumerate(self.convs):
            # Graph convolution with EWGNN layer
            L_hl = conv(x, x, edge_index, edge_weight)

            # Calculate eta
            if self.eta_mode == 'adaptive':
                eta = self.energy_layer(prev_x, x, edge_index)  # Dynamic eta
            elif self.eta_mode == 'fixed':
                eta = self.fixed_eta  # Fixed eta
            else:
                eta = 0.0  # Default no wave

            # Apply wave dynamics and update step with eta
            out = L_hl + eta * (x - prev_x)

            # Apply residual connection if enabled
            if self.residual:
                out = self.alpha * out + (1 - self.alpha) * layer_[i]

            # Apply batch normalization if enabled
            if self.use_bn:
                out = self.bns[i + 1](out)

            # Apply dropout
            out = F.dropout(out, p=self.dropout, training=self.training)

            # Update states
            layer_.append(out)
            embeddings.append(out)
            prev_x = x
            x = out

        # Output MLP layer
        x_out = self.output_fc(x)

        if return_embeddings:
            return x_out, embeddings  # Return predictions and all layer embeddings
        return x_out