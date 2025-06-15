import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np

from torch_geometric.nn import MessagePassing, GCNConv,SAGPooling, GATConv, LayerNorm
from torch_geometric.utils import add_self_loops, degree, softmax
from torch_geometric.nn import global_add_pool, global_mean_pool, global_max_pool,GCNConv
from torch_geometric.utils.num_nodes import maybe_num_nodes
import torch
import torch.nn as nn

import math
from torch_geometric.nn.resolver import activation_resolver
from typing import List, Union, Optional, Dict, Any
from torch import Tensor
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
import math

from torch.nn.modules.container import ModuleList
from torch_geometric.utils import softmax
from torch_geometric.nn.aggr import MultiAggregation

class MLP_OUT(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        num_layers = 3
        self.fc1_1 = nn.Linear(input_dim, hidden_dim)
        self.layer_list = nn.ModuleList([])
        for i in range(num_layers):
            self.layer_list.append(mlp_layer(hidden_dim, hidden_dim))

        self.fc1_2 = nn.Linear(hidden_dim, output_dim)
        self.LN_1 = nn.LayerNorm(hidden_dim)
        self.LN_2 = nn.LayerNorm(output_dim)
        self.act = nn.ELU()

    def forward(self, x):
        x = self.act(self.fc1_1(x))
        x = self.LN_1(x)
        for layer in self.layer_list:
            x = layer(x)
        x = self.LN_2(self.act(self.fc1_2(x)))
        return x

class mlp_layer(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        dropout = 0.1
        self.fc1 = nn.Linear(input_dim,output_dim)
        self.bias = nn.Parameter(torch.zeros(output_dim))
        self.ln = nn.LayerNorm(output_dim)
        self.act = nn.ELU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.act(self.fc1(x))
        x = self.ln(x)+ self.bias
        x = self.dropout(x)
        return x


num_atom_type = 119  # including the extra mask tokens
num_chirality_tag = 3

num_bond_type = 5  # including aromatic and self-loop edge
num_bond_direction = 3
class model(nn.Module):
    def __init__(self,kge_dim=512, rel_total=86, n_heads=4,n_blocks=4):
        super().__init__()

        self.rel_total = rel_total
        self.kge_dim = kge_dim
        self.n_blocks = n_blocks

        self.x_embedding1 = nn.Embedding(num_atom_type, kge_dim)
        self.x_embedding2 = nn.Embedding(num_chirality_tag, kge_dim)
        nn.init.xavier_uniform_(self.x_embedding1.weight.data)
        nn.init.xavier_uniform_(self.x_embedding2.weight.data)

        self.edge_embedding1 = nn.Embedding(num_bond_type, kge_dim)
        self.edge_embedding2 = nn.Embedding(num_bond_direction, kge_dim)
        nn.init.xavier_uniform_(self.edge_embedding1.weight.data)
        nn.init.xavier_uniform_(self.edge_embedding2.weight.data)

        self.initial_norm = LayerNorm(kge_dim)

        self.blocks = []
        self.net_norms = ModuleList()
        self.tf_list = nn.ModuleList([])
        for i in range(self.n_blocks):
            block = MVN_DDI_Block(n_heads, kge_dim,kge_dim//n_heads)
            self.add_module(f"block{i}", block)
            self.blocks.append(block)

        for i in range(1):
            encoder_layer = nn.TransformerEncoderLayer(d_model=kge_dim, nhead=n_heads, dim_feedforward=kge_dim*2, dropout=0.1)
            transformer_encoder_fusion = nn.TransformerEncoder(encoder_layer, num_layers=4)
            self.tf_list.append(transformer_encoder_fusion)



        self.out_linear = MLP_OUT(kge_dim*2, kge_dim//2, rel_total)

    def forward(self, triple):
        h_unimol_samples,h_data,  t_unimol_samples, t_data= triple
        h_unimol_samples = h_unimol_samples.unsqueeze(1)
        t_unimol_samples = t_unimol_samples.unsqueeze(1)
        ##################### pre-processing ################################
        h_data.x = self.x_embedding1(h_data.x[:, 0]) + self.x_embedding2(h_data.x[:, 1])
        t_data.x = self.x_embedding1(t_data.x[:, 0]) + self.x_embedding2(t_data.x[:, 1])
        h_data.x = self.initial_norm(h_data.x)
        t_data.x = self.initial_norm(t_data.x)
        h_data.edge_attr = self.edge_embedding1(h_data.edge_attr[:, 0]) + self.edge_embedding2(h_data.edge_attr[:, 1])
        t_data.edge_attr = self.edge_embedding1(t_data.edge_attr[:, 0]) + self.edge_embedding2(t_data.edge_attr[:, 1])
        all_h_weights = []
        all_t_weights = []
        for i, block in enumerate(self.blocks):
            h_data,t_data = block(h_data, t_data)
            all_h_weights.append(h_data.atom_weights)  # 收集每层的原子权重
            all_t_weights.append(t_data.atom_weights)

        h_global_graph_emb = F.relu(global_mean_pool(h_data.x, h_data.batch))
        t_global_graph_emb = F.relu(global_mean_pool(t_data.x, t_data.batch))

        h_global_graph_emb = h_global_graph_emb.unsqueeze(1)
        t_global_graph_emb = t_global_graph_emb.unsqueeze(1)
        ############################ fusion #############################################
        # drug1 = h_unimol_samples
        # drug2 = t_unimol_samples
        # all_h_weights = 0
        # all_t_weights = 0
        # drug1 = h_global_graph_emb
        # drug2 = t_global_graph_emb
        drug1 = torch.cat([h_unimol_samples, h_global_graph_emb], dim=1)
        drug2 = torch.cat([t_unimol_samples, t_global_graph_emb], dim=1)
        drug1 = drug1.permute(1,0,2)
        drug2 = drug2.permute(1,0,2)
        for i in range(1):
            drug1 = self.tf_list[i](drug1)
            drug2 = self.tf_list[i](drug2)
        drug1 = drug1.sum(dim=0)
        drug2 = drug2.sum(dim=0)

        emb = torch.cat([drug1, drug2], dim=1)
        ############################ prediction #########################################
        scores = self.out_linear(emb)
        return scores, all_h_weights, all_t_weights

    def save_model(self, file_path):
        # 保存模型的状态字典
        torch.save(self.state_dict(), file_path)
        print(f"Model saved to {file_path}")

    def load_model(self, file_path):
        # 加载模型的状态字典
        self.load_state_dict(torch.load(file_path))
        print(f"Model loaded from {file_path}")

    def process(self, data):
        pass




class MVN_DDI_Block(nn.Module):
    def __init__(self, n_heads, in_features, head_out_feats):
        super().__init__()
        self.n_heads = n_heads
        self.in_features = in_features
        self.out_features = head_out_feats

        self.feature_conv = GATConv(in_features, head_out_feats, heads=n_heads)
        self.intraAtt = GTConv(in_features,in_features,in_features)
        self.norm = LayerNorm(in_features)


    def forward(self, h_data, t_data):
        h_data.x = F.elu(self.feature_conv(h_data.x, h_data.edge_index, h_data.edge_attr))
        t_data.x = F.elu(self.feature_conv(t_data.x, t_data.edge_index, t_data.edge_attr))

        h_data = self.intraAtt(h_data)
        t_data = self.intraAtt(t_data)

        h_data.x = F.elu(self.norm(h_data.x, h_data.batch))
        t_data.x = F.elu(self.norm(t_data.x, t_data.batch))

        return h_data, t_data


class GTConv(MessagePassing):
    def __init__(
        self,
        node_in_dim: int,
        hidden_dim: int,
        edge_in_dim: int,
        num_heads: int = 4,
        gate=True,
        qkv_bias=False,
        dropout: float = 0.0,
        norm: str = "ln",
        act: str = "relu",
        aggregators: List[str] = ["sum"],
    ):
        """
        Graph Transformer Convolution (GTConv) module.

        Args:
            node_in_dim (int): Dimensionality of the input node features.
            hidden_dim (int): Dimensionality of the hidden representations.
            edge_in_dim (int, optional): Dimensionality of the input edge features.
                                         Default is None.
            num_heads (int, optional): Number of attention heads. Default is 8.
            dropout (float, optional): Dropout probability. Default is 0.0.
            gate (bool, optional): Use a gate attantion mechanism.
                                   Default is False
            qkv_bias (bool, optional): Bias in the attention mechanism.
                                       Default is False
            norm (str, optional): Normalization type. Options: "bn" (BatchNorm), "ln" (LayerNorm).
                                  Default is "bn".
            act (str, optional): Activation function name. Default is "relu".
            aggregators (List[str], optional): Aggregation methods for the messages aggregation.
                                               Default is ["sum"].
        """
        super().__init__(node_dim=0, aggr=MultiAggregation(aggregators, mode="cat"))

        assert (
            "sum" in aggregators
        )  # makes sure that the original sum_j is always part of the message passing
        assert hidden_dim % num_heads == 0
        assert (edge_in_dim is None) or (edge_in_dim > 0)

        self.aggregators = aggregators
        self.num_aggrs = len(aggregators)

        self.WQ = nn.Linear(node_in_dim, hidden_dim, bias=qkv_bias)
        self.WK = nn.Linear(node_in_dim, hidden_dim, bias=qkv_bias)
        self.WV = nn.Linear(node_in_dim, hidden_dim, bias=qkv_bias)
        self.WO = nn.Linear(hidden_dim * self.num_aggrs, node_in_dim, bias=True)

        if edge_in_dim is not None:
            self.WE = nn.Linear(edge_in_dim, hidden_dim, bias=True)
            self.WOe = nn.Linear(hidden_dim, edge_in_dim, bias=True)
            self.ffn_e = MLP(
                input_dim=edge_in_dim,
                output_dim=edge_in_dim,
                hidden_dims=hidden_dim,
                num_hidden_layers=1,
                dropout=dropout,
                act=act,
            )
            if norm.lower() in ["bn", "batchnorm", "batch_norm"]:
                self.norm1e = nn.BatchNorm1d(edge_in_dim)
                self.norm2e = nn.BatchNorm1d(edge_in_dim)
            elif norm.lower() in ["ln", "layernorm", "layer_norm"]:
                self.norm1e = nn.LayerNorm(edge_in_dim)
                self.norm2e = nn.LayerNorm(edge_in_dim)
            else:
                raise ValueError
        else:
            assert gate is False
            self.WE = self.register_parameter("WE", None)
            self.WOe = self.register_parameter("WOe", None)
            self.ffn_e = self.register_parameter("ffn_e", None)
            self.norm1e = self.register_parameter("norm1e", None)
            self.norm2e = self.register_parameter("norm2e", None)

        if norm.lower() in ["bn", "batchnorm", "batch_norm"]:
            self.norm1 = nn.BatchNorm1d(node_in_dim)
            self.norm2 = nn.BatchNorm1d(node_in_dim)
        elif norm.lower() in ["ln", "layernorm", "layer_norm"]:
            self.norm1 = nn.LayerNorm(node_in_dim)
            self.norm2 = nn.LayerNorm(node_in_dim)

        if gate:
            self.n_gate = nn.Linear(node_in_dim, hidden_dim, bias=True)
            self.e_gate = nn.Linear(edge_in_dim, hidden_dim, bias=True)
        else:
            self.n_gate = self.register_parameter("n_gate", None)
            self.e_gate = self.register_parameter("e_gate", None)

        self.dropout_layer = nn.Dropout(p=dropout)

        self.ffn = MLP(
            input_dim=node_in_dim,
            output_dim=node_in_dim,
            hidden_dims=hidden_dim,
            num_hidden_layers=1,
            dropout=dropout,
            act=act,
        )

        self.num_heads = num_heads
        self.node_in_dim = node_in_dim
        self.edge_in_dim = edge_in_dim
        self.hidden_dim = hidden_dim
        self.dropout = dropout
        self.norm = norm.lower()
        self.gate = gate
        self.qkv_bias = qkv_bias
        self.alpha = None
        self.reset_parameters()


    def reset_parameters(self):
        """
        Note: The output of the Q-K-V layers does not pass through the activation layer (as opposed to the input),
              so the variance estimation should differ by a factor of two from the default
              kaiming_uniform initialization.
        """
        nn.init.xavier_uniform_(self.WQ.weight)
        nn.init.xavier_uniform_(self.WK.weight)
        nn.init.xavier_uniform_(self.WV.weight)
        nn.init.xavier_uniform_(self.WO.weight)
        if self.edge_in_dim is not None:
            nn.init.xavier_uniform_(self.WE.weight)
            nn.init.xavier_uniform_(self.WOe.weight)

    def forward(self, data):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        x_ = x
        edge_attr_ = edge_attr

        Q = self.WQ(x).view(-1, self.num_heads, self.hidden_dim // self.num_heads)
        K = self.WK(x).view(-1, self.num_heads, self.hidden_dim // self.num_heads)
        V = self.WV(x).view(-1, self.num_heads, self.hidden_dim // self.num_heads)
        if self.gate:
            G = self.n_gate(x).view(
                -1, self.num_heads, self.hidden_dim // self.num_heads
            )
        else:
            G = torch.ones_like(V)  # G*V = V

        out = self.propagate(
            edge_index, Q=Q, K=K, V=V, G=G, edge_attr=edge_attr, size=None
        )
        out = out.view(-1, self.hidden_dim * self.num_aggrs)  # concatenation

        # 提取 alpha 并处理原子权重
        alpha = self.alpha  # 形状为 (num_edges, num_heads)
        # print(global_alpha)
        atom_weights = torch.zeros(x.size(0), device=x.device)  # 初始化原子权重张量

        # 聚合来自边的权重到节点
        src_nodes = edge_index[0]  # 边的源节点索引
        atom_weights.scatter_add_(0, src_nodes, alpha.mean(dim=1))  # 多头注意力取平均
        # print(atom_weights.shape)
        # NODES
        out = self.dropout_layer(out)
        out = self.WO(out) + x_
        out = self.norm1(out)
        # FFN--nodes
        ffn_in = out
        out = self.ffn(out)
        out = self.norm2(ffn_in + out)

        if self.edge_in_dim is None:
            out_eij = None
        else:
            out_eij = self._eij
            self._eij = None
            out_eij = out_eij.view(-1, self.hidden_dim)

            # EDGES
            out_eij = self.dropout_layer(out_eij)
            out_eij = self.WOe(out_eij) + edge_attr_  # Residual connection
            out_eij = self.norm1e(out_eij)
            # FFN--edges
            ffn_eij_in = out_eij
            out_eij = self.ffn_e(out_eij)
            out_eij = self.norm2e(ffn_eij_in + out_eij)
        data.edge_attr = out_eij
        data.x = out
        data.atom_weights = atom_weights
        return data

    def message(self, Q_i, K_j, V_j, G_j, index, edge_attr=None):
        d_k = Q_i.size(-1)
        qijk = (Q_i * K_j) / math.sqrt(d_k)
        if self.edge_in_dim is not None:
            assert edge_attr is not None
            E = self.WE(edge_attr).view(-1, self.num_heads, self.hidden_dim // self.num_heads)
            qijk = E * qijk
            self._eij = qijk
        else:
            self._eij = None

        if self.gate:
            assert edge_attr is not None
            e_gate = self.e_gate(edge_attr).view(-1, self.num_heads, self.hidden_dim // self.num_heads)
            qijk = torch.mul(qijk, torch.sigmoid(e_gate))

        qijk = (Q_i * K_j).sum(dim=-1) / math.sqrt(d_k)

        alpha = softmax(qijk, index)  # Log-Sum-Exp trick used. No need for clipping (-5,5)
        self.alpha = alpha.detach()
        if self.gate:
            V_j_g = torch.mul(V_j, torch.sigmoid(G_j))
        else:
            V_j_g = V_j

        return alpha.view(-1, self.num_heads, 1) * V_j_g

    def __repr__(self) -> str:
        aggrs = ",".join(self.aggregators)
        return (
            f"{self.__class__.__name__}({self.node_in_dim}, "
            f"{self.hidden_dim}, heads={self.num_heads}, "
            f"aggrs: {aggrs}, "
            f"qkv_bias: {self.qkv_bias}, "
            f"gate: {self.gate})"
        )

class MLP(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dims: Union[int, List[int]],
        num_hidden_layers: int = 1,
        dropout: float = 0.0,
        act: str = "relu",
        act_kwargs: Optional[Dict[str, Any]] = None,
    ):
        """
        Multi-Layer Perceptron (MLP) module.

        Args:
            input_dim (int): Dimensionality of the input features.
            output_dim (int): Dimensionality of the output features.
            hidden_dims (Union[int, List[int]]): Hidden layer dimensions.
                If int, same hidden dimension is used for all layers.
            num_hidden_layers (int, optional): Number of hidden layers. Default is 1.
            dropout (float, optional): Dropout probability. Default is 0.0.
            act (str, optional): Activation function name. Default is "relu".
            act_kwargs (Dict[str, Any], optional): Additional arguments for the activation function.
                                                   Default is None.
        """
        super(MLP, self).__init__()

        if isinstance(hidden_dims, int):
            hidden_dims = [hidden_dims] * num_hidden_layers

        assert len(hidden_dims) == num_hidden_layers

        hidden_dims = [input_dim] + hidden_dims
        layers = []

        for i_dim, o_dim in zip(hidden_dims[:-1], hidden_dims[1:]):
            layers.append(nn.Linear(i_dim, o_dim, bias=True))
            layers.append(activation_resolver(act, **(act_kwargs or {})))
            if dropout > 0.0:
                layers.append(nn.Dropout(p=dropout))

        layers.append(nn.Linear(hidden_dims[-1], output_dim, bias=True))
        self.mlp = nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass of the MLP module.

        Args:
            x (Any): Input tensor.

        Returns:
            Any: Output tensor.
        """
        return self.mlp(x)
