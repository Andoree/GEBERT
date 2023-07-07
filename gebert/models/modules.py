from torch import nn as nn
from torch.nn import functional as F
from torch_geometric.nn import SAGEConv, GATv2Conv, DataParallel


class GraphSAGEEncoder(nn.Module):
    def __init__(self, in_channels, num_outer_layers: int, num_inner_layers: int, num_hidden_channels,
                 dropout_p: float, set_out_input_dim_equal: bool = False, parallel=False):
        super().__init__()
        self.convs = nn.ModuleList()
        self.num_outer_layers = num_outer_layers
        self.num_inner_layers = num_inner_layers
        self.num_hidden_channels = num_hidden_channels
        self.dropout_p = dropout_p
        self.set_out_input_dim_equal = set_out_input_dim_equal

        for i in range(num_outer_layers):
            inner_convs = nn.ModuleList()
            for j in range(num_inner_layers):
                input_num_channels = in_channels if (j == 0 and i == 0) else num_hidden_channels

                output_num_channels = num_hidden_channels
                if set_out_input_dim_equal and (i == num_outer_layers - 1) and (j == num_inner_layers - 1):
                    output_num_channels = in_channels

                if parallel:
                    sage_conv = DataParallel(SAGEConv(input_num_channels, output_num_channels))
                else:
                    sage_conv = SAGEConv(input_num_channels, output_num_channels)
                inner_convs.append(sage_conv)

            self.convs.append(inner_convs)

        self.gelu = nn.GELU()

    def forward(self, embs, adjs, *args, **kwargs):
        x = embs
        for i, ((edge_index, _, size), inner_convs_list) in enumerate(zip(adjs, self.convs)):
            for j, conv in enumerate(inner_convs_list):
                x = conv(x, edge_index)
                if not (i == self.num_outer_layers - 1 and j == self.num_inner_layers - 1):
                    x = F.dropout(x, p=self.dropout_p, training=self.training)

                    x = self.gelu(x)

        return x


class GATv2Encoder(nn.Module):
    def __init__(self, in_channels, num_outer_layers: int, num_inner_layers: int, num_hidden_channels, dropout_p: float,
                 num_att_heads: int, attention_dropout_p: float, set_out_input_dim_equal, add_self_loops):
        super().__init__()
        self.num_outer_layers = num_outer_layers

        self.num_inner_layers = num_inner_layers
        self.num_att_heads = num_att_heads
        self.num_hidden_channels = num_hidden_channels
        self.dropout_p = dropout_p
        self.convs = nn.ModuleList()
        for i in range(num_outer_layers):
            inner_convs = nn.ModuleList()
            for j in range(num_inner_layers):
                input_num_channels = in_channels if (j == 0 and i == 0) else num_hidden_channels

                output_num_channels = num_hidden_channels
                if set_out_input_dim_equal and (i == num_outer_layers - 1) and (j == num_inner_layers - 1):
                    output_num_channels = in_channels
                assert output_num_channels % num_att_heads == 0
                gat_head_output_size = output_num_channels // num_att_heads
                gat_conv = GATv2Conv(in_channels=input_num_channels, out_channels=gat_head_output_size,
                                     heads=num_att_heads, dropout=attention_dropout_p,
                                     add_self_loops=add_self_loops, edge_dim=in_channels, share_weights=True)
                inner_convs.append(gat_conv)
            self.convs.append(inner_convs)
        self.gelu = nn.GELU()

    def forward(self, embs, adjs, edge_type_list, batch_size):
        x = embs
        for i, ((edge_index, _, size), inner_convs_list, rel_type) in enumerate(
                zip(adjs, self.convs, edge_type_list)):
            for j, conv in enumerate(inner_convs_list):
                if not (i == self.num_outer_layers - 1 and j == self.num_inner_layers - 1):
                    x = F.dropout(x, p=self.dropout_p, training=self.training)
                    if not self.remove_activations:
                        x = self.gelu(x)
        return x
