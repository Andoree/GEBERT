from abc import ABC
from typing import List, Union
import torch
from torch_geometric.nn import DeepGraphInfomax
from torch_geometric.loader.neighbor_sampler import EdgeIndex

from gebert.models.modules import GATv2Encoder, GraphSAGEEncoder

EPS = 1e-15


class Float32DeepGraphInfomax(DeepGraphInfomax):
    r"""The Deep Graph Infomax model from the
    `"Deep Graph Infomax" <https://arxiv.org/abs/1809.10341>`_
    paper based on user-defined encoder and summary model :math:`\mathcal{E}`
    and :math:`\mathcal{R}` respectively, and a corruption function
    :math:`\mathcal{C}`.

    Args:
        hidden_channels (int): The latent space dimensionality.
        encoder (Module): The encoder module :math:`\mathcal{E}`.
        summary (callable): The readout function :math:`\mathcal{R}`.
        corruption (callable): The corruption function :math:`\mathcal{C}`.
    """

    def __init__(self, hidden_channels, encoder, summary, corruption):
        super(Float32DeepGraphInfomax, self).__init__(hidden_channels=hidden_channels,
                                                      encoder=encoder, summary=summary, corruption=corruption)

    def discriminate(self, z, summary, sigmoid=True):
        r"""Given the patch-summary pair :obj:`z` and :obj:`summary`, computes
        the probability scores assigned to this patch-summary pair.

        This method is the modified version of the implementation from PyTorch Geometric but it calculates 'value'
        variable using torch.float32 datatype instead of torch.float16. Without this fix the model may fail to compute
        DGI loss.

        Args:
            z (Tensor): The latent space.
            summary (Tensor): The summary vector.
            sigmoid (bool, optional): If set to :obj:`False`, does not apply
                the logistic sigmoid function to the output.
                (default: :obj:`True`)
        """
        summary = summary.t() if summary.dim() > 1 else summary
        value = torch.matmul(z, torch.matmul(self.weight, summary)).float()
        return torch.sigmoid(value) if sigmoid else value

    def forward(self, *args, **kwargs):
        """Returns the latent space for the input arguments, their
        corruptions and their summary representation."""
        pos_z = self.encoder(*args, **kwargs)
        cor = self.corruption(*args, **kwargs)
        cor = cor if isinstance(cor, tuple) else (cor,)
        neg_z = self.encoder(*cor, **kwargs)
        summary = self.summary(pos_z, *args, **kwargs)
        return pos_z, neg_z, summary


class AbstractDGIModel(ABC):
    graph_encoder: Union[GraphSAGEEncoder, GATv2Encoder]
    dgi: Float32DeepGraphInfomax

    @staticmethod
    def summary_fn(z, *args, **kwargs):
        batch_size = kwargs.get("batch_size")
        if batch_size is not None:
            z = z[:batch_size]
        return torch.sigmoid(z.mean(dim=0))

    @staticmethod
    def corruption_fn(embs, adjs: List[EdgeIndex], *args, **kwargs):
        corrupted_adjs_list = []
        for adj in adjs:
            edge_index = adj.edge_index
            # size = adj.size
            edge_index_src = edge_index[0]
            edge_index_trg = edge_index[1]
            num_edges = len(edge_index_trg)
            perm_trg_nodes = torch.randperm(num_edges)
            corr_edge_index_trg = edge_index_trg[perm_trg_nodes]
            corr_edge_index = torch.stack((edge_index_src, corr_edge_index_trg)).to(edge_index.device)
            corrupted_adj = EdgeIndex(corr_edge_index, adj.e_id, adj.size).to(edge_index.device)
            corrupted_adjs_list.append(corrupted_adj)
        return embs, corrupted_adjs_list

    def graph_encode(self, text_embed_1, text_embed_2, adjs, batch_size, **kwargs):
        pos_graph_embs_1, neg_graph_embs_1, graph_summary_1 = self.dgi(text_embed_1, adjs, batch_size=batch_size,
                                                                       **kwargs)
        pos_graph_embs_2, neg_graph_embs_2, graph_summary_2 = self.dgi(text_embed_2, adjs, batch_size=batch_size,
                                                                       **kwargs)
        pos_graph_embs_1, neg_graph_embs_1 = pos_graph_embs_1[:batch_size], neg_graph_embs_1[:batch_size]
        pos_graph_embs_2, neg_graph_embs_2 = pos_graph_embs_2[:batch_size], neg_graph_embs_2[:batch_size]
        assert pos_graph_embs_1.size()[0] == neg_graph_embs_1.size()[0] == batch_size
        assert pos_graph_embs_2.size()[0] == neg_graph_embs_2.size()[0] == batch_size

        return pos_graph_embs_1, neg_graph_embs_1, graph_summary_1, pos_graph_embs_2, neg_graph_embs_2, graph_summary_2

    def graph_emb(self, text_embed_1, text_embed_2, adjs, batch_size, **kwargs):
        pos_graph_emb_1 = self.graph_encoder(text_embed_1, adjs, batch_size=batch_size, **kwargs)[:batch_size]
        pos_graph_emb_2 = self.graph_encoder(text_embed_2, adjs, batch_size=batch_size, **kwargs)[:batch_size]
        assert pos_graph_emb_1.size()[0] == pos_graph_emb_2.size()[0] == batch_size

        return pos_graph_emb_1, pos_graph_emb_2
