import logging

import torch.nn as nn
from pytorch_metric_learning import miners, losses
from torch.cuda.amp import autocast

from gebert.models.abstract_dgi_model import Float32DeepGraphInfomax, AbstractDGIModel
from gebert.models.abstract_graphsapbert_model import AbstractGraphSapMetricLearningModel
from gebert.models.modules import GraphSAGEEncoder


class GraphSAGEDGISapMetricLearning(nn.Module, AbstractGraphSapMetricLearningModel, AbstractDGIModel):
    def __init__(self, bert_encoder, use_cuda, loss, graphsage_num_outer_layers, graphsage_num_inner_layers,
                 graphsage_num_hidden_channels, graphsage_dropout_p, dgi_loss_weight, intermodal_loss_weight,
                 multigpu_flag, use_intermodal_miner=True, intermodal_miner_margin=0.2, use_miner=True,
                 miner_margin=0.2, type_of_triplets="all", agg_mode="cls",
                 sapbert_loss_weight: float = 1.0, graph_loss_weight=0.0, ):

        logging.info(
            "Sap_Metric_Learning! use_cuda={} loss={} use_miner={} miner_margin={} type_of_triplets={} agg_mode={}".format(
                use_cuda, loss, use_miner, miner_margin, type_of_triplets, agg_mode
            ))
        super(GraphSAGEDGISapMetricLearning, self).__init__()
        self.bert_encoder = bert_encoder
        self.use_cuda = use_cuda
        self.loss = loss
        self.use_miner = use_miner
        self.use_intermodal_miner = use_intermodal_miner
        self.miner_margin = miner_margin
        self.agg_mode = agg_mode
        self.convs = nn.ModuleList()
        self.bert_hidden_dim = bert_encoder.config.hidden_size
        self.graph_loss_weight = graph_loss_weight
        self.sapbert_loss_weight = sapbert_loss_weight
        self.dgi_loss_weight = dgi_loss_weight
        self.intermodal_loss_weight = intermodal_loss_weight

        if self.use_intermodal_miner:
            self.intermodal_miner = miners.TripletMarginMiner(margin=intermodal_miner_margin,
                                                              type_of_triplets=type_of_triplets)
        else:
            self.intermodal_miner = None
        self.intermodal_loss = losses.MultiSimilarityLoss(alpha=2, beta=50, base=0.5)

        if multigpu_flag:
            self.bert_encoder = nn.DataParallel(bert_encoder)
        else:
            self.bert_encoder = bert_encoder
        self.graphsage_dropout_p = graphsage_dropout_p
        self.graph_encoder = GraphSAGEEncoder(in_channels=self.bert_hidden_dim,
                                              num_outer_layers=graphsage_num_outer_layers,
                                              num_inner_layers=graphsage_num_inner_layers,
                                              dropout_p=graphsage_dropout_p,
                                              num_hidden_channels=graphsage_num_hidden_channels,
                                              set_out_input_dim_equal=True)

        self.dgi = Float32DeepGraphInfomax(
            hidden_channels=self.bert_hidden_dim, encoder=self.graphhsage_encoder,
            summary=self.summary_fn, corruption=self.corruption_fn)
        self.dgi_loss_weight = dgi_loss_weight

        if self.use_miner:
            self.miner = miners.TripletMarginMiner(margin=miner_margin, type_of_triplets=type_of_triplets)
        else:
            self.miner = None

        if self.loss == "ms_loss":
            self.loss = losses.MultiSimilarityLoss(alpha=2, beta=50, base=0.5)  # 1,2,3; 40,50,60
        elif self.loss == "circle_loss":
            self.loss = losses.CircleLoss()
        elif self.loss == "triplet_loss":
            self.loss = losses.TripletMarginLoss()
        elif self.loss == "infoNCE":
            self.loss = losses.NTXentLoss(temperature=0.07)  # The MoCo paper uses 0.07, while SimCLR uses 0.5.
        elif self.loss == "lifted_structure_loss":
            self.loss = losses.LiftedStructureLoss()
        elif self.loss == "nca_loss":
            self.loss = losses.NCALoss()
        logging.info(f"Using miner: {self.miner}")
        logging.info(f"Using loss function: {self.loss}")

    @autocast()
    def forward(self, term_1_input_ids, term_1_att_masks, term_2_input_ids, term_2_att_masks, concept_ids,
                adjs, batch_size):
        """
        query : (N, h), candidates : (N, topk, h)

        output : (N, topk)
        """
        text_loss, text_embed_1, text_embed_2, hard_pairs = \
            self.calc_text_loss_return_text_embeddings(term_1_input_ids, term_1_att_masks,
                                                       term_2_input_ids, term_2_att_masks, concept_ids, batch_size)

        pos_graph_embs_1, neg_graph_embs_1, graph_summary_1, pos_graph_embs_2, neg_graph_embs_2, graph_summary_2 = \
            self.graph_encode(text_embed_1, text_embed_2, adjs=adjs, batch_size=batch_size)

        dgi_loss_1 = self.dgi.loss(pos_graph_embs_1, neg_graph_embs_1, graph_summary_1)
        dgi_loss_2 = self.dgi.loss(pos_graph_embs_2, neg_graph_embs_2, graph_summary_2)
        if not self.common_hard_pairs:
            graph_loss, hard_pairs = self.calculate_sapbert_loss(pos_graph_embs_1[:batch_size],
                                                                 pos_graph_embs_2[:batch_size],
                                                                 concept_ids[:batch_size], )

            intermodal_loss = self.calculate_intermodal_loss(text_embed_1, text_embed_2, pos_graph_embs_1,
                                                             pos_graph_embs_2,
                                                             concept_ids, batch_size, )
        else:
            graph_loss, hard_pairs = self.calculate_sapbert_loss(pos_graph_embs_1[:batch_size],
                                                                 pos_graph_embs_2[:batch_size],
                                                                 concept_ids[:batch_size], hard_pairs=hard_pairs)

            intermodal_loss = self.calculate_intermodal_loss(text_embed_1, text_embed_2, pos_graph_embs_1,
                                                             pos_graph_embs_2,
                                                             concept_ids, batch_size, hard_pairs=hard_pairs)

        return text_loss, graph_loss, (dgi_loss_1 + dgi_loss_2) / 2, intermodal_loss
