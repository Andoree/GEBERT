import codecs
import logging
import random
from typing import List, Dict
from typing import Tuple, Set

import torch
from torch.utils.data import Dataset
from torch_cluster import random_walk
from torch_geometric.loader import NeighborSampler as RawNeighborSampler
from torch_geometric.utils.num_nodes import maybe_num_nodes
from torch_sparse import SparseTensor
from transformers import AutoTokenizer, AutoModel, BertTokenizerFast

from gebert.utils.data_utils import node_ids2tokenizer_output
from gebert.utils.io import load_node_id2terms_list, load_edges_tuples, load_adjacency_list


def tokenize_node_terms(node_id_to_terms_dict, tokenizer, max_length: int) -> Dict[int, List[List[int]]]:
    node_id_to_token_ids_dict = {}
    for node_id, terms_list in node_id_to_terms_dict.items():
        node_tokenized_terms = []
        for term in terms_list:
            tokenizer_output = tokenizer.encode_plus(term, max_length=max_length,
                                                     padding="max_length", truncation=True, add_special_tokens=True,
                                                     return_tensors="pt", )
            node_tokenized_terms.append(tokenizer_output)
        node_id_to_token_ids_dict[node_id] = node_tokenized_terms
    return node_id_to_token_ids_dict


class NeighborSampler(RawNeighborSampler):
    def __init__(self, node_id_to_token_ids_dict, seq_max_length, random_walk_length: int, *args, **kwargs):
        super(NeighborSampler, self).__init__(*args, **kwargs)
        self.node_id_to_token_ids_dict = node_id_to_token_ids_dict
        self.seq_max_length = seq_max_length
        self.random_walk_length = random_walk_length
        self.num_nodes = kwargs["num_nodes"]

    def __len__(self):
        return self.num_nodes

    def sample(self, batch):
        if not isinstance(batch, torch.Tensor):
            batch = torch.tensor(batch)
        row, col, _ = self.adj_t.coo()

        # For each node in `batch`, we sample a direct neighbor (as positive
        # example) and a random node (as negative example):
        pos_batch = random_walk(row, col, batch, walk_length=self.random_walk_length, coalesced=False)[:, 1]

        neg_batch = torch.randint(0, self.adj_t.size(1), (batch.numel(),), dtype=torch.long)

        batch = torch.cat([batch, pos_batch, neg_batch], dim=0)

        (batch_size, n_id, adjs) = super(NeighborSampler, self).sample(batch)
        batch_input_ids = []
        batch_attention_masks = []
        for idx in n_id:
            idx = idx.item()
            tokenized_terms_list = self.node_id_to_token_ids_dict[idx]
            selected_term_tokenizer_output = random.choice(tokenized_terms_list)
            input_ids = selected_term_tokenizer_output["input_ids"][0]
            attention_mask = selected_term_tokenizer_output["attention_mask"][0]
            assert len(input_ids) == len(attention_mask) == self.seq_max_length
            batch_input_ids.append(input_ids)
            batch_attention_masks.append(attention_mask)
        batch_input_ids = torch.stack(batch_input_ids)
        batch_attention_masks = torch.stack(batch_attention_masks)

        return batch_size, n_id, adjs, batch_input_ids, batch_attention_masks


def convert_edges_tuples_to_edge_index(edges_tuples: List[Tuple[int, int]], remove_selfloops=False) -> torch.Tensor:
    logging.info("Converting edge tuples to edge index")
    # edge_index = torch.zeros(size=[2, len(edges_tuples)], dtype=torch.long)
    edge_strs_set = set()
    for idx, (node_id_1, node_id_2) in enumerate(edges_tuples):
        if node_id_2 < node_id_1:
            node_id_1, node_id_2 = node_id_2, node_id_1
        if not (remove_selfloops and node_id_1 == node_id_2):
            edge_str = f"{node_id_1}~{node_id_2}"
            edge_strs_set.add(edge_str)
    edge_index = torch.zeros(size=[2, len(edge_strs_set) * 2], dtype=torch.long)
    for idx, edge_str in enumerate(edge_strs_set):
        ids = edge_str.split('~')
        node_id_1 = int(ids[0])
        node_id_2 = int(ids[1])
        if not (remove_selfloops and node_id_1 == node_id_2):
            edge_index[0][idx] = node_id_1
            edge_index[1][idx] = node_id_2
            edge_index[0][len(edge_strs_set) + idx] = node_id_2
            edge_index[1][len(edge_strs_set) + idx] = node_id_1

    logging.info(f"Edge index is created. The size is {edge_index.size()}, there are {edge_index.max()} nodes")

    return edge_index


def create_one_hop_adjacency_lists(num_nodes: int, edge_index):
    adjacency_lists = [set() for _ in range(num_nodes)]
    for (node_id_1, node_id_2) in zip(edge_index[0], edge_index[1]):
        adjacency_lists[node_id_1].add(node_id_2)
    return adjacency_lists


class Node2vecDataset(Dataset):
    def __init__(self, edge_index, node_id_to_token_ids_dict: Dict[int, List[List[int]]], walk_length: int,
                 walks_per_node: int, p: float, q: float, num_negative_samples: int, context_size: int,
                 seq_max_length, num_nodes=None, ):
        assert walk_length >= context_size
        self.node_id_to_token_ids_dict = node_id_to_token_ids_dict
        self.walks_per_node = walks_per_node
        if random_walk is None:
            raise ImportError('`Node2Vec` requires `torch-cluster`.')

        N = maybe_num_nodes(edge_index, num_nodes)
        row, col = edge_index
        self.adj = SparseTensor(row=row, col=col, sparse_sizes=(N, N))
        self.adj = self.adj.to('cpu')
        self.walk_length = walk_length - 1
        self.context_size = context_size
        self.walks_per_node = walks_per_node
        self.p = p
        self.q = q
        self.num_negative_samples = num_negative_samples
        self.seq_max_length = seq_max_length
        self.adjacency_lists = create_one_hop_adjacency_lists(num_nodes=num_nodes, edge_index=edge_index)

    def __getitem__(self, idx):
        return idx

    def __len__(self):
        return self.adj.sparse_size(0)

    def sample(self, batch):
        if not isinstance(batch, torch.Tensor):
            batch = torch.tensor(batch)
        pos_batch = self.pos_sample(batch)
        neg_batch = self.neg_sample(batch, pos_batch)

        pos_batch_input_ids, pos_batch_attention_masks = self.node_ids2tokenizer_output(pos_batch)
        neg_batch_input_ids, neg_batch_attention_masks = self.node_ids2tokenizer_output(neg_batch)

        return pos_batch_input_ids, pos_batch_attention_masks, neg_batch_input_ids, neg_batch_attention_masks

    def pos_sample(self, batch):
        batch = batch.repeat(self.walks_per_node)

        rowptr, col, _ = self.adj.coo()
        rw = random_walk(rowptr, col, batch, self.walk_length, self.p, self.q)
        if not isinstance(rw, torch.Tensor):
            rw = rw[0]

        walks = []
        num_walks_per_rw = 1 + self.walk_length + 1 - self.context_size
        for j in range(num_walks_per_rw):
            walks.append(rw[:, j:j + self.context_size])
        return torch.cat(walks, dim=0)

    def neg_sample(self, batch, pos_batch: torch.Tensor):
        batch = batch.repeat(self.walks_per_node * self.num_negative_samples)
        # num_nodes = self.adj.sparse_size(0)
        neg_rw_tensors_list = []
        for anchor_node_id, positive_node_ids in zip(batch, pos_batch):
            assert len(batch.size()) == 1 and len(pos_batch.size()) == 2
            pos_node_ids_set = set((int(x) for x in positive_node_ids.unique()))
            anchor_node_neighborhood_node_ids = self.adjacency_lists[anchor_node_id]
            reroll_neg_batch = True
            while reroll_neg_batch:
                rw = torch.randint(self.adj.sparse_size(0), (self.walk_length,))
                neg_rw_node_ids_set = set((int(x) for x in rw.unique()))
                neg_rw_pos_batch_intersection = neg_rw_node_ids_set.intersection(pos_node_ids_set)
                if len(neg_rw_pos_batch_intersection) > 0:
                    continue
                neg_rw_anchor_node_neighborhood_intersection = neg_rw_node_ids_set.intersection(
                    anchor_node_neighborhood_node_ids)
                if len(neg_rw_anchor_node_neighborhood_intersection) > 0:
                    continue
                reroll_neg_batch = False
                for positive_node_id in pos_node_ids_set:
                    positive_node_neighborhood_node_ids = self.adjacency_lists[positive_node_id]
                    neg_rw_positive_node_neighborhood_intersection = neg_rw_node_ids_set.intersection(
                        positive_node_neighborhood_node_ids)
                    if len(neg_rw_positive_node_neighborhood_intersection) > 0:
                        reroll_neg_batch = True
                        break
            neg_rw_tensors_list.append(rw)
        rw = torch.stack(neg_rw_tensors_list)

        rw = torch.cat([batch.view(-1, 1), rw], dim=-1)

        walks = []
        num_walks_per_rw = 1 + self.walk_length + 1 - self.context_size
        for j in range(num_walks_per_rw):
            walks.append(rw[:, j:j + self.context_size])
        return torch.cat(walks, dim=0)

    def node_ids2tokenizer_output(self, batch):
        batch_size = batch.size()[0]
        num_samples = batch.size()[1]

        tokenizer_outputs = [random.choice(self.node_id_to_token_ids_dict[node_id.item()]) for node_id in
                             batch.view(-1)]
        batch_input_ids = torch.stack([tok_output["input_ids"][0] for tok_output in tokenizer_outputs])
        batch_attention_masks = torch.stack([tok_output["attention_mask"][0] for tok_output in tokenizer_outputs])

        batch_input_ids = batch_input_ids.view(batch_size, num_samples, self.seq_max_length)
        batch_attention_masks = batch_attention_masks.view(batch_size, num_samples, self.seq_max_length)

        return batch_input_ids, batch_attention_masks


def load_data_and_bert_model(train_node2terms_path: str, train_edges_path: str, val_node2terms_path: str,
                             val_edges_path: str, text_encoder_name: str, text_encoder_seq_length: int,
                             drop_relations_info: bool, use_fast: bool = True, do_lower_case=True,
                             no_val=True):
    train_node_id2terms_dict = load_node_id2terms_list(dict_path=train_node2terms_path, )
    train_edges_tuples = load_edges_tuples(train_edges_path)
    if drop_relations_info:
        train_edges_tuples = [(t[0], t[1]) for t in train_edges_tuples]

    tokenizer = AutoTokenizer.from_pretrained(text_encoder_name, do_lower_case=do_lower_case, use_fast=use_fast)
    bert_encoder = AutoModel.from_pretrained(text_encoder_name, )

    val_node_id2token_ids_dict = None
    val_edges_tuples = None
    if not no_val:
        val_node_id2terms_dict = load_node_id2terms_list(dict_path=val_node2terms_path, )
        val_edges_tuples = load_edges_tuples(val_edges_path)
        if drop_relations_info:
            val_edges_tuples = [(t[0], t[1]) for t in val_edges_tuples]
        logging.info("Tokenizing val node names")
        val_node_id2token_ids_dict = tokenize_node_terms(val_node_id2terms_dict, tokenizer,
                                                         max_length=text_encoder_seq_length)

    logging.info("Tokenizing training node names")
    train_node_id2token_ids_dict = tokenize_node_terms(train_node_id2terms_dict, tokenizer,
                                                       max_length=text_encoder_seq_length)


    return bert_encoder, tokenizer, train_node_id2token_ids_dict, train_edges_tuples, val_node_id2token_ids_dict, val_edges_tuples


def convert_edges_tuples_to_oriented_edge_index_with_relations(edges_tuples: List[Tuple[int, int]],
                                                               use_rel_or_rela: str, remove_selfloops=False) \
        -> Tuple[torch.Tensor, torch.Tensor]:
    logging.info("Converting edge tuples to edge index")
    edge_strs_set = set()
    for idx, (node_id_1, node_id_2, rel_id, rela_id) in enumerate(edges_tuples):
        if use_rel_or_rela == "rel":
            pass
        elif use_rel_or_rela == "rela":
            rel_id = rela_id
        else:
            raise ValueError(f"Invalid 'use_rel_or_rela' parameter value: {use_rel_or_rela}")
        if not (remove_selfloops and node_id_1 == node_id_2):
            edge_str = f"{node_id_1}~{node_id_2}~{rel_id}"
            edge_strs_set.add(edge_str)
    edge_index = torch.zeros(size=[2, len(edge_strs_set)], dtype=torch.long)
    edge_rel_ids = torch.zeros(len(edge_strs_set), dtype=torch.long)
    for idx, edge_str in enumerate(edge_strs_set):
        edge_attributes = edge_str.split('~')
        node_id_1 = int(edge_attributes[0])
        node_id_2 = int(edge_attributes[1])
        rel_id = int(edge_attributes[2])

        edge_index[0][idx] = node_id_1
        edge_index[1][idx] = node_id_2
        edge_rel_ids[idx] = rel_id
    logging.info(f"Edge index is created. The size is {edge_index.size()}, there are {edge_index.max()} nodes")

    return edge_index, edge_rel_ids


class SimpleDataset(Dataset):
    def __init__(self, num_elements):
        self.num_elements = num_elements

    def __getitem__(self, idx):
        return idx

    def __len__(self):
        return self.num_elements


def load_positive_pairs(triplet_file_path: str) -> Tuple[List[str], List[str], List[int]]:
    term_1_list: List[str] = []
    term_2_list: List[str] = []
    concept_ids: List[int] = []
    with codecs.open(triplet_file_path, 'r', encoding="utf-8") as inp_file:
        for line in inp_file:
            attrs = line.strip().split('||')
            concept_id = int(attrs[0])
            term_1 = attrs[1]
            term_2 = attrs[2]

            concept_ids.append(concept_id)
            term_1_list.append(term_1)
            term_2_list.append(term_2)
    return term_1_list, term_2_list, concept_ids


def map_terms2term_id(term_1_list: List[str], term_2_list: List[str]) -> Tuple[List[int], List[int], Dict[str, int]]:
    unique_terms: Set[str] = set()
    unique_terms.update(term_1_list)
    unique_terms.update(term_2_list)
    term2id = {term: term_id for term_id, term in enumerate(sorted(unique_terms))}

    term_1_ids = [term2id[term] for term in term_1_list]
    term_2_ids = [term2id[term] for term in term_2_list]

    return term_1_ids, term_2_ids, term2id


def create_term_id2tokenizer_output(term2id: Dict[str, int], max_length: int, tokenizer: BertTokenizerFast):
    logging.info("Tokenizing terms....")
    term_id2tok_out = {}
    for term, term_id in term2id.items():
        tok_out = tokenizer.encode_plus(
            term,
            max_length=max_length,
            padding="max_length",
            truncation=True,
            add_special_tokens=True,
            return_tensors="pt")
        term_id2tok_out[term_id] = tok_out
    logging.info("Finished tokenizing terms....")
    return term_id2tok_out


def load_tree_dataset_and_bert_model(node2terms_path: str,  text_encoder_name: str,
                                     parent_children_adjacency_list_path: str, child_parents_adjacency_list_path: str,
                                     text_encoder_seq_length: int, use_fast: bool = True, do_lower_case=True):
    node_id2terms_dict = load_node_id2terms_list(dict_path=node2terms_path, )

    parent_children_adjacency_list = load_adjacency_list(input_path=parent_children_adjacency_list_path)
    child_parents_adjacency_list = load_adjacency_list(input_path=child_parents_adjacency_list_path)

    tokenizer = AutoTokenizer.from_pretrained(text_encoder_name, do_lower_case=do_lower_case, use_fast=use_fast)
    bert_encoder = AutoModel.from_pretrained(text_encoder_name, )
    node_id2token_ids_dict = tokenize_node_terms(node_id2terms_dict, tokenizer,
                                                       max_length=text_encoder_seq_length)

    return bert_encoder, tokenizer, node_id2token_ids_dict, parent_children_adjacency_list, child_parents_adjacency_list

class PositivePairNeighborSampler(RawNeighborSampler):
    def __init__(self, pos_pairs_term_1_id_list: List[int], pos_pairs_term_2_id_list: List[int],
                 pos_pairs_concept_ids_list: List[int], term_id2tokenizer_output: Dict,
                 node_id2token_ids_dict, seq_max_length, *args, **kwargs):
        super(PositivePairNeighborSampler, self).__init__(*args, **kwargs)
        self.node_id_to_token_ids_dict = node_id2token_ids_dict
        assert len(pos_pairs_term_1_id_list) == len(pos_pairs_term_2_id_list) == len(pos_pairs_concept_ids_list)
        self.pos_pairs_term_1_id_list = pos_pairs_term_1_id_list
        self.pos_pairs_term_2_id_list = pos_pairs_term_2_id_list
        self.pos_pairs_concept_ids_list = pos_pairs_concept_ids_list

        self.term_id2tokenizer_output = term_id2tokenizer_output
        self.seq_max_length = seq_max_length

    def __len__(self):
        return len(self.pos_pairs_term_1_id_list) // self.batch_size

    def sample(self, batch):
        term_1_ids = [self.pos_pairs_term_1_id_list[idx] for idx in batch]
        term_1_tok_out = [self.term_id2tokenizer_output[idx] for idx in term_1_ids]
        term_1_input_ids = torch.stack([t_out["input_ids"][0] for t_out in term_1_tok_out])
        term_1_att_masks = torch.stack([t_out["attention_mask"][0] for t_out in term_1_tok_out])

        term_2_ids = [self.pos_pairs_term_2_id_list[idx] for idx in batch]
        term_2_tok_out = [self.term_id2tokenizer_output[idx] for idx in term_2_ids]
        term_2_input_ids = torch.stack([t_out["input_ids"][0] for t_out in term_2_tok_out])
        term_2_att_masks = torch.stack([t_out["attention_mask"][0] for t_out in term_2_tok_out])

        assert term_1_input_ids.size()[1] == term_1_att_masks.size()[1] == self.seq_max_length
        assert term_2_input_ids.size()[1] == term_2_att_masks.size()[1] == self.seq_max_length

        triplet_concept_ids = torch.LongTensor([self.pos_pairs_concept_ids_list[idx] for idx in batch])

        assert len(triplet_concept_ids) == len(term_1_input_ids)
        (batch_size, n_id, adjs) = super(PositivePairNeighborSampler, self).sample(triplet_concept_ids)
        neighbor_node_ids = n_id[batch_size:]

        term_1_neighbor_input_ids, term_1_neighbor_att_masks = node_ids2tokenizer_output(
            batch=neighbor_node_ids, node_id_to_token_ids_dict=self.node_id_to_token_ids_dict,
            seq_max_length=self.seq_max_length)
        term_2_neighbor_input_ids, term_2_neighbor_att_masks = node_ids2tokenizer_output(
            batch=neighbor_node_ids, node_id_to_token_ids_dict=self.node_id_to_token_ids_dict,
            seq_max_length=self.seq_max_length)

        assert term_1_neighbor_input_ids.size() == term_1_neighbor_att_masks.size() \
               == term_2_neighbor_att_masks.size()
        assert term_2_neighbor_input_ids.size() == term_2_neighbor_att_masks.size()

        term_1_input_ids = torch.cat((term_1_input_ids, term_1_neighbor_input_ids), dim=0)
        term_1_att_masks = torch.cat((term_1_att_masks, term_1_neighbor_att_masks), dim=0)
        term_2_input_ids = torch.cat((term_2_input_ids, term_2_neighbor_input_ids), dim=0)
        term_2_att_masks = torch.cat((term_2_att_masks, term_2_neighbor_att_masks), dim=0)
        term_1_input = (term_1_input_ids, term_1_att_masks)
        term_2_input = (term_2_input_ids, term_2_att_masks,)

        batch_dict = {
            "term_1_input": term_1_input, "term_2_input": term_2_input, "adjs": adjs, "batch_size": batch_size,
            "concept_ids": n_id  # "concept_ids": triplet_concept_ids
        }
        return batch_dict


class PositiveRelationalNeighborSampler(RawNeighborSampler):
    def __init__(self, pos_pairs_term_1_id_list: List[int], pos_pairs_term_2_id_list: List[int],
                 pos_pairs_concept_ids_list: List[int], term_id2tokenizer_output: Dict,
                 rel_ids, node_id2token_ids_dict, seq_max_length, *args, **kwargs):
        super(PositiveRelationalNeighborSampler, self).__init__(*args, **kwargs)
        self.node_id2token_ids_dict = node_id2token_ids_dict
        assert len(pos_pairs_term_1_id_list) == len(pos_pairs_term_2_id_list) == len(pos_pairs_concept_ids_list)
        self.pos_pairs_term_1_id_list = pos_pairs_term_1_id_list
        self.pos_pairs_term_2_id_list = pos_pairs_term_2_id_list
        self.pos_pairs_concept_ids_list = pos_pairs_concept_ids_list
        self.rel_ids = rel_ids
        self.term_id2tokenizer_output = term_id2tokenizer_output
        self.seq_max_length = seq_max_length

        self.num_edges = self.edge_index.size()[1]

        assert self.num_edges == len(rel_ids)

    def __len__(self):
        return len(self.pos_pairs_term_1_id_list) // self.batch_size

    def sample(self, batch):
        term_1_ids = [self.pos_pairs_term_1_id_list[idx] for idx in batch]
        term_1_tok_out = [self.term_id2tokenizer_output[idx] for idx in term_1_ids]
        term_1_input_ids = torch.stack([t_out["input_ids"][0] for t_out in term_1_tok_out])
        term_1_att_masks = torch.stack([t_out["attention_mask"][0] for t_out in term_1_tok_out])

        term_2_ids = [self.pos_pairs_term_2_id_list[idx] for idx in batch]
        term_2_tok_out = [self.term_id2tokenizer_output[idx] for idx in term_2_ids]
        term_2_input_ids = torch.stack([t_out["input_ids"][0] for t_out in term_2_tok_out])
        term_2_att_masks = torch.stack([t_out["attention_mask"][0] for t_out in term_2_tok_out])

        assert term_1_input_ids.size()[1] == term_1_att_masks.size()[1] == self.seq_max_length
        assert term_2_input_ids.size()[1] == term_2_att_masks.size()[1] == self.seq_max_length

        triplet_concept_ids = torch.LongTensor([self.pos_pairs_concept_ids_list[idx] for idx in batch])
        assert len(triplet_concept_ids) == len(term_1_input_ids)

        (batch_size, n_id, adjs) = super(PositiveRelationalNeighborSampler, self).sample(triplet_concept_ids)

        neighbor_node_ids = n_id[batch_size:]

        if not isinstance(adjs, list):
            adjs = [adjs, ]
        e_ids_list = [adj.e_id for adj in adjs]
        rel_ids_list = [self.rel_ids[e_ids] for e_ids in e_ids_list]

        term_1_neighbor_input_ids, term_1_neighbor_att_masks = node_ids2tokenizer_output(
            batch=neighbor_node_ids, node_id_to_token_ids_dict=self.node_id2token_ids_dict,
            seq_max_length=self.seq_max_length)
        term_2_neighbor_input_ids, term_2_neighbor_att_masks = node_ids2tokenizer_output(
            batch=neighbor_node_ids, node_id_to_token_ids_dict=self.node_id2token_ids_dict,
            seq_max_length=self.seq_max_length)
        assert term_1_neighbor_input_ids.size() == term_1_neighbor_att_masks.size() \
               == term_2_neighbor_att_masks.size()
        assert term_2_neighbor_input_ids.size() == term_2_neighbor_att_masks.size()

        term_1_input_ids = torch.cat((term_1_input_ids, term_1_neighbor_input_ids), dim=0)
        term_1_att_masks = torch.cat((term_1_att_masks, term_1_neighbor_att_masks), dim=0)
        term_2_input_ids = torch.cat((term_2_input_ids, term_2_neighbor_input_ids), dim=0)
        term_2_att_masks = torch.cat((term_2_att_masks, term_2_neighbor_att_masks), dim=0)
        term_1_input = (term_1_input_ids, term_1_att_masks)
        term_2_input = (term_2_input_ids, term_2_att_masks,)

        batch_dict = {
            "term_1_input": term_1_input, "term_2_input": term_2_input, "adjs": adjs, "batch_size": batch_size,
            "concept_ids": n_id, "rel_ids_list": rel_ids_list,  # "concept_ids": triplet_concept_ids

        }
        return batch_dict
