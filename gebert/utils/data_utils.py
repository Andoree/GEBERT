import random
from typing import Dict, List

import torch


def node_ids2tokenizer_output(batch: torch.Tensor, node_id_to_token_ids_dict: Dict[int, List[List[Dict[str, int]]]],
                              seq_max_length: int):
    """
    Given a batch of node ids, for each node in the batch samples a random term of this node.
    Then, takes pre-calculated BERT tokenizer output of this term. Returns a term's token ids and attention masks
    :param batch: Batch of node ids
    :param node_id_to_token_ids_dict: Dictionary {node_id : List}
    :param seq_max_length: BERT sequence length
    :return: Token ids and attention masks tensor
    """
    output_shape = list(batch.size()) + [seq_max_length, ]
    tokenizer_outputs = [random.choice(node_id_to_token_ids_dict[node_id.item()]) for node_id in
                         batch.view(-1)]
    batch_input_ids = torch.stack([tok_output["input_ids"][0] for tok_output in tokenizer_outputs])
    batch_attention_masks = torch.stack([tok_output["attention_mask"][0] for tok_output in tokenizer_outputs])

    batch_input_ids = batch_input_ids.view(output_shape)
    batch_attention_masks = batch_attention_masks.view(output_shape)

    return batch_input_ids, batch_attention_masks


def create_rel_id2inverse_rel_id_map(rel2id: Dict[str, int]) -> Dict[int, int]:
    """
    Takes rel2id map and for each non-loop ("LOOP") relation creates an inverse relation.
    :param rel2id: Dict {relation name : relation id}
    :return: Dict {relation id : inverse relation id}
    """
    num_rels = len(rel2id)
    assert num_rels - 1 == max(rel2id.values())
    inv_rel_id = num_rels
    rel_id2reverse_rel_id = {}
    for rel, rel_id in rel2id.items():
        if rel != "LOOP":
            rel_id2reverse_rel_id[rel_id] = inv_rel_id
            inv_rel_id += 1
        else:
            rel_id2reverse_rel_id[rel_id] = rel_id
    return rel_id2reverse_rel_id
