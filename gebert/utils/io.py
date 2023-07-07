import codecs
import logging
import os.path
from typing import Dict, List, Tuple, Iterable

import pandas as pd
from tqdm import tqdm


def save_node_id2terms_list(save_path: str, mapping: Dict[str, List[str]], node_terms_sep: str = '\t',
                            terms_sep: str = '|||'):
    num_concepts = len(mapping)
    logging.info(f"Saving CUIs and terms. There are {num_concepts}")
    with codecs.open(save_path, 'w+', encoding="utf-8") as out_file:
        for node_id, terms_list in tqdm(mapping.items(), miniters=num_concepts // 50, total=num_concepts):
            assert node_terms_sep not in str(node_id) and terms_sep not in str(node_id)
            for term in terms_list:
                if node_terms_sep in term:
                    raise ValueError(f"col_sep {node_terms_sep} is present in data being saved")
                if terms_sep in term:
                    raise ValueError(f"terms_sep {terms_sep} is present in data being saved")
            terms_str = terms_sep.join(terms_list)
            out_file.write(f"{node_id}{node_terms_sep}{terms_str}\n")
    logging.info("Finished saving CUIs and terms")


def save_dict(save_path: str, dictionary: Dict, sep: str = '\t'):
    logging.info("Saving dictionary")
    with codecs.open(save_path, 'w+', encoding="utf-8") as out_file:
        for key, val in dictionary.items():
            key, val = str(key), str(val)
            if sep in key or sep in val:
                raise Exception(f"Separator {sep} is present in dictionary being saved")
            out_file.write(f"{key}{sep}{val}\n")
    logging.info("Finished saving dictionary")


def save_tuples(save_path: str, tuples: List[Tuple], sep='\t'):
    logging.info("Saving tuples")
    with codecs.open(save_path, 'w+', encoding="utf-8") as out_file:
        for t in tuples:
            s = sep.join((str(x) for x in t))
            out_file.write(f"{s}\n")
    logging.info("Finished saving tuples")


def load_node_id2terms_list(dict_path: str, node_terms_sep: str = '\t', terms_sep: str = '|||') \
        -> Dict[int, List[str]]:
    logging.info("Loading node_id to terms map")
    node_id2_terms: Dict[int, List[str]] = {}
    with codecs.open(dict_path, 'r', encoding="utf-8") as inp_file:
        for line in inp_file:
            attrs = line.strip().split(node_terms_sep)
            node_id = int(attrs[0])
            terms_split = attrs[1].split(terms_sep)
            node_id2_terms[node_id] = terms_split
    logging.info(f"Loaded node_id to terms map, there are {len(node_id2_terms.keys())} entries")
    return node_id2_terms


def load_edges_tuples(path: str, sep: str = '\t') -> List[Tuple]:
    logging.info(f"Starting loading tuples from: {path}")
    tuples = []
    with codecs.open(path, 'r', encoding="utf-8") as inp_file:
        for line in inp_file:
            attrs = line.strip().split(sep)
            node_id_1 = int(attrs[0])
            node_id_2 = int(attrs[1])
            rel_id = int(attrs[2]) if len(attrs) > 2 else None
            rela_id = int(attrs[3]) if len(attrs) > 2 else None
            tuples.append((node_id_1, node_id_2, rel_id, rela_id))
    logging.info(f"Finished loading tuples, there are {len(tuples)} tuples")
    return tuples


def load_dict(path: str, sep: str = '\t') -> Dict[str, str]:
    df = pd.read_csv(path, header=None, names=("key", "value"), sep=sep, encoding="utf-8")
    return dict(zip(df.key, df.value))


def read_mrconso(fpath) -> pd.DataFrame:
    columns = ['CUI', 'LAT', 'TS', 'LUI', 'STT', 'SUI', 'ISPREF', 'AUI', 'SAUI', 'SCUI', 'SDUI', 'SAB', 'TTY', 'CODE',
               'STR', 'SRL', 'SUPPRESS', 'CVF', 'NOCOL']
    return pd.read_csv(fpath, names=columns, sep='|', encoding='utf-8', quoting=3)


def read_mrsty(fpath) -> pd.DataFrame:
    columns = ['CUI', 'TUI', 'STN', 'STY', 'ATUI', 'CVF', 'NOCOL']
    return pd.read_csv(fpath, names=columns, sep='|', encoding='utf-8', quoting=3)


def read_mrrel(fpath) -> pd.DataFrame:
    columns = ["CUI1", "AUI1", "STYPE1", "REL", "CUI2", "AUI2", "STYPE2", "RELA", "RUI", "SRUI", "RSAB", "VSAB",
               "SL", "RG", "DIR", "SUPPRESS", "CVF", 'NOCOL']
    return pd.read_csv(fpath, names=columns, sep='|', encoding='utf-8')


def read_mrdef(fpath) -> pd.DataFrame:
    columns = ["CUI", "AUI", "ATUI", "SATUI", "SAB", "DEF", "SUPPRESS", "CVF", 'NOCOL']
    return pd.read_csv(fpath, names=columns, sep='|', encoding='utf-8')


def read_sem_groups(fpath) -> pd.DataFrame:
    columns = ["Semantic Group Abbrev", "Semantic Group Name", "TUI", "Full Semantic Type Name"]
    return pd.read_csv(fpath, names=columns, sep='|', encoding='utf-8')


def update_log_file(path: str, dict_to_log: Dict):
    with codecs.open(path, 'a+', encoding="utf-8") as out_file:
        s = ', '.join((f"{k} : {v}" for k, v in dict_to_log.items()))
        out_file.write(f"{s}\n")


def save_encoder_from_checkpoint(bert_encoder, bert_tokenizer, save_path: str):
    output_dir = os.path.dirname(save_path)
    if not os.path.exists(output_dir) and output_dir != '':
        os.makedirs(output_dir)
    logging.info(f"Saving textual encoder and tokenizer to {save_path}")
    bert_encoder.cpu().save_pretrained(save_path)
    bert_encoder.config.save_pretrained(save_path)
    bert_tokenizer.save_pretrained(save_path)
    logging.info(f"Successfully saved textual encoder and tokenizer to {save_path}")


def write_strings(fpath: str, strings_list: List[str]):
    with codecs.open(fpath, 'w+', encoding="utf-8") as out_file:
        for s in strings_list:
            out_file.write(f"{s.strip()}\n")


def save_adjacency_list(adjacency_list: Dict[int, Iterable[int]], save_path: str, source_node_sep='\t',
                        inter_target_nodes_sep=','):
    with codecs.open(save_path, 'w+', encoding="utf-8") as out_file:
        for src_node, trg_nodes_list in adjacency_list.items():
            s = inter_target_nodes_sep.join((str(x) for x in trg_nodes_list))
            out_file.write(f"{src_node}{source_node_sep}{s}\n")


def load_adjacency_list(input_path: str, source_node_sep='\t', inter_target_nodes_sep=',') -> Dict[int, List[int]]:
    adjacency_list: Dict[int, List[int]] = {}
    with codecs.open(input_path, 'r', encoding="utf-8") as inp_file:
        for line in inp_file:
            src_target_nodes_tuple = line.strip().split(source_node_sep)
            src_node_id = int(src_target_nodes_tuple[0])
            target_nodes_list = list(map(int, src_target_nodes_tuple[1].split(inter_target_nodes_sep)))
            adjacency_list[src_node_id] = target_nodes_list
    return adjacency_list
