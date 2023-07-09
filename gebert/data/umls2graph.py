import logging
import os
from argparse import ArgumentParser
from typing import Dict, List, Tuple, Set

import pandas as pd
from tqdm import tqdm

from gebert.utils.io import save_node_id2terms_list, save_dict, save_tuples, read_mrconso, read_mrrel


def get_concept_list_groupby_cui(mrconso_df: pd.DataFrame, cui2node_id: Dict[str, int]) \
        -> (Dict[int, Set[str]], Dict[int, str], Dict[str, int]):
    logging.info("Started creating CUI to terms mapping")
    node_id2terms_list: Dict[int, Set[str]] = {}
    logging.info(f"Removing duplicated (CUI, STR) pairs, {mrconso_df.shape[0]} rows before deletion")
    mrconso_df.drop_duplicates(subset=("CUI", "STR"), keep="first", inplace=True)
    logging.info(f"Removed duplicated (CUI, STR) pairs, {mrconso_df.shape[0]} rows after deletion")

    unique_cuis_set = set(mrconso_df["CUI"].unique())
    logging.info(f"There are {len(unique_cuis_set)} unique CUIs in dataset")
    # node_id2cui: Dict[int, str] = {node_id: cui for node_id, cui in enumerate(unique_cuis_set)}
    # cui2node_id: Dict[str, int] = {cui: node_id for node_id, cui in node_id2cui.items()}
    # assert len(node_id2cui) == len(cui2node_id)
    for _, row in tqdm(mrconso_df.iterrows(), miniters=mrconso_df.shape[0] // 50):
        cui = row["CUI"].strip()
        term_str = row["STR"].strip().lower()
        if term_str == '':
            continue
        node_id = cui2node_id[cui]
        if node_id2terms_list.get(node_id) is None:
            node_id2terms_list[node_id] = set()
        node_id2terms_list[node_id].add(term_str.strip())
    logging.info("CUI to terms mapping is created")
    return node_id2terms_list


def extract_umls_oriented_edges_with_relations(mrrel_df: pd.DataFrame, cui2node_id: Dict[str, int],
                                               rel2rel_id: Dict[str, int], rela2rela_id: Dict[str, int],
                                               ignore_not_mapped_edges=False) -> List[Tuple[int, int, int, int]]:
    cuis_relation_str_set = set()
    logging.info("Started generating graph edges")
    edges: List[Tuple[int, int, int, int]] = []
    not_mapped_edges_counter = 0
    for idx, row in tqdm(mrrel_df.iterrows(), miniters=mrrel_df.shape[0] // 100, total=mrrel_df.shape[0]):
        cui_1 = row["CUI1"].strip()
        cui_2 = row["CUI2"].strip()
        rel = row["REL"]
        rela = row["RELA"]
        # Separator validation
        for att in (cui_1, cui_2, rel, rela):
            assert "~~" not in str(att)
        if cui2node_id.get(cui_1) is not None and cui2node_id.get(cui_2) is not None:
            cuis_relation_str = f"{cui_1}~~{cui_2}~~{rel}~~{rela}"
            if cuis_relation_str not in cuis_relation_str_set:
                cui_1_node_id = cui2node_id[cui_1]
                cui_2_node_id = cui2node_id[cui_2]
                rel_id = rel2rel_id[rel]
                rela_id = rela2rela_id[rela]
                edges.append((cui_1_node_id, cui_2_node_id, rel_id, rela_id))
            cuis_relation_str_set.add(cuis_relation_str)
        else:
            if not ignore_not_mapped_edges:
                raise AssertionError(f"Either CUI {cui_1} or {cui_2} are not found in CUI2node_is mapping")
            else:
                not_mapped_edges_counter += 1
    if ignore_not_mapped_edges:
        logging.info(f"{not_mapped_edges_counter} edges are not mapped to any node")
    logging.info(f"Finished generating edges. There are {len(edges)} edges")

    return edges


def create_graph_files(mrconso_df: pd.DataFrame, mrrel_df: pd.DataFrame, rel2id: Dict[str, int],
                       cui2node_id: Dict[str, int], rela2id: Dict[str, int], output_node_id2terms_list_path: str,
                       output_node_id2cui_path: str, output_edges_path: str, output_rel2rel_id_path: str,
                       output_rela2rela_id_path, ignore_not_mapped_edges: bool):
    node_id2cui: Dict[int, str] = {node_id: cui for cui, node_id in cui2node_id.items()}
    node_id2terms_list = get_concept_list_groupby_cui(mrconso_df=mrconso_df, cui2node_id=cui2node_id)
    logging.info("Generating edges....")

    edges = extract_umls_oriented_edges_with_relations(mrrel_df=mrrel_df, cui2node_id=cui2node_id,
                                                       rel2rel_id=rel2id, rela2rela_id=rela2id,
                                                       ignore_not_mapped_edges=ignore_not_mapped_edges)

    logging.info("Saving the result....")
    save_node_id2terms_list(save_path=output_node_id2terms_list_path, mapping=node_id2terms_list, )
    save_dict(save_path=output_node_id2cui_path, dictionary=node_id2cui)
    save_dict(save_path=output_rel2rel_id_path, dictionary=rel2id)
    save_dict(save_path=output_rela2rela_id_path, dictionary=rela2id)
    save_tuples(save_path=output_edges_path, tuples=edges)


def create_cui2node_id_mapping(mrconso_df: pd.DataFrame) -> Dict[str, int]:
    unique_cuis_set = set(mrconso_df["CUI"].unique())
    cui2node_id: Dict[str, int] = {cui: node_id for node_id, cui in enumerate(unique_cuis_set)}

    return cui2node_id


def create_relations2id_dicts(mrrel_df: pd.DataFrame):
    mrrel_df.REL.fillna("NAN", inplace=True)
    mrrel_df.RELA.fillna("NAN", inplace=True)
    rel2id = {rel: rel_id for rel_id, rel in enumerate(mrrel_df.REL.unique())}
    rela2id = {rela: rela_id for rela_id, rela in enumerate(mrrel_df.RELA.unique())}
    rel2id["LOOP"] = max(rel2id.values()) + 1
    rela2id["LOOP"] = max(rela2id.values()) + 1
    logging.info(f"There are {len(rel2id.keys())} unique RELs and {len(rela2id.keys())} unique RELAs")
    print("REL2REL_ID", )
    for k, v in rel2id.items():
        print(k, v)
    print("RELA2RELA_ID", rela2id)
    for k, v in rel2aid.items():
        print(k, v)
    return rel2id, rela2id


def main():
    parser = ArgumentParser()
    parser.add_argument('--mrconso')
    parser.add_argument('--mrrel')
    parser.add_argument('--split_val', action="store_true")
    parser.add_argument('--train_proportion', type=float)
    parser.add_argument('--output_dir', type=str)
    args = parser.parse_args()

    split_val = args.split_val
    output_dir = args.output_dir
    if not os.path.exists(output_dir) and output_dir != '':
        os.makedirs(output_dir)

    logging.info("Loading MRCONSO....")
    mrconso_df = read_mrconso(args.mrconso)
    mrconso_df["STR"].fillna('', inplace=True)
    logging.info("Loading MRREL....")
    mrrel_df = read_mrrel(args.mrrel)

    logging.info("Generating node index....")
    rel2id, rela2id = create_relations2id_dicts(mrrel_df)
    if split_val:
        train_dir = os.path.join(output_dir, "train/")
        val_dir = os.path.join(output_dir, "val/")
        for d in (train_dir, val_dir):
            if not os.path.exists(d):
                os.makedirs(d)
        train_proportion = args.train_proportion
        num_rows = mrconso_df.shape[0]
        shuffled_mrconso = mrconso_df.sample(frac=1.0, random_state=42)
        del mrconso_df
        num_train_rows = int(num_rows * train_proportion)
        train_mrconso_df = shuffled_mrconso[:num_train_rows]
        val_mrconso_df = shuffled_mrconso[num_train_rows:]
        del shuffled_mrconso

        train_output_node_id2terms_list_path = os.path.join(train_dir, "node_id2terms_list")
        val_output_node_id2terms_list_path = os.path.join(val_dir, "node_id2terms_list")
        train_output_node_id2cui_path = os.path.join(train_dir, "id2cui")
        val_output_node_id2cui_path = os.path.join(val_dir, "id2cui")
        train_output_edges_path = os.path.join(train_dir, "edges")
        val_output_edges_path = os.path.join(val_dir, "edges")

        train_output_rel2rel_id_path = os.path.join(train_dir, "rel2id")
        val_output_rel2rel_id_path = os.path.join(val_dir, "rel2id")
        train_output_rela2rela_id_path = os.path.join(train_dir, "rela2id")
        val_output_rela2rela_id_path = os.path.join(val_dir, "rela2id")

        train_cui2node_id = create_cui2node_id_mapping(mrconso_df=train_mrconso_df)
        val_cui2node_id = create_cui2node_id_mapping(mrconso_df=val_mrconso_df)
        logging.info("Creating train graph files")
        create_graph_files(mrconso_df=train_mrconso_df, mrrel_df=mrrel_df, rel2id=rel2id, rela2id=rela2id,
                           cui2node_id=train_cui2node_id,
                           output_node_id2terms_list_path=train_output_node_id2terms_list_path,
                           output_node_id2cui_path=train_output_node_id2cui_path,
                           output_edges_path=train_output_edges_path,
                           output_rel2rel_id_path=train_output_rel2rel_id_path,
                           output_rela2rela_id_path=train_output_rela2rela_id_path, ignore_not_mapped_edges=True, )
        logging.info("Creating val graph files")
        create_graph_files(mrconso_df=val_mrconso_df, mrrel_df=mrrel_df, rel2id=rel2id, rela2id=rela2id,
                           cui2node_id=val_cui2node_id,
                           output_node_id2terms_list_path=val_output_node_id2terms_list_path,
                           output_node_id2cui_path=val_output_node_id2cui_path,
                           output_edges_path=val_output_edges_path, output_rel2rel_id_path=val_output_rel2rel_id_path,
                           output_rela2rela_id_path=val_output_rela2rela_id_path,
                           ignore_not_mapped_edges=True, )
    else:
        logging.info("Creating graph files")
        output_node_id2terms_list_path = os.path.join(output_dir, "node_id2terms_list")
        output_node_id2cui_path = os.path.join(output_dir, "id2cui")
        output_edges_path = os.path.join(output_dir, "edges")
        output_rel2rel_id_path = os.path.join(output_dir, f"rel2id")
        output_rela2rela_id_path = os.path.join(output_dir, f"rela2id")
        cui2node_id = create_cui2node_id_mapping(mrconso_df=mrconso_df)
        create_graph_files(mrconso_df=mrconso_df, mrrel_df=mrrel_df, rel2id=rel2id, rela2id=rela2id,
                           cui2node_id=cui2node_id,
                           output_node_id2terms_list_path=output_node_id2terms_list_path,
                           output_node_id2cui_path=output_node_id2cui_path,
                           output_edges_path=output_edges_path, output_rel2rel_id_path=output_rel2rel_id_path,
                           output_rela2rela_id_path=output_rela2rela_id_path, ignore_not_mapped_edges=True, )


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S', )
    main()
