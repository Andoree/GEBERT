import logging
import os.path
from argparse import ArgumentParser
from typing import List, Dict

import pandas as pd
from tqdm.auto import tqdm
import itertools
import random

from gebert.data.umls2graph import create_relations2id_dicts, create_cui2node_id_mapping, create_graph_files
from gebert.utils.io import read_mrrel, read_mrconso, write_strings


def create_cui2synonyms_list_mapping(cui_synonym_pair_strings: List[str]) -> Dict[str, List[str]]:
    cui2synonyms_list = {}
    for line in tqdm(cui_synonym_pair_strings):
        cui, name = line.split("||")
        if cui2synonyms_list.get(cui) is None:
            cui2synonyms_list[cui] = []
        cui2synonyms_list[cui].append(name)
    return cui2synonyms_list


def create_tradename_mapping(mrrel_df: pd.DataFrame, cui2synonyms_list: Dict[str, List[str]]) -> Dict[str, List[str]]:
    tradename_mapping = {}
    for idx, row in tqdm(mrrel_df.iterrows(), total=mrrel_df.shape[0]):
        if row["RELA"] == "has_tradename" or row["RELA"] == "tradename_of":
            cui_1, cui_2 = row["CUI1"], row["CUI2"]
            try:
                sfs = cui2synonyms_list[cui_2]
                tradename_mapping[cui_1] = sfs
            except:
                continue
    return tradename_mapping


def gen_pairs(input_list):
    return list(itertools.combinations(input_list, r=2))


def generate_positive_pairs_from_synonyms(concept_id2synonyms_list: Dict[str, List[str]], max_pairs_per_cui=50, ) -> \
        List[str]:
    pos_pairs = []
    for concept_id, synonyms_list in tqdm(concept_id2synonyms_list.items()):
        synonym_pairs = gen_pairs(synonyms_list)
        if len(synonym_pairs) > max_pairs_per_cui:
            synonym_pairs = random.sample(synonym_pairs, max_pairs_per_cui)
        for p in synonym_pairs:
            line = str(concept_id) + "||" + p[0] + "||" + p[1]
            pos_pairs.append(line)
    return pos_pairs


def generate_positive_pairs(mrconso_df: pd.DataFrame, mrrel_df: pd.DataFrame,
                            cui2node_id: Dict[str, int]) -> List[str]:
    """
    :param mrconso_df: MRCONSO.RRF's Dataframe
    :param mrrel_df: MRREL's Dataframe
    :return: List of positive (synonym) pairs: <concept_id, term_1, term_2>. '||' is the separator.
    """

    cui_term_pairs_list: List[str] = []
    for idx, row in tqdm(mrconso_df.iterrows()):
        cui, lang, synonym = row["CUI"], row["LAT"], row["STR"]
        row = cui + "||" + synonym.lower()
        cui_term_pairs_list.append(row)
    cui_term_pairs_list = list(set(cui_term_pairs_list))

    logging.info(f"{len(cui_term_pairs_list)} <CUI, synonym> concepts remaining after duplicates drop.")
    logging.info(f"<CUI, synonym> examples:\n" + '\n'.join(cui_term_pairs_list[:3]))

    cui2synonyms_list = create_cui2synonyms_list_mapping(cui_synonym_pair_strings=cui_term_pairs_list)
    logging.info(f"Created CUI to synonyms mapping, there are {len(cui2synonyms_list.keys())} entries")

    tradename_mapping: Dict[str, List[str]] = create_tradename_mapping(mrrel_df=mrrel_df,
                                                                       cui2synonyms_list=cui2synonyms_list)
    logging.info(f"Created tradename mapping, there are {len(tradename_mapping.keys())} entries")
    # adding tradenames
    for cui, synonyms_list in tradename_mapping.items():
        for s in synonyms_list:
            cui_term_pair = cui + "||" + s.lower()
            cui_term_pairs_list.append(cui_term_pair)
    logging.info(f"There are {len(cui_term_pairs_list)} <CUI, synonym> concepts after tradenames addition")
    cui_term_pairs_list = list(set(cui_term_pairs_list))
    logging.info(f"There are {len(cui_term_pairs_list)} <CUI, synonym> concepts with tradenames after duplicates drop")

    cui2synonyms_list = create_cui2synonyms_list_mapping(cui_synonym_pair_strings=cui_term_pairs_list)
    cui2synonyms_list = {cui: syns for cui, syns in cui2synonyms_list.items()}  # if cui2node_id.get(cui) is not None}
    # cui2node_id = {cui: node_id for node_id, cui in enumerate(sorted(cui2synonyms_list.keys()))}
    node_id2synonyms_list = {cui2node_id[cui]: synonyms_list for cui, synonyms_list in cui2synonyms_list.items()}
    pos_pairs = generate_positive_pairs_from_synonyms(concept_id2synonyms_list=node_id2synonyms_list, )

    return pos_pairs


def main(args):
    random.seed(42)
    output_dir = args.output_dir
    last_subdir = os.path.basename(output_dir.rstrip('/'))
    output_dir = output_dir.rstrip('/').rstrip(last_subdir)
    logging.info("Loading MRCONSO....")
    mrconso_df = read_mrconso(args.mrconso)
    logging.info(f"MRCONSO is loaded. There are {mrconso_df.shape[0]} rows")

    if args.ontology is not None:
        mrconso_df = mrconso_df[mrconso_df.SAB.isin(args.ontology)]
        last_subdir = '_'.join(args.ontology) + f"_{last_subdir}"
    logging.info(f"There are {mrconso_df.shape[0]} MRCONSO rows after ontology filtering")
    if args.langs is not None:
        mrconso_df = mrconso_df[mrconso_df.LAT.isin(args.langs)]
        last_subdir = '_'.join(args.langs) + f"_{last_subdir}"
    output_dir = os.path.join(output_dir, last_subdir)
    if not os.path.exists(output_dir) and output_dir != '':
        os.makedirs(output_dir)

    logging.info(f"There are {mrconso_df.shape[0]} MRCONSO rows after language filtering")
    mrconso_df["STR"].fillna('', inplace=True)
    filtered_mrconso_present_cuis_set = set(mrconso_df["CUI"].unique())

    logging.info("Loading MRREL....")
    # mrrel_df = read_mrrel(args.mrrel)
    mrrel_df = read_mrrel(args.mrrel)[["CUI1", "REL", "RELA", "CUI2"]]
    logging.info(f"Removing MRREL duplicated rows. There are {mrrel_df.shape[0]} rows with duplicates")
    mrrel_df.drop_duplicates(inplace=True)
    logging.info(f"Filtering MRREL by CUI1 and CUI2 fields. There are {mrrel_df.shape[0]} rows before filtering")
    mrrel_df = mrrel_df[(mrrel_df['CUI1'].isin(filtered_mrconso_present_cuis_set)) & (
        mrrel_df['CUI2'].isin(filtered_mrconso_present_cuis_set))]
    logging.info(f"Finished filtering MRREL by CUI1 and CUI2 fields. "
                 f"There are {mrrel_df.shape[0]} rows after filtering")
    rel2id, rela2id = create_relations2id_dicts(mrrel_df)
    cui2node_id = create_cui2node_id_mapping(mrconso_df=mrconso_df)

    logging.info("Creating graph files")
    output_node_id2terms_list_path = os.path.join(output_dir, "node_id2terms_list")
    output_node_id2cui_path = os.path.join(output_dir, "id2cui")
    output_edges_path = os.path.join(output_dir, "edges")
    output_rel2rel_id_path = os.path.join(output_dir, f"rel2id")
    output_rela2rela_id_path = os.path.join(output_dir, f"rela2id")

    create_graph_files(mrconso_df=mrconso_df, mrrel_df=mrrel_df, rel2id=rel2id, rela2id=rela2id,
                       cui2node_id=cui2node_id,
                       output_node_id2terms_list_path=output_node_id2terms_list_path,
                       output_node_id2cui_path=output_node_id2cui_path,
                       output_edges_path=output_edges_path, output_rel2rel_id_path=output_rel2rel_id_path,
                       output_rela2rela_id_path=output_rela2rela_id_path, ignore_not_mapped_edges=True, )
    pos_pairs = generate_positive_pairs(mrconso_df=mrconso_df, mrrel_df=mrrel_df, cui2node_id=cui2node_id)
    if args.split_val:
        output_train_pos_pairs_path = os.path.join(output_dir, f"train_pos_pairs")
        output_val_pos_pairs_path = os.path.join(output_dir, f"val_pos_pairs")

        train_proportion = args.train_proportion
        num_pos_pairs = len(pos_pairs)
        # Shuffling positive pairs
        random.shuffle(pos_pairs)
        num_train_pos_pairs = int(num_pos_pairs * train_proportion)
        train_pos_pairs = pos_pairs[:num_train_pos_pairs]
        val_pos_pairs = pos_pairs[num_train_pos_pairs:]
        logging.info(f"Positive pairs are split: {len(train_pos_pairs)} in train and {len(val_pos_pairs)} in val")
        logging.info(f"Train positive pair examples:\n" + '\n'.join(train_pos_pairs[:3]))
        logging.info(f"Val positive pair examples:\n" + '\n'.join(val_pos_pairs[:3]))
        write_strings(fpath=output_train_pos_pairs_path, strings_list=train_pos_pairs)
        write_strings(fpath=output_val_pos_pairs_path, strings_list=val_pos_pairs)
    else:
        random.shuffle(pos_pairs)
        output_pos_pairs_path = os.path.join(output_dir, f"train_pos_pairs")
        logging.info(f"Positive pair examples:\n" + '\n'.join(pos_pairs[:3]))
        write_strings(fpath=output_pos_pairs_path, strings_list=pos_pairs)
    # pos_pairs[:3]
    # with open('./training_file_umls2020aa_en_uncased_no_dup_pairwise_pair_th50.txt', 'w') as f:
    #     for line in pos_pairs:
    #         f.write("%s\n" % line)


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S', )
    parser = ArgumentParser()
    parser.add_argument('--mrconso', type=str)
    parser.add_argument('--mrrel', type=str)
    parser.add_argument('--langs', nargs='+', default=None)
    parser.add_argument('--split_val', action="store_true")
    parser.add_argument('--train_proportion', type=float)
    parser.add_argument('--ontology', default=None, nargs='+')
    parser.add_argument('--output_dir', type=str)
    arguments = parser.parse_args()

    main(arguments)
