# GEBERT: Graph-Enriched Biomedical Entity Representation Transformer

This repository presents source code for pretraining BERT-based biomedical entity representation models on UMLS synonyms and concept graphs. The model is published at [CLEF 2023 conference](https://clef2023.clef-initiative.eu/). 

# Pre-trained models

We release two GEBERT versions that use GraphSAGE and GAT graph encoders, respectively. The checkpoints can be accessed via HuggingFace:

[GAT-GEBERT](https://huggingface.co/andorei/gebert_eng_gat): andorei/gebert_eng_gat


[GraphSAGE-GEBERT](https://huggingface.co/andorei/gebert_eng_graphsage): andorei/gebert_eng_graphsage

## Dependencies

To train GEBERT, we used Python version 3.10. Required packages are listed in requirements.txt file. [PyTorch geometric](https://pytorch-geometric.readthedocs.io) requires the torch-cluster, torch-scatter, and torch-sparse, so we recommend to install them prior to the installation of torch-geometric.

## Data

To train a model, you need to download a latest UMLS [release](https://www.nlm.nih.gov/research/umls/licensedcontent/umlsknowledgesources.html). In the original GEBERT paper we utilized the 2020AB version.

To train a GEBERT model, two data components are required:

* List of synonymous concept name pairs. 
* UMLS graph description

To obtain both synonyms and graph description, simply run create_positive_triplets_dataset.py script with appropriate environment variables:


```bash
python gebert/data/create_positive_triplets_dataset.py \
--mrconso "${UMLS_DIR}/MRCONSO.RRF" \
--mrrel "${UMLS_DIR}/MRREL.RRF" \
--langs "ENG" \
--output_dir $GRAPH_DATA_DIR 
```

## GEBERT pre-training

As examples of training scripts please see [graphsage_gebert_train_example.sh](https://github.com/Andoree/GEBERT/blob/main/graphsage_gebert_train_example.sh) and [gat_gebert_train_example.sh](https://github.com/Andoree/GEBERT/blob/main/gat_gebert_train_example.sh). To enable/disable multi-GPU training, please add/remove the "--parallel" flag. 

## Evaluation

For evaluation, we adopted the evaluation code and data from [https://github.com/insilicomedicine/Fair-Evaluation-BERT](https://github.com/insilicomedicine/Fair-Evaluation-BERT).


## Citation


```bibtex
@inproceedings{sakhovskiy2023gebert,
	title={Graph-Enriched Biomedical Entity Representation Transformer},
	author={Sakhovskiy, Andrey and Semenova, Natalia and Kadurin, Artur and Tutubalina, Elena},
	booktitle={Proceedings of the Fourteenth International Conference of the CLEF Association (CLEF 2023)},
	year={2023}
}
```
