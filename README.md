# GEBERT: Graph-Enriched Biomedical Entity Representation Transformer

This repository presents source code for pretraining BERT-based biomedical entity representation models on UMLS synonyms and concept graphs. The model is published at [CLEF 2023 conference] (https://clef2023.clef-initiative.eu/). 

## Data

To train a model, you need to download a latest UMLS [release](https://www.nlm.nih.gov/research/umls/licensedcontent/umlsknowledgesources.html). In the original GEBERT paper we utilized the 2020AB version.

To train a GEBERT model, two data components are required:

* List of synonymous concept name pairs. 
* UMLS graph description

To obtain both synonyms and graph description, simply run with appropriate environment variables:

```bash
python gebert/data/create_positive_triplets_dataset.py \
--mrconso "${UMLS_DIR}/MRCONSO.RRF" \
--mrrel "${UMLS_DIR}/MRREL.RRF" \
--langs "ENG" \
--output_dir $GRAPH_DATA_DIR 
```

## Citation

```bibtex
@inproceedings{sakhovskiy2023gebert,
	title={Graph-Enriched Biomedical Entity Representation Transformer},
	author={Sakhovskiy, Andrey and Semenova, Natalia and Kadurin, Artur and Tutubalina, Elena},
	booktitle={Proceedings of the Fourteenth International Conference of the CLEF Association (CLEF 2023)},
	pages={},
	month = ,
	year={2023}
}
```
