#!/bin/bash

CUDA_VISIBLE_DEVICES=0,1
GRAPH_DATA_DIR="data/graph_datasets/ENG_ENG_FULL"
BASE_TEXT_ENCODER="microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext"

python /home/etutubalina/graph_entity_linking/graphmel/graphmel/scripts/self_alignment_pretraining/train_graphsage_dgi_sapbert.py --train_dir=$GRAPH_DATA_DIR \
--text_encoder=$BASE_TEXT_ENCODER \
--dataloader_num_workers=0 \
--graphsage_num_outer_layers 1 \
--graphsage_num_inner_layers 3 \
--graphsage_num_hidden_channels 768 \
--graphsage_num_neighbors 3 \
--graphsage_dropout_p 0.3 \
--intermodal_loss_weight 0.1 \
--graph_loss_weight 0.1 \
--modality_distance "sapbert" \
--text_loss_weight 1.0 \
--use_intermodal_miner \
--intermodal_miner_margin 0.2 \
--remove_selfloops \
--max_length=32 \
--use_cuda \
--learning_rate=2e-5 \
--weight_decay=0.01  \
--batch_size=128 \
--num_epochs=1 \
--parallel \
--amp \
--random_seed=42 \
--loss="ms_loss" \
--use_miner \
--type_of_triplets "all" \
--miner_margin 0.2 \
--agg_mode "cls" \
--save_every_N_epoch=1 \
--output_dir="results/graphsage_gebert_english/"
