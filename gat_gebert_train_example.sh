#!/bin/bash

CUDA_VISIBLE_DEVICES=0,1
GRAPH_DATA_DIR="data/graph_datasets/ENG_ENG_FULL"
BASE_TEXT_ENCODER="microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext"


python gebert/training/train_gat_gebert.py --train_dir=$GRAPH_DATA_DIR \
--text_encoder=$BASE_TEXT_ENCODER \
--dataloader_num_workers=0 \
--gat_num_outer_layers 1 \
--gat_num_inner_layers 3 \
--gat_num_hidden_channels 768 \
--gat_num_neighbors 3 \
--gat_num_att_heads 2 \
--gat_dropout_p 0.3 \
--gat_attention_dropout_p 0.1 \
--use_rel_or_rela "rel" \
--graph_loss_weight 0.1 \
--intermodal_loss_weight 0.1 \
--text_loss_weight 1.0 \
--intermodal_miner_margin 0.2 \
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
--output_dir="results/gat_gebert_english"

