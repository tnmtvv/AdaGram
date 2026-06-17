#!/bin/bash

# Set the GPU device for the first pipeline run (grid search)
CUDA_VISIBLE_DEVICES=4 python pipeline.py \
    --config ./configs/classification/config_Splice.yaml \
    --output Splice_grid_search.csv

# Update the config file with the best parameters found
# Note: No need to specify a GPU for this CPU-bound script
# python update_config.py \
#     --input-csv ./results/Splice_grid_search.csv \
#     --config-yaml ./configs/classification/config_Splice.yaml

# # Set the GPU device for the second pipeline run (with best params and more seeds)
# CUDA_VISIBLE_DEVICES=4 python pipeline.py \
#     --config ./configs/classification/config_Splice.yaml \
#     --output Splice_seeds.csv
