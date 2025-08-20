#!/bin/bash

# Set the GPU device for the first pipeline run (grid search)
CUDA_VISIBLE_DEVICES=2 python pipeline.py \
    --config ./configs/classification/config_AU.yaml \
    --output AU_grid_search.csv

# # Update the config file with the best parameters found
# # Note: No need to specify a GPU for this CPU-bound script
# python update_config.py \
#     --input-csv ./results/AU_grid_search.csv \
#     --config-yaml ./configs/classification/config_AU.yaml

# # Set the GPU device for the second pipeline run (with best params and more seeds)
# CUDA_VISIBLE_DEVICES=2 python pipeline.py \
#     --config ./configs/classification/config_AU.yaml \
#     --output AU_seeds.csv
