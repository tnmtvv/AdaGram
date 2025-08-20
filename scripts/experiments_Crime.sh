#!/bin/bash

# Set the GPU device for the first pipeline run (grid search)
CUDA_VISIBLE_DEVICES=3 python pipeline.py \
    --config ./configs/regression/config_Crime.yaml \
    --output Crime_grid_search.csv

# Update the config file with the best parameters found
# Note: No need to specify a GPU for this CPU-bound script
python update_config.py \
    --input-csv ./results/Crime_grid_search.csv \
    --config-yaml ./configs/regression/config_Crime.yaml

# Set the GPU device for the second pipeline run (with best params and more seeds)
CUDA_VISIBLE_DEVICES=3 python pipeline.py \
    --config ./configs/regression/config_Crime.yaml \
    --output Crime_seeds.csv
