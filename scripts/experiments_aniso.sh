#!/bin/bash

# Set the GPU device for the first pipeline run (grid search)
CUDA_VISIBLE_DEVICES=6 python pipeline.py \
    --config ./configs/classification/config_Synth_Aniso.yaml \
    --output Aniso_grid_search_ext.csv

# Update the config file with the best parameters found
# Note: No need to specify a GPU for this CPU-bound script
python update_config.py \
    --input-csv ./results/Aniso_grid_search_ext.csv \
    --config-yaml ./configs/classification/config_Synth_Aniso.yaml

# Set the GPU device for the second pipeline run (with best params and more seeds)
CUDA_VISIBLE_DEVICES=6 python pipeline.py \
    --config ./configs/classification/config_Synth_Aniso.yaml \
    --output Aniso_seeds_ext.csv