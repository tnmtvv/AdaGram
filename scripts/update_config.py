import pandas as pd
import numpy as np
from ruamel.yaml import YAML
import os
import argparse

def find_and_update_best_params(df_grid_search, config_path):
    """
    Finds the best hyperparameters for each optimizer based on the highest
    accuracy (and then earliest epoch) and updates a YAML configuration file 
    with these parameters, using block-style formatting for lists.

    Args:
        df_grid_search (pd.DataFrame): DataFrame containing experiment results.
        config_path (str): The file path to the YAML configuration file.
    """
    # 1. Filter for test results and extract base optimizer name
    df_test = df_grid_search[df_grid_search['mode'] == 'test'].copy()
    df_test['base_optimizer'] = df_test['optimizer'].apply(lambda x: x.split(' ')[0])

    # 2. Sort by accuracy (descending) and then by epoch (ascending)
    df_sorted = df_test.sort_values(by=['accuracy', 'epoch'], ascending=[False, True])

    # 3. Get the single best parameter set for each base optimizer
    df_best_per_optimizer = df_sorted.drop_duplicates(subset=['base_optimizer'], keep='first')
    
    print("--- Best performing parameters found for each optimizer ---")
    print(df_best_per_optimizer[['base_optimizer', 'optimizer', 'accuracy', 'epoch', 'lr', 'batch_size', 'eps', 'rank']])
    print("-" * 60)

    # 4. Load the YAML configuration file and set the desired style
    yaml = YAML()
    yaml.preserve_quotes = True
    yaml.default_flow_style = True

    try:
        with open(config_path, 'r') as f:
            config = yaml.load(f)
    except FileNotFoundError:
        print(f"Error: Configuration file not found at {config_path}")
        return
    
    if 'data' in config and 'data_seeds' in config['data']:
        print("Updating data_seeds to [1, 2, 3, 4, 5]")
        config['data']['data_seeds'] = [1, 2, 3, 4, 5]
    else:
        print("Warning: 'data' or 'data_seeds' key not found in config. Skipping seed update.")

    # 5. Iterate through the best results and update the config
    for _, row in df_best_per_optimizer.iterrows():
        opt_name = row['base_optimizer'] 
        
        if opt_name in config['optimizers']:
            print(f"Updating configuration for: {opt_name}")
            
            # Update parameters
            config['optimizers'][opt_name]['learning_rates'] = [row['lr']]
            config['optimizers'][opt_name]['batch_size'] = [int(row['batch_size'])]
            
            if pd.notna(row['eps']) and str(row['eps']).lower() != 'nan':
                config['optimizers'][opt_name]['eps'] = [row['eps']]
            
            if config['optimizers'][opt_name].get('requires_rank') and pd.notna(row['rank']):
                config['optimizers'][opt_name]['ranks'] = [int(row['rank'])]
        else:
            print(f"Warning: Optimizer '{opt_name}' from results not found in config. Skipping.")

    # 6. Write the updated configuration back to the file
    try:
        with open(config_path, 'w') as f:
            yaml.dump(config, f)
        print(f"\nConfiguration file '{config_path}' has been successfully updated.")
    except Exception as e: 
        print(f"Error writing to configuration file: {e}")

def main():
    """Main function to parse arguments and run the script."""
    parser = argparse.ArgumentParser(
        description="Find best hyperparameters from a grid search CSV and update a YAML config file."
    )
    parser.add_argument(
        "--input-csv",
        required=True,
        help="Path to the input CSV file containing grid search results."
    )
    parser.add_argument(
        "--config-yaml",
        required=True,
        help="Path to the YAML configuration file to update."
    )
    
    args = parser.parse_args()

    # Load the CSV file
    try:
        df_results = pd.read_csv(args.input_csv)
        print(f"Successfully loaded {args.input_csv}. Showing first 15 rows:")
        print(df_results.head(15))
        print("-" * 60)
    except FileNotFoundError:
        print(f"Error: Input CSV file not found at {args.input_csv}")
        return
        
    # Find the best parameters and update the config file
    find_and_update_best_params(df_results, args.config_yaml)

if __name__ == "__main__":
    main()
