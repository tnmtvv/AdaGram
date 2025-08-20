import pandas as pd
import numpy as np
from ruamel.yaml import YAML
import argparse

# Define optimizers that require special handling for different ranks
RANK_BASED_OPTIMIZERS = ['AdaGramPS', 'AdaGramFR_svd']

def _get_best_rank_based_params(df_rank_based):
    """
    Processes optimizers that have multiple ranks, finding the best parameters
    for each unique rank combination (including full-rank).
    
    Args:
        df_rank_based (pd.DataFrame): A DataFrame filtered to only contain
                                      optimizers defined in RANK_BASED_OPTIMIZERS.
                                      
    Returns:
        pd.DataFrame: A DataFrame with the best entry for each unique
                      optimizer-rank combination.
    """
    if df_rank_based.empty:
        return pd.DataFrame()
        
    df = df_rank_based.copy()

    def make_unique_opt_key(row):
        """Creates a unique key for an optimizer and its rank."""
        base_opt = row['base_optimizer']
        if pd.isna(row['rank']):
            return f'{base_opt}_fullrank'
        else:
            return f"{base_opt} rank {int(row['rank'])}"
            
    df['unique_opt_key'] = df.apply(make_unique_opt_key, axis=1)
    
    print(f"Found unique rank-based optimizer keys: {np.unique(df['unique_opt_key'])}")
    
    # Keep the first entry for each unique key (already sorted by best performance)
    df_best = df.drop_duplicates(subset=['unique_opt_key'], keep='first')
    return df_best


def find_and_update_best_params(df_grid_search, config_path):
    """
    Finds the best hyperparameters for each optimizer, creating separate entries
    for each rank of optimizers like AdaGramPS and AdaGramFR_svd, and then
    updates the YAML configuration file.
    """
    # 1. Filter for test mode results and create a base optimizer name
    df_test = df_grid_search[df_grid_search['mode'] == 'test'].copy()
    df_test['base_optimizer'] = df_test['optimizer'].apply(lambda x: x.split(' ')[0])
    
    # 2. Sort results to find the best (highest accuracy or lowest RMSE)
    yaml = YAML()
    yaml.preserve_quotes = True
    
    try:
        with open(config_path, 'r') as f:
            config = yaml.load(f)
    except FileNotFoundError:
        print(f"Error: Configuration file not found at {config_path}")
        return

    # Sort based on classification (accuracy) or regression (rmse)
    if config.get("dataset", {}).get("sparse_config", {}).get("if_class", False):
        df_sorted = df_test.sort_values(by=['accuracy', 'epoch'], ascending=[False, True])
        metric_col = 'accuracy'
    else:
        df_sorted = df_test.sort_values(by=['rmse', 'epoch'], ascending=[True, True])
        metric_col = 'rmse'
        
    # 3. Separate rank-based optimizers from standard ones
    is_rank_based = df_sorted['base_optimizer'].isin(RANK_BASED_OPTIMIZERS)
    df_rank_based = df_sorted[is_rank_based]
    df_standard = df_sorted[~is_rank_based]
    
    # 4. Process each group to find the best parameters
    df_rank_based_best = _get_best_rank_based_params(df_rank_based)
    df_standard_best = df_standard.drop_duplicates(subset=['base_optimizer'], keep='first')
    
    # 5. Combine results into a single DataFrame
    df_best_per_optimizer = pd.concat([df_rank_based_best, df_standard_best], ignore_index=True)
    
    print("\n--- Best performing parameters found for each optimizer/rank ---")
    print(df_best_per_optimizer[['base_optimizer', 'optimizer', 'rank', metric_col, 'epoch', 'lr', 'batch_size', 'eps']])
    print("-" * 70)
    
    # 6. Update data seeds in the configuration
    if 'data' in config and 'data_seeds' in config['data']:
        print("Updating data_seeds to [1, 2, 3, 4, 5, 10]")
        config['data']['data_seeds'] = [1, 2, 3, 4, 5, 10]
    else:
        print("Warning: 'data' or 'data_seeds' key not found in config. Skipping seed update.")

    # 7. Update optimizers section in the configuration
    
    # Disable the base templates for rank-based optimizers
    for opt_base_name in RANK_BASED_OPTIMIZERS:
        if opt_base_name in config.get('optimizers', {}):
            config['optimizers'][opt_base_name]['enabled'] = False

    for _, row in df_best_per_optimizer.iterrows():
        base_opt = row['base_optimizer']
        
        # Use the unique key for rank-based optimizers, otherwise use the base name
        if base_opt in RANK_BASED_OPTIMIZERS:
            opt_name = row['unique_opt_key']
        else:
            opt_name = base_opt
            
        # Create a new config entry if it doesn't exist
        if opt_name not in config.get('optimizers', {}):
            print(f"Optimizer '{opt_name}' not found in config. Adding new entry.")
            config.setdefault('optimizers', {})[opt_name] = {'enabled': True, 'testing': False}
            if base_opt in RANK_BASED_OPTIMIZERS:
                config['optimizers'][opt_name]['requires_rank'] = True

        print(f"Updating configuration for: {opt_name}")
        
        # Update parameters
        config['optimizers'][opt_name]['learning_rates'] = [row['lr']]
        config['optimizers'][opt_name]['batch_size'] = [int(row['batch_size'])]

        if base_opt in RANK_BASED_OPTIMIZERS and 'alpha' in row and pd.notna(row['alpha']):
            config['optimizers'][opt_name]['alphas'] = [row['alpha']]
        
        if 'eps' in row and pd.notna(row['eps']):
            config['optimizers'][opt_name]['eps'] = [row['eps']]
            
        # Update rank if required
        if config['optimizers'][opt_name].get('requires_rank'):
            rank_val = int(row['rank']) if pd.notna(row['rank']) else None
            config['optimizers'][opt_name]['ranks'] = [rank_val]
    
    # 8. Write the updated configuration back to the file
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
        print("-" * 70)
    except FileNotFoundError:
        print(f"Error: Input CSV file not found at {args.input_csv}")
        return
        
    # Find the best parameters and update the config file
    find_and_update_best_params(df_results, args.config_yaml)


if __name__ == "__main__":
    main()

