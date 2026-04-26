import os
import pandas as pd


def create_dataset():
    """
    Merges all individual feature files into a single unified dataset for model training.
    Uses 'id_announcement' as the common key to align data from different sources.
    """
    # Define the primary file containing encoded equipment features
    base_files = '../files_other/training_data_encoded.parquet'
    if not os.path.exists(base_files):
        print(f"Critical error: {base_files} does not exist")
        return

    master_df = pd.read_parquet(base_files)

    # List of all feature modules to be merged
    modules = [
        '../files_other/price_scaler.parquet',
        '../files_other/course_scaler.parquet',
        '../files_other/yearProduction_scaler.parquet',
        '../files_other/power_hp_scaler.parquet',
        '../files_other/capacity_cm3_scaler.parquet',
        '../files_other/accident_free_scaler.parquet',
        '../files_other/fuel_data.parquet',
        '../files_other/mark_data.parquet',
        '../files_other/transmission_data.parquet',
        '../files_other/body_type_data.parquet',
        '../files_other/mark_model_data.parquet',
    ]

    # Iteratively merge each module into the master dataframe using an inner join
    for module in modules:
        if os.path.exists(module):
            mod_df = pd.read_parquet(module)
            # 'how=inner' ensures we only keep cars that exist in all files
            master_df = pd.merge(master_df, mod_df, how='inner', on='id_announcement')
        else:
            print(f"Warning: {module} does not exist. Skipping...")

    # Data cleaning: remove any rows with missing values after merging
    master_df = master_df.dropna()

    # Final preparation: remove the unique identifier (ID) as the neural network
    # should only see mathematical features, not database keys.
    final_dataset = master_df.drop(columns=['id_announcement'])

    # Save the consolidated training data
    final_dataset.to_parquet('../files_other/final_dataset.parquet')
    print(f"Success: final_dataset.parquet has been created with {len(final_dataset)} records.")


if __name__ == '__main__':
    create_dataset()