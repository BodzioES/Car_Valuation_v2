import os
import pandas as pd

def create_dataset():
    base_files = 'training_data_encoded.parquet'
    master_df = pd.read_parquet(base_files)

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
        '../files_other/body_type_data.parquet'
    ]

    for module in modules:
        if os.path.exists(module):
            mod_df = pd.read_parquet(module)
            master_df = pd.merge(master_df, mod_df, how='inner', on='id_announcement')
        else:
            print(f"{module} does not exist")

    master_df = master_df.dropna()

    final_dataset = master_df.drop(columns=['id_announcement'])
    final_dataset.to_parquet('../files_other/final_dataset.parquet')


if __name__ == '__main__':
    create_dataset()



