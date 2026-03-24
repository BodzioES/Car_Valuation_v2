import pandas as pd

def create_dataset():
    try:
        # marka
        # model
        df_year = pd.read_parquet('yearProduction_scaler.parquet')
        df_mileage = pd.read_parquet('course_scaler.parquet')
        # power_h -------------------------------------
        # capacity_cm3
        # fuel
        # transmission
        # body_type
        # accident_free
        df_price = pd.read_parquet('price_scaler.parquet')
        df_equip = pd.read_parquet('training_data_encoded.parquet')

    except FileNotFoundError as e:
        print(e)
        return

    master_df = pd.merge(df_equip, df_price, on='id_announcement', how='inner')

    master_df = pd.merge(master_df, df_year, on='id_announcement', how='inner')

    master_df = pd.merge(master_df, df_mileage, on='id_announcement', how='inner')

    initial_count = len(master_df)
    master_df = master_df.dropna()
    final_count = len(master_df)

    if initial_count > final_count:
        print(f"Odrzucono {initial_count - final_count} niepełnych ogłoszeń.")

    if 'id_announcement' in master_df.columns:
        master_df = master_df.drop(columns=['id_announcement'])

    output_name = 'final_dataset.parquet'
    master_df.to_parquet(output_name)

    print("Success")

if __name__ == '__main__':
    create_dataset()



