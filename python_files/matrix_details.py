import json
import os
import joblib
import numpy as np
import pandas as pd
import psycopg2
from sklearn.preprocessing import MinMaxScaler

from db_config import get_db_connection

# Database connection setup
conn = get_db_connection()
cursor = conn.cursor()

# Load the predefined equipment features map to ensure consistent encoding
if os.path.exists('../json_files/features_map.json'):
    with open('../json_files/features_map.json', 'r') as f:
        features_map = json.load(f)

def get_item():
    """
    Encodes car equipment into a multi-hot binary matrix.
    Each car gets a row where 1 means a feature exists and 0 means it does not.
    """
    query = f"SELECT id_announcement, equipment FROM announcements"
    df_raw = pd.read_sql(query, conn)

    num_features = len(features_map)
    num_cars = len(df_raw)

    # Initialize a zero-filled dataframe for features
    encoded_features = pd.DataFrame(
        np.zeros((num_cars, num_features)),
        columns = list(features_map.keys()),
    )

    # Fill the matrix based on individual car equipment lists
    for i, row in df_raw.iterrows():
        car_equip = row['equipment']

        if isinstance(car_equip, str):
            car_equip = json.loads(car_equip)

        for item in car_equip:
            if item in features_map:
                encoded_features.at[i, item] = 1

    final_df = pd.concat([df_raw[['id_announcement']], encoded_features], axis=1)
    final_df.to_parquet('../files_other/training_data_encoded.parquet')

def get_price():
    """
    Normalizes vehicle prices using MinMaxScaler (range 0 to 1).
    Saves both the scaled data and the scaler object for later price reversal (inverse transform).
    """
    query = f"SELECT id_announcement, price FROM announcements"
    df_price = pd.read_sql(query, conn)

    scaler = MinMaxScaler()
    df_price['price_scaled'] = scaler.fit_transform(df_price[['price']])

    # Persist the scaler object to handle future price predictions in the GUI
    joblib.dump(scaler, '../files_other/price_scaler.pkl')
    df_price[['id_announcement', 'price_scaled']].to_parquet('../files_other/price_scaler.parquet')
    print("The prices was scaled and saved")

def get_course():
    """
    Cleans and scales mileage data. Ignores unrealistic values above 1,000,000 km.
    """
    query = f"SELECT id_announcement, course FROM announcements WHERE course IS NOT NULL AND course < 1000000"
    df_raw = pd.read_sql(query, conn)

    scaler = MinMaxScaler()
    df_raw['course'] = scaler.fit_transform(df_raw[['course']])

    joblib.dump(scaler, '../files_other/course_scaler.pkl')
    df_raw[['id_announcement', 'course']].to_parquet('../files_other/course_scaler.parquet')

def get_year():
    """
    Normalizes the production year into a 0-1 scale.
    """
    query = f"SELECT id_announcement, year_production FROM announcements"
    df_raw = pd.read_sql(query, conn)

    scaler = MinMaxScaler()
    df_raw['year_production'] = scaler.fit_transform(df_raw[['year_production']])

    joblib.dump(scaler, '../files_other/yearProduction_scaler.pkl')
    df_raw[['id_announcement','year_production']].to_parquet('../files_other/yearProduction_scaler.parquet')

def get_power():
    """
    Normalizes engine horsepower.
    """
    query = f"SELECT id_announcement, power_hp FROM announcements"
    df_raw = pd.read_sql(query, conn)

    scaler = MinMaxScaler()
    df_raw['power_hp'] = scaler.fit_transform(df_raw[['power_hp']])

    joblib.dump(scaler, '../files_other/power_hp_scaler.pkl')
    df_raw[['id_announcement','power_hp']].to_parquet('../files_other/power_hp_scaler.parquet')

def get_capacity():
    """
    Normalizes engine displacement (cm3).
    """
    query = f"SELECT id_announcement, capacity_cm3 FROM announcements"
    df_raw = pd.read_sql(query, conn)

    scaler = MinMaxScaler()
    df_raw['capacity_cm3'] = scaler.fit_transform(df_raw[['capacity_cm3']])

    joblib.dump(scaler, '../files_other/capacity_cm3_scaler.pkl')
    df_raw[['id_announcement', 'capacity_cm3']].to_parquet('../files_other/capacity_cm3_scaler.parquet')

def get_categories_data():
    """
    Converts categorical text data (Transmission, Body Type, Fuel) into One-Hot encoded vectors.
    """
    categories = ['transmission','body_type','fuel']
    query = f"SELECT id_announcement, {', '.join(categories)} FROM announcements"
    df_raw = pd.read_sql(query, conn)

    for cat in categories:
        with open(f"../json_files/{cat}_map.json", "r") as f:
            mapping = json.load(f)

        matrix = np.zeros((len(df_raw), len(mapping)), dtype=int)

        for i, val in enumerate(df_raw[cat]):
            if val in mapping:
                index = mapping[val]
                matrix[i, index] = 1

        df_final = pd.DataFrame(matrix, columns=[f"{cat}_{name}" for name in mapping.keys()])
        df_final['id_announcement'] = df_raw['id_announcement']
        df_final.to_parquet(f"../files_other/{cat}_data.parquet")

def get_accident():
    """
    Converts boolean accident-free status into simple integer (0 or 1).
    """
    query = f"SELECT id_announcement, accident_free FROM announcements"
    df_raw = pd.read_sql(query, conn)

    df_raw['accident_free'] = df_raw['accident_free'].astype(int)
    df_raw[['id_announcement', 'accident_free']].to_parquet('../files_other/accident_free_scaler.parquet')

def get_mark_model_data():
    """
    Performs Target Encoding for car make and model combinations.
    Uses 'Smoothing' to handle rare models by blending their mean price with the global average.
    """
    query = f"""
        SELECT id_announcement, CONCAT(mark, '_', model) AS mark_model, price
        FROM announcements
        WHERE mark is NOT NULL AND model IS NOT NULL
    """

    df_raw = pd.read_sql(query, conn)
    global_mean = df_raw['price'].mean()
    stats = df_raw.groupby('mark_model')['price'].agg(['count', 'mean']).reset_index()

    # Smoothing factor 'm' prevents overfitting for rare car models
    m = 10
    stats['smoothed_price'] = (stats['count'] * stats['mean'] + m * global_mean) / (stats['count'] + m)

    scaler = MinMaxScaler()
    stats['encoded_value'] = scaler.fit_transform(stats[['smoothed_price']])

    mark_model_map = dict(zip(stats['mark_model'], stats['encoded_value']))

    # Store mapping for application logic
    with open('../json_files/mark_model_map.json', 'w', encoding='utf-8') as f:
        json.dump(mark_model_map, f, ensure_ascii=False, indent=4)

    joblib.dump(scaler, '../files_other/mark_model_map_scaler.pkl')

    df_raw['mark_model_encoded'] = df_raw['mark_model'].map(mark_model_map)

    df_final = df_raw[['id_announcement', 'mark_model_encoded']]
    df_final.to_parquet('../files_other/mark_model_data.parquet')

if __name__ == "__main__":
    get_year()
    get_course()
    get_item()
    get_price()
    get_accident()
    get_categories_data()
    get_capacity()
    get_power()
    get_mark_model_data()
    print("Process finished")