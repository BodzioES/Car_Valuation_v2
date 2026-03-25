import json
import os

import joblib
import numpy as np
import pandas as pd
import psycopg2
from sklearn.preprocessing import MinMaxScaler

conn = psycopg2.connect(
        host="localhost",
        database="valuation_db",
        user="postgres",
        password="lorakium1515",
)
cursor = conn.cursor()

if os.path.exists('../json_files/features_map.json'):
    with open('../json_files/features_map.json', 'r') as f:
        features_map = json.load(f)

def get_item():
    query = "SELECT id_announcement, equipment FROM announcements"
    df_raw = pd.read_sql(query, conn)

    num_features = len(features_map)
    num_cars = len(df_raw)

    encoded_features = pd.DataFrame(
        np.zeros((num_cars, num_features)),
        columns = list(features_map.keys()),
    )

    for i, row in df_raw.iterrows():
        car_equip = row['equipment']

        if isinstance(car_equip,str):
            car_equip = json.loads(car_equip)

        for item in car_equip:
            if item in features_map:
                encoded_features.at[i, item] = 1

    final_df = pd.concat([df_raw[['id_announcement']], encoded_features], axis=1)
    final_df.to_parquet('../files_other/training_data_encoded.parquet')

def get_price():
    query = "SELECT id_announcement, price FROM announcements WHERE price > 500"
    df_price = pd.read_sql(query, conn)

    scaler = MinMaxScaler()

    df_price['price_scaled'] = scaler.fit_transform(df_price[['price']])

    joblib.dump(scaler, 'price_scaler.pkl')

    df_price[['id_announcement', 'price_scaled']].to_parquet('../files_other/price_scaler.parquet')
    print("The prices was scaled and saved")

def get_course():
    query = "SELECT id_announcement, course FROM announcements WHERE course IS NOT NULL AND course < 1000000"
    df_raw = pd.read_sql(query, conn)

    scaler = MinMaxScaler()
    df_raw['course'] = scaler.fit_transform(df_raw[['course']])

    joblib.dump(scaler, 'course_scaler.pkl')
    df_raw[['id_announcement', 'course']].to_parquet('../files_other/course_scaler.parquet')

def get_year():
    query = "SELECT id_announcement, year_production FROM announcements WHERE year_production > 1000"
    df_raw = pd.read_sql(query, conn)

    scaler = MinMaxScaler()
    df_raw['year_production'] = scaler.fit_transform(df_raw[['year_production']])

    joblib.dump(scaler, 'yearProduction_scaler.pkl')
    df_raw[['id_announcement','year_production']].to_parquet('../files_other/yearProduction_scaler.parquet')

def get_power():
    query = "SELECT id_announcement, power_hp FROM announcements"
    df_raw = pd.read_sql(query, conn)

    scaler = MinMaxScaler()
    df_raw['power_hp'] = scaler.fit_transform(df_raw[['power_hp']])

    joblib.dump(scaler, 'power_hp_scaler.pkl')
    df_raw[['id_announcement','power_hp']].to_parquet('../files_other/power_hp_scaler.parquet')

def get_capacity():
    query = "SELECT id_announcement, capacity_cm3 FROM announcements"
    df_raw = pd.read_sql(query, conn)

    scaler = MinMaxScaler()
    df_raw['capacity_cm3'] = scaler.fit_transform(df_raw[['capacity_cm3']])

    joblib.dump(scaler, 'capacity_cm3_scaler.pkl')
    df_raw[['id_announcement', 'capacity_cm3']].to_parquet('../files_other/capacity_cm3_scaler.parquet')

def get_categories_data():
    categories = ['mark','transmission','body_type','fuel']

    query = f"SELECT id_announcement, {', '.join(categories)} FROM announcements"
    df_raw = pd.read_sql(query, conn)

    for cat in categories:
        with open(f"../json_files/{cat}_map.json","r") as f:
            mapping = json.load(f)

        matrix = np.zeros((len(df_raw),len(mapping)), dtype=int)

        for i, val in enumerate(df_raw[cat]):
            if val in mapping:
                index = mapping[val]
                matrix[i, index] = 1

        df_final = pd.DataFrame(matrix, columns=[f"{cat}_{name}" for name in mapping.keys()])
        df_final['id_announcement'] = df_raw['id_announcement']
        df_final.to_parquet(f"../files_other/{cat}_data.parquet")

def get_accident():
    query = "SELECT id_announcement, accident_free FROM announcements"
    df_raw = pd.read_sql(query, conn)

    df_raw['accident_free'] = df_raw['accident_free'].astype(int)

    df_raw[['id_announcement', 'accident_free']].to_parquet('../files_other/accident_free_scaler.parquet')


if __name__ == "__main__":
    get_year()
    get_course()
    get_item()
    get_price()
    get_accident()
    get_categories_data()
    get_capacity()
    get_power()
    print("Process finished")

