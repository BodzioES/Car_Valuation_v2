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

if os.path.exists('features_map.json'):
    with open('features_map.json','r') as f:
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
    final_df.to_parquet('training_data_encoded.parquet')

def get_price():
    query = "SELECT id_announcement, price FROM announcements WHERE price > 500"
    df_price = pd.read_sql(query, conn)

    scaler = MinMaxScaler()

    df_price['price_scaled'] = scaler.fit_transform(df_price[['price']])

    joblib.dump(scaler, 'price_scaler.pkl')

    df_price[['id_announcement', 'price_scaled']].to_parquet('price_scaler.parquet')
    print("The prices was scaled and saved")


def get_course():
    query = "SELECT id_announcement, course FROM announcements WHERE course IS NOT NULL AND course < 1000000"
    df_raw = pd.read_sql(query, conn)

    scaler = MinMaxScaler()
    df_raw['course'] = scaler.fit_transform(df_raw[['course']])

    joblib.dump(scaler, 'course_scaler.pkl')
    df_raw[['id_announcement', 'course']].to_parquet('course_scaler.parquet')

def get_year():
    query = "SELECT id_announcement, year_production FROM announcements WHERE year_production > 1000"
    df_raw = pd.read_sql(query, conn)

    scaler = MinMaxScaler()
    df_raw['year_production'] = scaler.fit_transform(df_raw[['year_production']])

    joblib.dump(scaler, 'yearProduction_scaler.pkl')
    df_raw[['id_announcement','year_production']].to_parquet('yearProduction_scaler.parquet')


if __name__ == "__main__":
    get_year()
    get_course()
    get_item()
    get_price()
    print("Process finished")

