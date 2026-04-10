import pandas as pd
import numpy as np
import joblib
import json
import os
from tensorflow.keras.models import load_model


def predict_car_price():
    try:
        model = load_model('car_valuation_model_modern.keras')

        path = '../files_other/'
        json_path = '../json_files/'

        scaler_price = joblib.load(f'{path}price_scaler.pkl')
        scaler_course = joblib.load(f'{path}course_scaler.pkl')
        scaler_year = joblib.load(f'{path}yearProduction_scaler.pkl')
        scaler_power = joblib.load(f'{path}power_hp_scaler.pkl')
        scaler_capacity = joblib.load(f'{path}capacity_cm3_scaler.pkl')

        with open(f'{json_path}features_map.json', 'r', encoding='utf-8') as f:
            features_map = json.load(f)
        with open(f'{json_path}mark_model_map.json', 'r', encoding='utf-8') as f:
            mark_model_map = json.load(f)
        with open(f'{json_path}fuel_map.json', 'r', encoding='utf-8') as f:
            fuel_map = json.load(f)
        with open(f'{json_path}transmission_map.json', 'r', encoding='utf-8') as f:
            transmission_map = json.load(f)
        with open(f'{json_path}body_type_map.json', 'r', encoding='utf-8') as f:
            body_type_map = json.load(f)

    except Exception as e:
        print(f"Błąd wczytywania zasobów: {e}")
        return

    print("\n--- SYSTEM WYCENY POJAZDÓW AI v2 ---")
    input_mark = input("Marka: ").strip()
    input_model = input("Model: ").strip()
    input_year = int(input("Rok produkcji: "))
    input_mileage = int(input("Przebieg (km): "))
    input_power = int(input("Moc (KM): "))
    input_capacity = float(input("Pojemność (cm3): "))
    input_fuel = input("Paliwo (np. Benzyna, Diesel, LPG): ").strip()
    input_transmission = input("Skrzynia (Manualna/Automatyczna): ").strip()
    input_body = input("Nadwozie (np. Sedan, Kombi, SUV): ").strip()
    input_accident = input("Bezwypadkowy? (Tak/Nie): ").strip().lower()

    print("\nWyposażenie (oddzielone przecinkiem, np. ABS, Klimatyzacja, Nawigacja):")
    input_equip = [item.strip() for item in input().split(",")]

    encoded_equip = np.zeros(len(features_map))
    for item in input_equip:
        if item in features_map:
            encoded_equip[features_map[item]] = 1

    s_year = scaler_year.transform(pd.DataFrame([[input_year]], columns=['year_production']))[0][0]
    s_course = scaler_course.transform(pd.DataFrame([[input_mileage]], columns=['course']))[0][0]
    s_power = scaler_power.transform(pd.DataFrame([[input_power]], columns=['power_hp']))[0][0]
    s_capacity = scaler_capacity.transform(pd.DataFrame([[input_capacity]], columns=['capacity_cm3']))[0][0]
    s_accident = 1 if input_accident == "tak" else 0

    key = f"{input_mark}_{input_model}"
    s_mark_model = mark_model_map.get(key, 0.5)

    def get_one_hot(val, mapping):
        vec = np.zeros(len(mapping))
        if val in mapping:
            vec[mapping[val]] = 1
        return vec

    vec_fuel = get_one_hot(input_fuel, fuel_map)
    vec_trans = get_one_hot(input_transmission, transmission_map)
    vec_body = get_one_hot(input_body, body_type_map)

    input_data = np.concatenate([
        encoded_equip,
        [s_course, s_year, s_power, s_capacity, s_accident],
        vec_fuel,
        vec_trans,
        vec_body,
        [s_mark_model]
    ])

    input_data = input_data.reshape(1, -1)

    prediction_scaled = model.predict(input_data, verbose=0)
    final_price = scaler_price.inverse_transform(pd.DataFrame(prediction_scaled, columns=['price']))

    print("\n" + "=" * 40)
    print(f"ANALIZA AI DLA: {input_mark} {input_model} ({input_year})")
    print(f"SUGEROWANA CENA RYNKOWA: {round(float(final_price[0][0]), 2)} PLN")
    print("=" * 40)


if __name__ == "__main__":
    predict_car_price()
