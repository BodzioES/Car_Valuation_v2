import pandas as pd
import numpy as np
import joblib
import json

from tensorflow.keras.models import load_model


def predict_car_price():
    try:
        model = load_model('car_valuation_model.keras')
        scaler_price = joblib.load('price_scaler.pkl')
        scaler_mileage = joblib.load('mileage_scaler.pkl')
        scaler_year = joblib.load('yearProduction_scaler.pkl')

        with open('features_map.json', 'r') as f:
            features_map = json.load(f)

    except Exception as e:
        print(e)
        return

    print("\n Please provide vehicle parameters:")
    input_mark = input("Mark car: ")
    input_model = input("Model car: ")
    input_year = int(input("Production year: "))
    input_mileage = int(input("Mileage (km): "))
    input_power = int(input("Power (hp): "))
    input_capacity = float(input("Capacity in cm3: "))
    input_fuel = input("Fuel (e.g.: gas,diesel,electric,hybrid): ")
    input_transmission = input("Automatic or manual: ")
    input_body_type = input("Body Type (e.g.: Seda, coupe, SUV) : ")
    input_accident= bool(input("Accident (True/False): "))
    print("\n List the equipment (separated by a comma, e.g.: Air Conditioning, Radio, ABS")
    input_equip = input().split(",")
    input_equip = [item.strip() for item in input_equip]

    scaled_year = scaler_year.transform([[input_year]])[0][0]
    scaled_mileage = scaler_mileage.transform([[input_mileage]])[0][0]

    encoded_equip = np.zeros(len(features_map))

    for item in input_equip:
        if item in features_map:
            index = list(features_map.keys()).index(item)
            encoded_equip[index] = 1

    input_data = np.append(encoded_equip, [scaled_year, scaled_mileage])
    input_data = input_data.reshape(1, -1)

    prediction_scaled = model.predict(input_data, verbose=0)

    final_price = scaler_price.inverse_transform(prediction_scaled)

    print("\n" + "=" * 30)
    print(f"SUGEROWANA CENA RYNKOWA: {round(float(final_price[0][0]), 2)} PLN")
    print("=" * 30)


if __name__ == "__main__":
    predict_car_price()



