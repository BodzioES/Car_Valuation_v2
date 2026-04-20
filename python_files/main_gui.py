import os
import sys
import customtkinter as ctk
import pandas as pd
import numpy as np
import joblib
import json
from tensorflow.keras.models import load_model

def get_resource_path(relative_path):
    try:
        base_path = sys._MEIPASS
    except Exception:
        base_path = os.path.abspath(".")
    return os.path.join(base_path, relative_path)

sys.path.append(get_resource_path('.'))
from translations import *

path_files = get_resource_path('files_other/')
json_path = get_resource_path('json_files/')

scaler_price = joblib.load(os.path.join(path_files, 'price_scaler.pkl'))
scaler_course = joblib.load(os.path.join(path_files, 'course_scaler.pkl'))
scaler_year = joblib.load(os.path.join(path_files, 'yearProduction_scaler.pkl'))
scaler_power = joblib.load(os.path.join(path_files, 'power_hp_scaler.pkl'))
scaler_capacity = joblib.load(os.path.join(path_files, 'capacity_cm3_scaler.pkl'))

with open(os.path.join(json_path, 'features_map.json'), 'r', encoding='utf-8') as f:
    features_map = json.load(f)
with open(os.path.join(json_path, 'mark_model_map.json'), 'r', encoding='utf-8') as f:
    mark_model_map = json.load(f)
with open(os.path.join(json_path, 'fuel_map.json'), 'r', encoding='utf-8') as f:
    fuel_map = json.load(f)
with open(os.path.join(json_path, 'transmission_map.json'), 'r', encoding='utf-8') as f:
    transmission_map = json.load(f)
with open(os.path.join(json_path, 'body_type_map.json'), 'r', encoding='utf-8') as f:
    body_type_map = json.load(f)

model_modern = load_model(get_resource_path('car_valuation_model_modern.keras'))
model_legacy = load_model(get_resource_path('car_valuation_model.keras'))

ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")

class CarValuationApp(ctk.CTk):
    def __init__(self):
        super().__init__()

        self.title("Car Valuation AI - PRO")
        self.geometry("900x900")

        self.label_title = ctk.CTkLabel(self, text="AI VEHICLE ANALYSIS SYSTEM", font=("Segoe UI", 26, "bold"))
        self.label_title.pack(pady=20)

        self.main_frame = ctk.CTkFrame(self)
        self.main_frame.pack(pady=10, padx=20, fill="both", expand=True)

        all_keys = list(mark_model_map.keys())
        brands = sorted(list(set([k.split('_')[0] for k in all_keys])))

        self.option_mark = self.create_option("Select Brand", brands, 0, 0)
        self.option_mark.configure(command=self.update_models)
        self.option_model = self.create_option("Select Model", ["First select brand"], 0, 1)

        self.entry_year = self.create_input("Year", 1, 0)
        self.entry_mileage = self.create_input("Mileage (km)", 1, 1)
        self.entry_power = self.create_input("Power (HP)", 2, 0)
        self.entry_capacity = self.create_input("Capacity (cm3)", 2, 1)

        self.option_fuel = self.create_option("Fuel Type", list(FUEL_MAP.keys()), 3, 0)
        self.option_trans = self.create_option("Transmission", list(TRANSMISSION_MAP.keys()), 3, 1)
        self.option_body = self.create_option("Body Type", list(BODY_MAP.keys()), 4, 0)
        self.option_accident = self.create_option("Accident-free", list(ACCIDENT_MAP.keys()), 4, 1)

        self.label_eq = ctk.CTkLabel(self.main_frame, text="Select Equipment:", font=("Segoe UI", 12, "bold"))
        self.label_eq.grid(row=5, column=0, columnspan=2, pady=(15, 5))

        self.scroll_frame = ctk.CTkScrollableFrame(self.main_frame, width=600, height=250)
        self.scroll_frame.grid(row=6, column=0, columnspan=2, padx=20, pady=10, sticky="nsew")

        self.checkboxes = {}
        for english_name in sorted(EQUIPMENT_MAP.keys()):
            cb = ctk.CTkCheckBox(self.scroll_frame, text=english_name)
            cb.pack(anchor="w", padx=10, pady=2)
            self.checkboxes[english_name] = cb

        self.btn_analyze = ctk.CTkButton(self, text="START AI VALUATION",
                                         command=self.analyze_car,
                                         font=("Segoe UI", 18, "bold"), height=50)
        self.btn_analyze.pack(pady=20)

        self.result_label = ctk.CTkLabel(self, text="Ready for Analysis", font=("Segoe UI", 20, "bold"))
        self.result_label.pack(pady=10)

    def create_input(self, placeholder, row, col):
        entry = ctk.CTkEntry(self.main_frame, placeholder_text=placeholder, width=280)
        entry.grid(row=row, column=col, padx=20, pady=10)
        return entry

    def create_option(self, label, values, row, col):
        option = ctk.CTkOptionMenu(self.main_frame, values=values, width=280)
        option.grid(row=row, column=col, padx=20, pady=10)
        option.set(label)
        return option

    def update_models(self, selected_brand):
        all_keys = list(mark_model_map.keys())
        relevant_models = sorted([k.split('_')[1] for k in all_keys if k.startswith(selected_brand + "_")])
        self.option_model.configure(values=relevant_models)
        self.option_model.set("Select Model")

    def analyze_car(self):
        try:
            input_mark = self.option_mark.get()
            input_model = self.option_model.get()
            input_year = int(self.entry_year.get())
            input_mileage = int(self.entry_mileage.get())
            input_power = int(self.entry_power.get())
            input_capacity = float(self.entry_capacity.get())

            input_fuel = FUEL_MAP.get(self.option_fuel.get())
            input_transmission = TRANSMISSION_MAP.get(self.option_trans.get())
            input_body = BODY_MAP.get(self.option_body.get())
            input_accident = ACCIDENT_MAP.get(self.option_accident.get())

            encoded_equip = np.zeros(len(features_map))
            for english_name, cb in self.checkboxes.items():
                if cb.get() == 1:
                    polish_name = EQUIPMENT_MAP[english_name]
                    if polish_name in features_map:
                        encoded_equip[features_map[polish_name]] = 1

            model = model_modern if input_year >= 2000 else model_legacy

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
            ]).reshape(1, -1)

            prediction_scaled = model.predict(input_data, verbose=0)
            final_price = scaler_price.inverse_transform(pd.DataFrame(prediction_scaled, columns=['price']))

            self.result_label.configure(
                text=f"ESTIMATED VALUE: {round(float(final_price[0][0]), 2)} PLN",
                text_color="#2ecc71"
            )

        except Exception as e:
            print(f"Error: {e}")
            self.result_label.configure(text="Error: Check Input Data", text_color="#e74c3c")

if __name__ == "__main__":
    app = CarValuationApp()
    app.mainloop()