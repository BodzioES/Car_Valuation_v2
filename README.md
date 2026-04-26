# 🚗 Car Valuation AI

![Python](https://img.shields.io/badge/Python-3.8%2B-blue?logo=python&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange?logo=tensorflow&logoColor=white)
![PostgreSQL](https://img.shields.io/badge/PostgreSQL-Database-blue?logo=postgresql&logoColor=white)
![CustomTkinter](https://img.shields.io/badge/GUI-CustomTkinter-lightgrey)

A project for estimating car prices using neural networks. It uses two separate models — one for newer cars (2000+) and one for older ones — because a single model wasn’t handling both cases very well.

---

## 🎥 Demo
*(You can drop a GIF or a short video here showing how the app works — selecting a car, choosing equipment, and getting a price prediction)*  
Example:  
`![App Demo](assets/demo.gif)`

---

## ⚙️ How it works

This isn’t just a GUI — there’s a full pipeline behind it:

1. **Data collection (scraping)**  
   A script goes through car listing websites, collects links, and saves raw HTML using `requests` and `BeautifulSoup`.

2. **Storage & cleaning**  
   The data gets cleaned (missing values, number formats, etc.) and stored in a local `PostgreSQL` database.

3. **Feature engineering**  
   - 99+ equipment features turned into one-hot vectors  
   - make & model encoded using target encoding (with smoothing to avoid overfitting rare cars)  
   - numeric values scaled with `MinMaxScaler`

4. **Model training**  
   The processed data is fed into a neural network (Sequential with dropout) to learn pricing patterns.

5. **Prediction (GUI)**  
   The user enters car details in the `CustomTkinter` app. Then:
   - the app picks the right model (modern vs older cars)  
   - input data is scaled  
   - price is predicted  
   - result is converted back to a readable value  

---

## ✨ Features
- two separate models instead of one
- takes a lot of equipment details into account
- simple GUI with quick make/model selection
- scraper can resume if it stops halfway

---

## 📁 Project structure
- `python_files/main_gui.py` – main GUI app  
- `python_files/data_download.py` – scraping + database logic  
- `files_other/` – scalers, encoded data, older car model  
- `files_other_modern/` – stuff for newer cars  
- `json_files/` – mappings, encoding data, scraper state  
- `*.keras` – trained models  

---

## 🚀 Setup

### 1. Requirements
- Python 3.8+ 
- PostgreSQL (only needed if you want to scrape and train your own data)

### 2. Install dependencies
```bash
pip install pandas numpy tensorflow customtkinter joblib scikit-learn psycopg2-binary beautifulsoup4 requests python-dotenv