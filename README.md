# Car Valuation AI

System inteligentnej analizy i wyceny pojazdów oparty na głębokich sieciach neuronowych (Deep Learning). Aplikacja wykorzystuje dwa niezależne modele predykcyjne, aby zapewnić najwyższą dokładność zarówno dla samochodów nowoczesnych, jak i klasyków.

## Główne Funkcje
- **Zaawansowana Analiza Wyposażenia**: Uwzględnia ponad 40 elementów wyposażenia dodatkowego przy wycenie.
- **Dynamiczne UI**: Interfejs zbudowany w `CustomTkinter` z obsługą wyszukiwania marek i modeli.
- **Precyzyjne Skalowanie**: Każdy parametr (przebieg, moc, rok) jest normalizowany dedykowanymi skalerami Scikit-learn.

## Struktura Projektu
- `python_files/main_gui.py` - Główna aplikacja użytkownika.
- `files_other/` - Skalery i dane dla modelu ogólnego (Total).
- `files_other_modern/` - Skalery dedykowane dla aut nowoczesnych.
- `json_files/` - Mapowania kategorii i Target Encoding.
- `car_valuation_model.keras` - Wytrenowane sieci neuronowe.

## Metodologia
Aplikacja wykorzystuje architekturę sieci neuronowej typu Sequential z warstwami Dropout, co zapobiega overfittingowi i pozwala na realną ocenę wartości rynkowej zamiast prostego odtwarzania bazy danych. Dane kategoryczne (marka, model) są przetwarzane przy użyciu techniki Target Encoding z wygładzaniem (Smoothing).

## Instalacja i Uruchomienie
1. Zainstaluj wymagane biblioteki:
   ```bash
   pip install pandas numpy tensorflow customtkinter joblib scikit-learn