"""
Moduł potoku przetwarzania danych (Klasyfikacja Wieloklasowa / Multiclass).

Uwaga metodyczna:
Zgodnie z dobrymi praktykami inżynierii oprogramowania, monolityczny potok danych 
został zrefaktoryzowany i podzielony na mniejsze, atomowe podfunkcje.
Taki podział zapewnia:
1. Reużywalność: Poszczególne transformacje mogą być wielokrotnie wykorzystywane w innych modułach.
2. Czytelność i testowalność: Kod jest o wiele łatwiejszy w utrzymaniu i debugowaniu.
3. Elastyczność: Główna funkcja pełni rolę czytelnego "orkiestratora".

Spis dostępnych funkcji atomowych (Krok po kroku):
1. load_data(filepath) - Wczytuje surowy zbiór danych z pliku CSV.
2. standardize_column_names(df) - Ujednolica nazwy kolumn, usuwając problematyczne znaki specjalne (np. nawiasy).
3. create_multiclass_target(df)  - Zachowuje oryginalne 4 klasy Severity i przesuwa indeks o 1 w dół (wartości 0, 1, 2, 3) 
                                   co jest wymagane przez bibliotekę scikit-learn i modele wieloklasowe XGBoost.
4. extract_time_features(df)  - Inżynieria cech: wyciąga godzinę, miesiąc z daty i tworzy flagi Is_Weekend, Is_Rush_Hour.
5. impute_missing_values(df)  - Bezpiecznie uzupełnia braki w danych (geograficznych, pogodowych) za pomocą mediany/mody.
6. remove_noise_columns(df)  - Usuwa zbędne kolumny niosące szum informacyjny (nieprzydatne dla algorytmów ML).
7. encode_features(df)  - Koduje zmienne tekstowe i logiczne na wartości numeryczne zrozumiałe dla modeli.
8. split_data(df, target_col)  - Dzieli zbiór na treningowy i testowy ze stratyfikacją klas i zwalnia zasoby pamięci operacyjnej.
9. prepare_data_multiclass(filepath)  - Orkiestrator: Wywołuje powyższe kroki sekwencyjnie i zwraca gotowe zbiory X/y.
"""

import pandas as pd
import numpy as np
import gc
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

def load_data(filepath):
    print(f"1. Loading data from file: {filepath}")
    df = pd.read_csv(filepath)
    print(f"Initial data shape: {df.shape[0]} rows.")
    return df

def standardize_column_names(df):
    print("2. Standardizing column names.")
    df.columns = (
        df.columns
        .str.replace('(', '_', regex=False)
        .str.replace(')', '', regex=False)
        .str.replace('%', 'pct', regex=False)
    )
    return df

def create_multiclass_target(df):
    print("3. Creating Multiclass target variable (Severity_Multi).")
    # Ponieważ klasy w Scikit-Learn / XGBoost są wymagane od 0,
    # przesuwamy etykiety o 1 w dół (1,2,3,4 -> 0,1,2,3).
    df['Severity_Multi'] = df['Severity'] - 1
    return df

def extract_time_features(df):
    print("4. Extracting time features.")
    df['Start_Time'] = pd.to_datetime(df['Start_Time'], errors='coerce')
    df = df.dropna(subset=['Start_Time'])
    # Celowo pomijamy ekstrakcję roku (df['Start_Time'].dt.year), 
    # aby uniknąć overfittingu do danych historycznych w docelowym symulatorze "What-If".
    df['Hour'] = df['Start_Time'].dt.hour
    df['Day_of_Week'] = df['Start_Time'].dt.dayofweek
    df['Month'] = df['Start_Time'].dt.month

    df['Is_Weekend'] = np.where(df['Day_of_Week'] >= 5, 1, 0)
    
    rush_hour_mask = (df['Is_Weekend'] == 0) & (((df['Hour'] >= 7) & (df['Hour'] <= 9)) | ((df['Hour'] >= 15) & (df['Hour'] <= 18)))
    df['Is_Rush_Hour'] = np.where(rush_hour_mask, 1, 0)
    return df

def impute_missing_values(df):
    print("5. Imputing missing values.")
    df['End_Lat'] = df['End_Lat'].fillna(df['Start_Lat'])
    df['End_Lng'] = df['End_Lng'].fillna(df['Start_Lng'])

    df['Precipitation_in'] = df['Precipitation_in'].fillna(
        df.groupby(['County', 'Month'])['Precipitation_in'].transform('median')
    )
    df['Precipitation_in'] = df['Precipitation_in'].fillna(0)

    weather_numeric_cols = [
        'Wind_Chill_F', 'Wind_Speed_mph', 'Visibility_mi', 
        'Humidity_pct', 'Temperature_F', 'Pressure_in'
    ]
    for col in weather_numeric_cols:
        df[col] = df[col].fillna(df[col].median())

    most_frequent_weather = df['Weather_Condition'].mode()[0]
    df['Weather_Condition'] = df['Weather_Condition'].fillna(most_frequent_weather)
    return df

def remove_noise_columns(df):
    print("6. Removing noise columns and residual missing rows.")
    cols_to_drop = [
        'ID', 'Description', 'Street', 'Zipcode', 'Weather_Timestamp', 
        'Airport_Code', 'City', 'Country', 'Timezone', 
        'Start_Time', 'End_Time', 'Severity'
    ]
    df = df.drop(columns=[col for col in cols_to_drop if col in df.columns], errors='ignore')

    df = df.dropna()
    print(f"Data shape after cleaning: {df.shape[0]} rows.")
    return df

def encode_features(df):
    print("7. Encoding boolean and categorical features.")
    bool_cols = df.select_dtypes(include=['bool']).columns
    for col in bool_cols:
        df[col] = df[col].astype(int)

    le = LabelEncoder()
    object_cols = df.select_dtypes(include=['object']).columns
    for col in object_cols:
        df[col] = df[col].astype(str)
        df[col] = le.fit_transform(df[col])
    return df

def split_data(df, target_col):
    print("8. Splitting data into train and test sets (80/20) with stratification.")
    X = df.drop(columns=[target_col])
    y = df[target_col]

    del df
    gc.collect()

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    print(f"Finished. X_train shape: {X_train.shape}, X_test shape: {X_test.shape} ")
    return X_train, X_test, y_train, y_test

def prepare_data_multiclass(filepath):
    df = load_data(filepath)
    df = standardize_column_names(df)
    df = create_multiclass_target(df)
    df = extract_time_features(df)
    df = impute_missing_values(df)
    df = remove_noise_columns(df)
    df = encode_features(df)
    
    return split_data(df, target_col='Severity_Multi')