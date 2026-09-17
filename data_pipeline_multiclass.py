import pandas as pd
import numpy as np
import gc
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

def przygotuj_dane_multiclass(filepath):
    #Funkcja wczytująca i czyszcząca pełen zbiór danych US Accidents.
    #Zachowuje oryginalną 4-stopniową klasyfikację Severity (Multiclass).

    print(f"1. Wczytywanie danych z pliku: {filepath}")
    df = pd.read_csv(filepath)
    print(f"Początkowy rozmiar danych: {df.shape[0]} wierszy.")

    print("2. Standaryzacja nazw kolumn.")
    df.columns = (
        df.columns
        .str.replace('(', '_', regex=False)
        .str.replace(')', '', regex=False)
        .str.replace('%', 'pct', regex=False)
    )

    print("3. Utrzymanie oryginalnej zmiennej docelowej (Severity: 1, 2, 3, 4).")
    # W odróżnieniu od wersji binarnej, tutaj targetem jest bezpośrednio Severity.
    # Ponieważ klasy w Scikit-Learn / niektorych bibliotekach bywają wymagane od 0,
    # możemy przesunąć je o 1 w dół (1,2,3,4 -> 0,1,2,3) dla kompatybilności algorytmów.
    df['Severity_Multi'] = df['Severity'] - 1

    print("4. Ekstrakcja cech czasowych.")
    df['Start_Time'] = pd.to_datetime(df['Start_Time'], errors='coerce')
    df = df.dropna(subset=['Start_Time'])

    df['Hour'] = df['Start_Time'].dt.hour
    df['Day_of_Week'] = df['Start_Time'].dt.dayofweek
    df['Month'] = df['Start_Time'].dt.month

    df['Is_Weekend'] = np.where(df['Day_of_Week'] >= 5, 1, 0)
    
    rush_hour_mask = (df['Is_Weekend'] == 0) & (((df['Hour'] >= 7) & (df['Hour'] <= 9)) | ((df['Hour'] >= 15) & (df['Hour'] <= 18)))
    df['Is_Rush_Hour'] = np.where(rush_hour_mask, 1, 0)

    print("5. Korekta i uzupełnianie braków danych.")
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

    print("6. Usuwanie zbędnych kolumn i szumu informacyjnego.")
    cols_to_drop = [
        'ID', 'Description', 'Street', 'Zipcode', 'Weather_Timestamp', 
        'Airport_Code', 'City', 'Country', 'Timezone', 
        'Start_Time', 'End_Time', 'Severity'
    ]
    df = df.drop(columns=[col for col in cols_to_drop if col in df.columns], errors='ignore')

    df = df.dropna()
    print(f"Rozmiar danych po czyszczeniu: {df.shape[0]} wierszy.")

    print("7. Kodowanie zmiennych logicznych i tekstowych.")
    bool_cols = df.select_dtypes(include=['bool']).columns
    for col in bool_cols:
        df[col] = df[col].astype(int)

    le = LabelEncoder()
    object_cols = df.select_dtypes(include=['object']).columns
    for col in object_cols:
        df[col] = df[col].astype(str)
        df[col] = le.fit_transform(df[col])

    print("8. Podział na zbiór treningowy i testowy (80/20) z zachowaniem stratyfikacji.")
    X = df.drop(columns=['Severity_Multi'])
    y = df['Severity_Multi']

    del df
    gc.collect()

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    
    print(f"Zakończono. Wymiary X_train: {X_train.shape}, X_test: {X_test.shape} ")
    return X_train, X_test, y_train, y_test