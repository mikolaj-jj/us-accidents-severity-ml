import streamlit as st
import pandas as pd
import joblib
import numpy as np

st.set_page_config(
    page_title="Symulator What-If | Inżynieria Ruchu",
    page_icon="🚦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Wymagana definicja klasy dla autorskiego modelu, 
# aby Python mógł go odpakować z pliku .joblib
class CustomNewtonGBM:
    def __init__(self, n_estimators=10, learning_rate=0.1, max_depth=3, n_classes=4, class_weight='balanced'):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.n_classes = n_classes
        self.class_weight = class_weight 
        self.trees = [[] for _ in range(self.n_classes)]
        self.init_logits = None

    def _softmax(self, z):
        exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))
        return exp_z / np.sum(exp_z, axis=1, keepdims=True)

    def predict_proba(self, X):
        X_arr = np.array(X)
        N = len(X_arr)
        current_logits = np.tile(self.init_logits, (N, 1))
        
        for k in range(self.n_classes):
            for tree in self.trees[k]:
                current_logits[:, k] += self.learning_rate * tree.predict(X_arr)
                
        return self._softmax(current_logits)

    def predict(self, X):
        probs = self.predict_proba(X)
        return np.argmax(probs, axis=1)

# Ładowanie modeli i danych
@st.cache_resource
def load_xgboost():
    try: return joblib.load('xgboost_model.joblib')
    except: return None

@st.cache_resource
def load_lightgbm():
    try: return joblib.load('lightgbm_model.joblib')
    except: return None

@st.cache_resource
def load_random_forest():
    try: return joblib.load('random_forest_model.joblib')
    except: return None

@st.cache_resource
def load_custom_gbm():
    try: return joblib.load('custom_gbm_model.joblib')
    except: return None

@st.cache_data
def load_presets():
    try: return pd.read_csv('map_presets.csv')
    except: return pd.DataFrame()

xgb_model = load_xgboost()
lgb_model = load_lightgbm()
rf_model = load_random_forest()
custom_model = load_custom_gbm()
presets_df = load_presets()

# Ukrywamy prawdziwy target, bo aplikacja policzy go sobie sama na żywo
if not presets_df.empty and 'Target_Severity' in presets_df.columns:
    display_df = presets_df.drop(columns=['Target_Severity'])
else:
    display_df = presets_df.copy()

# Panel Boczny
with st.sidebar:
    st.title("Panel Sterowania", anchor=False)
    
    st.header("1. Wybór Silnika AI", anchor=False)
    model_choice = st.selectbox(
        "Wybierz architekturę:",
        [
            "XGBoost (Level-wise)", 
            "LightGBM (Leaf-wise)",
            "Random Forest (Baseline)",
            "Autorski Newton GBM"
        ],
        label_visibility="collapsed"
    )
    
    # Przełączanie wskaźnika w pamięci na wybrany model
    if model_choice == "XGBoost (Level-wise)": active_model = xgb_model
    elif model_choice == "LightGBM (Leaf-wise)": active_model = lgb_model
    elif model_choice == "Random Forest (Baseline)": active_model = rf_model
    else: active_model = custom_model
    
    st.header("2. Wybór Studium Przypadku", anchor=False)
    if not display_df.empty:
        preset_options = [f"Zdarzenie {i}" for i in range(len(display_df))]
        selected_preset_name = st.selectbox("Wybierz punkt z mapy:", preset_options, label_visibility="collapsed")
        
        selected_idx = int(selected_preset_name.split(" ")[1])
        current_data = display_df.iloc[selected_idx].to_dict()
    else:
        st.error("Brak pliku map_presets.csv. Uruchom skrypt eksportujący w Jupyter Notebook!")
        current_data = {}
        
    st.markdown("---")
    
    with st.expander("O Symulatorze", expanded=False):
        st.markdown("""
        **Cel Narzędzia:**  
        Aplikacja służy jako analityczny system wspomagania decyzji dla inżynierów ruchu i zarządców infrastruktury, pozwalając na optymalizację inwestycji w bezpieczeństwo drogowe.
        
        **Zasada Działania:**  
        System integruje dane przestrzenne z wyuczonymi modelami uczenia maszynowego. Po wybraniu przez użytkownika zidentyfikowanego zdażenia z mapy, silnik AI ewaluuje bazowe prawdopodobieństwo do jakiego typu wypadku doszło za pomocą danych warunków historycznych.
        
        **Symulacja "What-If":**  
        Rdzeniem aplikacji jest interaktywny panel testowania scenariuszy. Poprzez wirtualną modyfikację parametrów infrastruktury (np. instalacja sygnalizacji świetlnej, wytyczenie przejścia dla pieszych), użytkownik wymusza na modelu ponowną, natychmiastową predykcję. Pozwala to empirycznie oszacować stopień mitygacji ryzyka.
        
        **Wartość:**  
        Narzędzie demonstruje przejście od klasycznej analityki predykcyjnej do analityki preskryptywnej. Algorytmy ML nie pełnią tu jedynie roli biernych klasyfikatorów zagadnień, lecz aktywnie wspierają projektowanie bezpieczniejszej przestrzeni urbanistycznej.
        """)
    st.markdown("---")
    st.markdown("<div style='text-align: center; color: gray; font-size: 14px;'>Zaprojektowane i wdrożone przez:<br><b>Mikołaj Góralczyk</b></div>", unsafe_allow_html=True)

# Główny ekran aplikacji
st.title("Interaktywny Symulator Zdarzeń Drogowych", anchor=False)

st.markdown(f"### Lokalizacja: {selected_preset_name if not display_df.empty else 'Brak'}")

# Rysowanie mapy na 100% szerokości
if current_data:
    map_data = pd.DataFrame({
        'lat': [current_data.get('Start_Lat', 0)],
        'lon': [current_data.get('Start_Lng', 0)]
    })
    st.map(map_data, zoom=12)

st.markdown("---")

# Horyzontalny podział na Panel i Wykres
col_panel, col_wykres = st.columns([1.2, 1])

# Kontrolki do konfiguracji
with col_panel:
    st.subheader("Panel Inżyniera", anchor=False)
    st.write("Modyfikuj parametry, by obserwować reakcję modelu na wykresie słupkowym.")
    
    if current_data:
        p_col1, p_col2 = st.columns(2)
        
        with p_col1:
            st.markdown("**Infrastruktura drogowa**")
            val_signal = int(current_data.get('Traffic_Signal', 0))
            val_stop = int(current_data.get('Stop', 0))
            val_crossing = int(current_data.get('Crossing', 0))
            val_weekend = int(current_data.get('Is_Weekend', 0))
            
            new_signal = st.toggle("Sygnalizacja Świetlna", value=bool(val_signal))
            new_stop = st.toggle("Znak Stop", value=bool(val_stop))
            new_crossing = st.toggle("Przejście dla Pieszych", value=bool(val_crossing))
            new_weekend = st.toggle("Czy to weekend?", value=bool(val_weekend))
            
        with p_col2:
            st.markdown("**Środowisko i Czas**")
            val_temp = float(current_data.get('Temperature_F', 50.0))
            val_weather = int(current_data.get('Weather_Condition', 0))
            val_hour = int(current_data.get('Hour', 12))
            
            new_temp = st.slider("Temperatura (F)", min_value=-30.0, max_value=120.0, value=val_temp, step=1.0)
            new_hour = st.slider("Godzina", min_value=0, max_value=23, value=val_hour, step=1)
            new_weather = st.number_input("Zakodowana Pogoda", min_value=0, max_value=150, value=val_weather, step=1)
            
            with st.expander("Legenda Pogody"):
                leg_col1, leg_col2 = st.columns(2)
                with leg_col1:
                    st.markdown("""
                    <div style="font-size: 13px; line-height: 2;">
                    <b>5</b> - Pochmurno<br>
                    <b>6</b> - Pochmurno / Wiatr<br>
                    <b>14</b> - Mgła / Wiatr<br>
                    <b>78</b> - Płytka mgła
                    </div>
                    """, unsafe_allow_html=True)
                with leg_col2:
                    st.markdown("""
                    <div style="font-size: 13px; line-height: 2;">
                    <b>83</b> - Dym<br>
                    <b>89</b> - Śnieg z deszczem<br>
                    <b>96</b> - Burza z wiatrem
                    </div>
                    """, unsafe_allow_html=True)
            
        # Aktualizacja DataFrame'u pod spodem
        simulated_data = current_data.copy()
        simulated_data['Traffic_Signal'] = 1 if new_signal else 0
        simulated_data['Stop'] = 1 if new_stop else 0
        simulated_data['Crossing'] = 1 if new_crossing else 0
        simulated_data['Temperature_F'] = new_temp
        simulated_data['Weather_Condition'] = new_weather
        simulated_data['Hour'] = new_hour
        simulated_data['Is_Weekend'] = 1 if new_weekend else 0

        # Wymuszamy tę samą kolejność kolumn co w zbiorze treningowym
        input_df = pd.DataFrame([simulated_data])
        input_df = input_df[display_df.columns]
    else:
        st.warning("Brak danych do panelu.")
        input_df = None

# Wykres
with col_wykres:
    st.subheader("Wynik Symulacji", anchor=False)
    
    if active_model is None:
        st.error(f"Nie znaleziono pliku dla modelu: {model_choice}. Upewnij się, że wyeksportowałeś go z notatnika Jupyter!")
    elif input_df is not None:
        try:
            probabilities = active_model.predict_proba(input_df)[0]
            
            chart_data = pd.DataFrame({
                "Prawdopodobieństwo (%)": [p * 100 for p in probabilities],
                "Klasa": ["Severity 1", "Severity 2", "Severity 3", "Severity 4"]
            }).set_index("Klasa")
            
            predicted_class_idx = np.argmax(probabilities)
            predicted_class_name = f"Severity {predicted_class_idx + 1}"
            
            # Dynamiczny kolor tekstu (Czerwony dla Sev4, Zielony dla Sev1)
            if predicted_class_idx == 3: color = "#ff4b4b"
            elif predicted_class_idx == 2: color = "#ffa500"
            else: color = "#00cc66"
            
            st.markdown(f"### Najbardziej prawdopodobny scenariusz:<br>**<span style='color:{color}'>{predicted_class_name}</span>**", unsafe_allow_html=True)
            
            st.bar_chart(chart_data, height=250)
            
        except Exception as e:
            st.error(f"Wystąpił błąd podczas predykcji. Szczegóły: {e}")