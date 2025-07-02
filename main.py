# app.py
import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from sklearn.cluster import AgglomerativeClustering
import seaborn as sns

st.set_page_config(page_title="Análisis avanzado de acciones", layout="wide")

st.title("📈 Estrategia de alineamiento temporal y patrones en acciones")

# 1. Selección de acciones y parámetros
st.sidebar.header("Configuración")
default_tickers = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "META", "NVDA"]
tickers = st.sidebar.text_input("Tickers separados por coma", value=",".join(default_tickers))
tickers = [t.strip().upper() for t in tickers.split(",") if t.strip() != ""]
start_date = st.sidebar.date_input("Fecha de inicio", value=datetime.today()-timedelta(days=730))
end_date = st.sidebar.date_input("Fecha de fin", value=datetime.today())
max_lag = st.sidebar.slider("Máximo desfase (lag, días)", 1, 30, 10)
min_corr = st.sidebar.slider("Correlación mínima para sugerencias", 0.1, 1.0, 0.4, step=0.05)

st.write(f"**Acciones seleccionadas:** {', '.join(tickers)}")
st.write(f"Período: {start_date} a {end_date}")

# 2. Descarga de datos
@st.cache_data(show_spinner=False)
def load_data(tickers, start, end):
    data = yf.download(tickers, start=start, end=end, progress=False)["Close"]
    if isinstance(data, pd.Series):  # Solo un ticker
        data = data.to_frame()
    data = data.dropna(how="all")  # Borra días sin datos
    return data

try:
    df = load_data(tickers, start_date, end_date)
    st.write("Datos históricos descargados (vista previa):")
    st.dataframe(df.head())
except Exception as e:
    st.error("Error al descargar datos: " + str(e))
    st.stop()

# 3. Preprocesamiento y transformación
log_returns = np.log(df / df.shift(1)).dropna()
st.subheader("Retornos logarítmicos (vista previa)")
st.dataframe(log_returns.head())

# 4. Análisis de correlación cruzada desplazada (lead-lag)
st.subheader("Correlación desplazada entre pares de acciones")
results = []
heatmap_matrix = np.zeros((len(tickers), len(tickers)))
lags_matrix = np.zeros((len(tickers), len(tickers)))

def cross_correlation(a, b, max_lag):
    lags = range(-max_lag, max_lag + 1)
    cors = []
    for lag in lags:
        if lag < 0:
            cor = a[:lag].corr(b[-lag:])
        elif lag > 0:
            cor = a[lag:].corr(b[:-lag])
        else:
            cor = a.corr(b)
        cors.append(cor)
    return list(lags), cors

for i, t1 in enumerate(tickers):
    for j, t2 in enumerate(tickers):
        if i >= j:
            continue
        lags, cors = cross_correlation(log_returns[t1], log_returns[t2], max_lag)
        idx_max = np.nanargmax(np.abs(cors))
        lag_opt = lags[idx_max]
        cor_opt = cors[idx_max]
        results.append({
            "Acción 1": t1, "Acción 2": t2,
            "Lag óptimo": lag_opt,
            "Correlación (máx abs)": cor_opt
        })
        heatmap_matrix[i, j] = cor_opt
        lags_matrix[i, j] = lag_opt

# 5. Visualización interactiva
results_df = pd.DataFrame(results)
st.dataframe(results_df.style.background_gradient(cmap="Blues"), height=250)

# Heatmap de correlaciones máximas absolutas
st.markdown("#### Mapa de calor de correlaciones máximas (por lag óptimo)")
fig, ax = plt.subplots(figsize=(8, 6))
sns.heatmap(np.abs(heatmap_matrix), annot=True, xticklabels=tickers, yticklabels=tickers, ax=ax, cmap="coolwarm", cbar_kws={'label': '|Correlación máxima|'})
plt.title("Heatmap de correlaciones máximas (absolutas)")
st.pyplot(fig)

# 6. Visualización de lead-lag para pares específicos
st.markdown("### Explora el alineamiento de pares específicos")
pair = st.selectbox("Selecciona un par para analizar", [(row['Acción 1'], row['Acción 2']) for idx, row in results_df.iterrows()])
if pair:
    t1, t2 = pair
    lags, cors = cross_correlation(log_returns[t1], log_returns[t2], max_lag)
    idx_max = np.nanargmax(np.abs(cors))
    lag_opt = lags[idx_max]

    st.markdown(f"#### Correlación desplazada: {t1} vs {t2}")
    fig2, ax2 = plt.subplots()
    ax2.plot(lags, cors, marker='o')
    ax2.axvline(lag_opt, color='r', linestyle='--', label=f'Lag óptimo: {lag_opt}')
    ax2.set_xlabel("Desfase (días)")
    ax2.set_ylabel("Correlación")
    ax2.legend()
    st.pyplot(fig2)

    st.markdown("**Visualización de series alineadas**")
    s1 = log_returns[t1].shift(-lag_opt) if lag_opt > 0 else log_returns[t1]
    s2 = log_returns[t2] if lag_opt > 0 else log_returns[t2].shift(lag_opt)
    comp_df = pd.DataFrame({t1: s1, t2: s2}).dropna()
    st.line_chart(comp_df)

# 7. Descubrimiento de grupos (clustering) por patrones de correlación máxima
st.subheader("Descubrimiento de grupos de acciones similares (clustering)")

# Creamos matriz de distancias: 1 - |correlación máxima|
distance_matrix = 1 - np.abs(heatmap_matrix + heatmap_matrix.T)
np.fill_diagonal(distance_matrix, 0)
if len(tickers) > 2:
    model = AgglomerativeClustering(n_clusters=None, distance_threshold=0.8)
    clusters = model.fit_predict(distance_matrix)
    cluster_df = pd.DataFrame({"Ticker": tickers, "Cluster": clusters})
    st.dataframe(cluster_df.sort_values("Cluster"))
    st.markdown("Los tickers en el mismo cluster muestran patrones de retorno similares en alguna ventana temporal (con lag).")
else:
    st.info("Agrega más de dos acciones para ver clustering.")

# 8. Sugerencias automáticas de patrones interesantes
st.subheader("Sugerencias de patrones detectados (para exploración/modelado)")
suggestions = results_df[np.abs(results_df["Correlación (máx abs)"]) > min_corr]
if suggestions.empty:
    st.info("No se encontraron alineamientos muy significativos con los parámetros actuales.")
else:
    for _, row in suggestions.iterrows():
        signo = "anticipa" if row['Lag óptimo'] > 0 else "sigue"
        st.success(f"{row['Acción 1']} {signo} el comportamiento de {row['Acción 2']} por {abs(int(row['Lag óptimo']))} días (correlación {row['Correlación (máx abs)']:.2f})")
        st.markdown(f"> Puedes probar modelar **{row['Acción 2']}** usando retornos de **{row['Acción 1']}** con desfase de {abs(int(row['Lag óptimo']))} días.")

st.markdown("---")
st.caption("Desarrollado con 🧠 y Streamlit • Datos por Yahoo Finance • ¿Te gustaría agregar modelos predictivos automáticos? ¡Pídelo!")

