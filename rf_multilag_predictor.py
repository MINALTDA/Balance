# rf_multilag_predictor.py
import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

st.set_page_config(page_title="Random Forest Multilag Predictor", layout="wide")
st.title("🌳 Random Forest Multivariado con Múltiples Lags para Predicción de Retornos")

# 1. Configuración de tickers y parámetros
st.sidebar.header("Configuración")
default_tickers = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "META", "NVDA"]
tickers = st.sidebar.multiselect("Selecciona los tickers a usar como predictores", default_tickers, default=default_tickers)
target = st.sidebar.selectbox("Selecciona el ticker objetivo (a predecir)", tickers, index=3)  # Ej: AMZN
max_lag = st.sidebar.slider("Máximo lag por ticker", 1, 20, 10)
period_years = st.sidebar.slider("Años de histórico", 1, 10, 3)
n_estimators = st.sidebar.slider("N° de árboles en el Random Forest", 10, 200, 100, step=10)

st.write(f"**Objetivo:** predecir {target} usando {', '.join(tickers)} y hasta {max_lag} lags por acción.")

# 2. Descarga y preparación de datos
@st.cache_data(show_spinner=False)
def load_data(tickers, years):
    end = datetime.today()
    start = end - timedelta(days=365*years)
    df = yf.download(tickers, start=start, end=end, progress=False)["Close"]
    if isinstance(df, pd.Series):
        df = df.to_frame()
    return df

df = load_data(tickers, period_years)
returns = np.log(df / df.shift(1)).dropna()

# 3. Construcción de features: lags múltiples
features = []
for ticker in tickers:
    for lag in range(1, max_lag+1):
        col = f"{ticker}_lag{lag}"
        returns[col] = returns[ticker].shift(lag)
        features.append(col)

model_data = returns[[target] + features].dropna()
X = model_data[features].values
y = model_data[target].values

# 4. Ajuste del modelo Random Forest
rf = RandomForestRegressor(n_estimators=n_estimators, max_depth=7, random_state=42, n_jobs=-1)
rf.fit(X, y)
y_pred = rf.predict(X)

# 5. Métricas y visualización
st.subheader("Métricas del modelo Random Forest multivariado")
mse = mean_squared_error(y, y_pred)
r2 = r2_score(y, y_pred)
st.write(f"**R²:** {r2:.4f}")
st.write(f"**MSE:** {mse:.6e}")

st.subheader("Gráficos comparativos")
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
axes[0].plot(model_data.index, y, label="Real")
axes[0].plot(model_data.index, y_pred, label="Predicho", alpha=0.7)
axes[0].set_title(f"Retornos diarios de {target} (real vs. predicho)")
axes[0].legend()
axes[1].scatter(y, y_pred, alpha=0.3)
axes[1].set_xlabel("Retorno real")
axes[1].set_ylabel("Retorno predicho")
axes[1].set_title("Dispersión: Real vs. Predicho")
plt.tight_layout()
st.pyplot(fig)

# 6. Importancia de predictores
importancias = pd.Series(rf.feature_importances_, index=features)
st.subheader("Top 10 predictores más importantes según Random Forest")
st.dataframe(importancias.abs().sort_values(ascending=False).head(10))

st.caption("Si el desempeño sigue siendo bajo, prueba cambiar a predicción de dirección (clasificación) o a retornos semanales/mensuales.")

