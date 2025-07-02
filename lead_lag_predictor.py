# lead_lag_predictor.py
import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

st.set_page_config(page_title="Lead-Lag Predictivo", layout="wide")
st.title("🔮 Predicción Automática con Relaciones Lead-Lag entre Acciones")

# 1. Configuración de la relación Lead-Lag
st.sidebar.header("Configuración de la predicción")

# Ejemplo de relaciones sugeridas
ejemplos = [
    # (target, predictor, lag)
    ("AMZN", "MSFT", 10),
    ("NVDA", "MSFT", 5),
    ("META", "AMZN", 10),
    ("AMZN", "GOOGL", 1),
    ("TSLA", "AAPL", 8),
]

st.sidebar.markdown("#### Ejemplos de relaciones sugeridas")
ejemplo_idx = st.sidebar.selectbox(
    "Selecciona un ejemplo sugerido o personaliza abajo:", 
    list(range(len(ejemplos))), 
    format_func=lambda i: f"{ejemplos[i][0]} ← {ejemplos[i][1]} (lag={ejemplos[i][2]} días)"
)

default_target, default_predictor, default_lag = ejemplos[ejemplo_idx]

# Permite modificar o crear otras combinaciones
target = st.sidebar.text_input("Ticker objetivo (a predecir)", value=default_target)
predictor = st.sidebar.text_input("Ticker predictor (lead)", value=default_predictor)
lag_days = st.sidebar.number_input("Desfase (lag) en días", min_value=1, max_value=30, value=int(default_lag))
period_years = st.sidebar.slider("Años de histórico a descargar", 1, 10, 3)

st.write(f"**Vas a predecir:** {target} usando {predictor} desplazado {lag_days} días")

# 2. Descarga y preparación de datos
@st.cache_data(show_spinner=False)
def load_data(ticker, years):
    end = datetime.today()
    start = end - timedelta(days=365*years)
    df = yf.download(ticker, start=start, end=end, progress=False)[["Close"]].rename(columns={"Close": ticker})
    return df

df_target = load_data(target, period_years)
df_predictor = load_data(predictor, period_years)

# Unimos y calculamos retornos logarítmicos
data = df_target.join(df_predictor, how="inner")
returns = np.log(data / data.shift(1)).dropna()
returns = returns[[target, predictor]]

# Aplicamos el desfase (lag)
returns[f"{predictor}_lagged"] = returns[predictor].shift(lag_days)
model_data = returns[[target, f"{predictor}_lagged"]].dropna()

st.write("Vista previa de los retornos logarítmicos con lag aplicado:")
st.dataframe(model_data.head())

# 3. Ajuste del modelo predictivo
X = model_data[[f"{predictor}_lagged"]].values
y = model_data[target].values

model = LinearRegression()
model.fit(X, y)
y_pred = model.predict(X)

# 4. Métricas del modelo
st.subheader("Métricas del modelo predictivo")
mse = mean_squared_error(y, y_pred)
r2 = r2_score(y, y_pred)
st.write(f"**R²:** {r2:.4f}")
st.write(f"**MSE:** {mse:.6e}")

# 5. Visualización del ajuste
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

# 6. Predicción futura (opcional, usando el último valor disponible)
st.subheader("Predicción para próximos días (experimental)")
with st.expander("Ver predicción para los próximos días"):
    n_pred_days = st.slider("¿Cuántos días predecir?", 1, 5, 3)
    ultimos_rets = returns[[predictor]].iloc[-(lag_days+n_pred_days):][predictor].values
    pred_fut = []
    for i in range(n_pred_days):
        if len(ultimos_rets) < lag_days + i + 1:
            pred_fut.append(np.nan)
            continue
        x_pred = ultimos_rets[i:i+lag_days][-1].reshape(1, -1)
        pred_fut.append(model.predict(x_pred)[0])
    pred_fut = np.array(pred_fut)

    future_dates = pd.date_range(model_data.index[-1]+pd.Timedelta(days=1), periods=n_pred_days, freq="B")
    pred_df = pd.DataFrame({"Fecha": future_dates, f"Predicho {target} retorno": pred_fut})
    st.write(pred_df)
    st.line_chart(pred_df.set_index("Fecha"))

st.caption("Puedes ajustar los tickers y el lag en la barra lateral para replicar cualquier relación sugerida de la exploración.")

