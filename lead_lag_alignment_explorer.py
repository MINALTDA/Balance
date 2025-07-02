# # lead_lag_alignment_explorer.py

# import streamlit as st
# import yfinance as yf
# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt

# st.set_page_config(page_title="Explorador de Alineamiento Lead-Lag", layout="wide")
# st.title("🔍 Explorador Visual de Alineamiento Lead-Lag entre Acciones")

# st.sidebar.header("Parámetros")
# tickers = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "META", "NVDA"]
# t1 = st.sidebar.selectbox("Activo líder (A)", tickers, 0)
# t2 = st.sidebar.selectbox("Activo seguidor (B)", tickers, 1)
# lag = st.sidebar.slider("Lag aplicado a B (en días)", -10, 10, 0)
# lookback = st.sidebar.slider("Rolling window (días)", 10, 120, 60)

# st.write(f"**Visualiza cómo el spread y el z-score cambian al aplicar un desfase (lag) de {lag} días a {t2} respecto a {t1}.**")

# @st.cache_data(show_spinner=False)
# def load_prices(tickers):
#     df = yf.download(tickers, start="2018-01-01", progress=False)["Close"]
#     if isinstance(df, pd.Series):
#         df = df.to_frame()
#     return df

# prices = load_prices([t1, t2]).dropna()
# if lag > 0:
#     b_aligned = prices[t2].shift(lag)
#     label_lag = f"{t2} (retrasado {lag} días)"
# elif lag < 0:
#     a_aligned = prices[t1].shift(-lag)
#     label_lag = f"{t1} (retrasado {abs(lag)} días)"
# else:
#     b_aligned = prices[t2]
#     label_lag = f"{t2} (sin lag)"

# spread = np.log(prices[t1]) - np.log(b_aligned if lag>=0 else prices[t2])
# spread = spread.dropna()
# spread_mean = spread.rolling(window=lookback).mean()
# spread_std = spread.rolling(window=lookback).std()
# zscore = (spread - spread_mean) / spread_std

# st.subheader("Gráfico 1: Series de precios (con lag aplicado)")
# fig1, ax1 = plt.subplots(figsize=(10, 3))
# ax1.plot(prices.index, prices[t1], label=t1)
# if lag >= 0:
#     ax1.plot(prices.index, b_aligned, label=label_lag)
# else:
#     ax1.plot(prices.index, prices[t2], label=t2)
#     ax1.plot(prices.index, a_aligned, label=label_lag)
# ax1.set_ylabel("Precio")
# ax1.legend()
# st.pyplot(fig1)

# st.subheader("Gráfico 2: Spread logarítmico (A - B_lag)")
# fig2, ax2 = plt.subplots(figsize=(10, 3))
# ax2.plot(spread.index, spread, label=f"Spread log({t1}) - log({label_lag})")
# ax2.plot(spread_mean.index, spread_mean, label="Media rolling", linestyle='--')
# ax2.set_ylabel("Spread")
# ax2.legend()
# st.pyplot(fig2)

# st.subheader("Gráfico 3: Z-score del spread")
# fig3, ax3 = plt.subplots(figsize=(10, 3))
# ax3.plot(zscore.index, zscore, label="Z-score")
# ax3.axhline(0, color="gray", linestyle="--")
# ax3.set_ylabel("Z-score")
# ax3.legend()
# st.pyplot(fig3)

# st.caption(
#     "Explora diferentes lags y observa si el spread parece más 'revertido a la media' o presenta patrones que ayuden a definir una mejor estrategia lead-lag."
# )








# # lead_lag_alignment_explorer.py (Plotly version)

# import streamlit as st
# import yfinance as yf
# import pandas as pd
# import numpy as np
# import plotly.graph_objs as go

# st.set_page_config(page_title="Explorador Lead-Lag Interactivo", layout="wide")
# st.title("🔍 Explorador Visual de Alineamiento Lead-Lag entre Acciones (Plotly)")

# st.sidebar.header("Parámetros")
# tickers = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "META", "NVDA"]
# t1 = st.sidebar.selectbox("Activo líder (A)", tickers, 0)
# t2 = st.sidebar.selectbox("Activo seguidor (B)", tickers, 1)
# lag = st.sidebar.slider("Lag aplicado a B (en días)", -10, 10, 0)
# lookback = st.sidebar.slider("Rolling window (días)", 10, 120, 60)

# st.write(f"**Visualiza cómo el spread y el z-score cambian al aplicar un desfase (lag) de {lag} días a {t2} respecto a {t1}.**")

# @st.cache_data(show_spinner=False)
# def load_prices(tickers):
#     df = yf.download(tickers, start="2018-01-01", progress=False)["Close"]
#     if isinstance(df, pd.Series):
#         df = df.to_frame()
#     return df

# prices = load_prices([t1, t2]).dropna()
# if lag > 0:
#     b_aligned = prices[t2].shift(lag)
#     label_lag = f"{t2} (retrasado {lag} días)"
# elif lag < 0:
#     a_aligned = prices[t1].shift(-lag)
#     label_lag = f"{t1} (retrasado {abs(lag)} días)"
# else:
#     b_aligned = prices[t2]
#     label_lag = f"{t2} (sin lag)"

# spread = np.log(prices[t1]) - np.log(b_aligned if lag>=0 else prices[t2])
# spread = spread.dropna()
# spread_mean = spread.rolling(window=lookback).mean()
# spread_std = spread.rolling(window=lookback).std()
# zscore = (spread - spread_mean) / spread_std

# # -- Plotly interactive charts --
# st.subheader("Gráfico 1: Series de precios (con lag aplicado)")
# fig1 = go.Figure()
# fig1.add_trace(go.Scatter(x=prices.index, y=prices[t1], name=t1))
# if lag >= 0:
#     fig1.add_trace(go.Scatter(x=prices.index, y=b_aligned, name=label_lag))
# else:
#     fig1.add_trace(go.Scatter(x=prices.index, y=prices[t2], name=t2))
#     fig1.add_trace(go.Scatter(x=prices.index, y=a_aligned, name=label_lag))
# fig1.update_layout(height=350, yaxis_title="Precio")
# st.plotly_chart(fig1, use_container_width=True)

# st.subheader("Gráfico 2: Spread logarítmico (A - B_lag)")
# fig2 = go.Figure()
# fig2.add_trace(go.Scatter(x=spread.index, y=spread, name=f"Spread log({t1}) - log({label_lag})"))
# fig2.add_trace(go.Scatter(x=spread_mean.index, y=spread_mean, name="Media rolling", line=dict(dash='dash')))
# fig2.update_layout(height=350, yaxis_title="Spread")
# st.plotly_chart(fig2, use_container_width=True)

# st.subheader("Gráfico 3: Z-score del spread")
# fig3 = go.Figure()
# fig3.add_trace(go.Scatter(x=zscore.index, y=zscore, name="Z-score"))
# fig3.add_trace(go.Scatter(x=zscore.index, y=np.zeros_like(zscore), name="Media", line=dict(dash='dash')))
# fig3.update_layout(height=350, yaxis_title="Z-score")
# st.plotly_chart(fig3, use_container_width=True)

# st.caption(
#     "Explora diferentes lags y observa si el spread parece más revertido a la media o presenta patrones más claros. "
#     "Si existe relación lead-lag, notarás que los cruces por la media y los extremos del z-score se ven más claros en cierto lag."
# )









# lead_lag_alignment_comparator.py

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objs as go

st.set_page_config(page_title="Comparador Lead-Lag Óptimo", layout="wide")
st.title("🔎 Comparador Visual de Alineamiento Óptimo Lead-Lag")

st.sidebar.header("Parámetros")
tickers = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "META", "NVDA"]
t1 = st.sidebar.selectbox("Activo líder (A)", tickers, 0)
t2 = st.sidebar.selectbox("Activo seguidor (B)", tickers, 1)
max_lag = st.sidebar.slider("Lag máximo a explorar (días)", 1, 15, 7)
lookback = st.sidebar.slider("Rolling window (días)", 10, 120, 60)
metric = st.sidebar.selectbox("Criterio para lag óptimo", ["Menor autocorrelación abs. (reversión)", "Máx. Sharpe de spread"], 0)

st.write(
    f"""
    Compara el spread y z-score con **lag=0** y con **lag óptimo** detectado automáticamente
    ({'mínima autocorrelación' if metric.startswith('Menor') else 'máx. Sharpe'}).
    - Cambia los parámetros y observa la diferencia.
    """
)

@st.cache_data(show_spinner=False)
def load_prices(tickers):
    df = yf.download(tickers, start="2018-01-01", progress=False)["Close"]
    if isinstance(df, pd.Series):
        df = df.to_frame()
    return df

prices = load_prices([t1, t2]).dropna()

def calc_spread(prices, lag):
    # lag>0: retrasar B, lag<0: retrasar A
    if lag > 0:
        b_aligned = prices[t2].shift(lag)
        spread = np.log(prices[t1]) - np.log(b_aligned)
    elif lag < 0:
        a_aligned = prices[t1].shift(-lag)
        spread = np.log(a_aligned) - np.log(prices[t2])
    else:
        spread = np.log(prices[t1]) - np.log(prices[t2])
    return spread.dropna()

def sharpe_of_spread(spread):
    ret = spread.diff().dropna()
    if ret.std() == 0 or np.isnan(ret.std()):
        return -np.inf
    return np.mean(ret) / np.std(ret) * np.sqrt(252)

def min_autocorr_lag(spread):
    # Retorno del spread (dif1) debe ser menos autocorrelado posible (reversión a la media)
    ret = spread.diff().dropna()
    return abs(pd.Series(ret).autocorr(lag=1))

# --- BUSCA EL LAG ÓPTIMO ---
lags = list(range(-max_lag, max_lag + 1))
lags.remove(0)  # ya se compara por defecto
metrics = []

for lag in lags:
    s = calc_spread(prices, lag)
    if len(s) < lookback + 5:
        metrics.append(np.nan)
        continue
    if metric.startswith("Máx. Sharpe"):
        val = sharpe_of_spread(s)
    else:
        val = -min_autocorr_lag(s)  # Queremos minimizar autocorrelación abs.
    metrics.append(val)

if metric.startswith("Máx. Sharpe"):
    best_idx = np.nanargmax(metrics)
else:
    best_idx = np.nanargmax(metrics)  # -autocorr, así máximo es mejor
lag_opt = lags[best_idx]

# -- Calcula spreads y z-score para lag=0 y lag óptimo --
spread0 = calc_spread(prices, 0)
spread_opt = calc_spread(prices, lag_opt)

# Alinea longitudes
common_idx = spread0.index.intersection(spread_opt.index)
spread0 = spread0.loc[common_idx]
spread_opt = spread_opt.loc[common_idx]

spread_mean0 = spread0.rolling(window=lookback).mean()
spread_std0 = spread0.rolling(window=lookback).std()
zscore0 = (spread0 - spread_mean0) / spread_std0

spread_mean_opt = spread_opt.rolling(window=lookback).mean()
spread_std_opt = spread_opt.rolling(window=lookback).std()
zscore_opt = (spread_opt - spread_mean_opt) / spread_std_opt

# -- Plotly comparativo --
st.subheader(f"Comparativo de Spread logarítmico (lag=0 vs lag óptimo {lag_opt:+d})")
fig2 = go.Figure()
fig2.add_trace(go.Scatter(x=spread0.index, y=spread0, name=f"Spread lag=0"))
fig2.add_trace(go.Scatter(x=spread_opt.index, y=spread_opt, name=f"Spread lag óptimo ({lag_opt:+d})"))
fig2.add_trace(go.Scatter(x=spread_mean0.index, y=spread_mean0, name="Media rolling lag=0", line=dict(dash='dash')))
fig2.add_trace(go.Scatter(x=spread_mean_opt.index, y=spread_mean_opt, name="Media rolling lag óptimo", line=dict(dash='dash')))
fig2.update_layout(height=350, yaxis_title="Spread")
st.plotly_chart(fig2, use_container_width=True)

st.subheader(f"Comparativo de Z-score (lag=0 vs lag óptimo {lag_opt:+d})")
fig3 = go.Figure()
fig3.add_trace(go.Scatter(x=zscore0.index, y=zscore0, name=f"Z-score lag=0"))
fig3.add_trace(go.Scatter(x=zscore_opt.index, y=zscore_opt, name=f"Z-score lag óptimo ({lag_opt:+d})"))
fig3.add_trace(go.Scatter(x=zscore0.index, y=np.zeros_like(zscore0), name="Media", line=dict(dash='dash')))
fig3.update_layout(height=350, yaxis_title="Z-score")
st.plotly_chart(fig3, use_container_width=True)

# -- Información extra --
st.info(f"""
**Lag óptimo detectado:** {lag_opt:+d} días  
**Criterio:** {'Sharpe máximo' if metric.startswith('Máx') else 'Menor autocorrelación absoluta (reversión)'}
""")

st.caption(
    "Observa si con el lag óptimo el spread y/o su z-score muestran reversión a la media más clara o cruces por la media más simétricos. "
    "Esto respalda visualmente la selección del lag óptimo usado en los modelos de previsión y trading."
)
