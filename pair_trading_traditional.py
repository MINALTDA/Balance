import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from itertools import combinations

st.set_page_config(page_title="Ranking de Pair Trading", layout="wide")
st.title("🏆 Ranking automático de Pares para Pair Trading")

# ---- CONFIGURACIÓN ----
st.sidebar.header("Parámetros del backtest")
top_tickers = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "META", "NVDA"]
commission = st.sidebar.number_input("Comisión ida+vuelta (ej: 0.001 = 0.1%)", value=0.001, step=0.0001, format="%.4f")
lookback = st.sidebar.slider("Ventana rolling (días)", 20, 120, 60)
z_entry = st.sidebar.slider("Z-score entrada", 1.0, 3.0, 2.0)
z_exit = st.sidebar.slider("Z-score salida", 0.0, 2.0, 0.5)

# ---- DESCARGA DATOS ----
@st.cache_data(show_spinner=False)
def load_prices(tickers):
    df = yf.download(tickers, start="2018-01-01", progress=False)["Close"]
    if isinstance(df, pd.Series):
        df = df.to_frame()
    return df

prices = load_prices(top_tickers).dropna()

# ---- FUNCIÓN PAIR TRADING ----
def pair_trading(prices, t1, t2, lookback, z_entry, z_exit, commission):
    df = pd.DataFrame({
        t1: prices[t1],
        t2: prices[t2],
    }).dropna()
    log_spread = np.log(df[t1] / df[t2])
    spread_mean = log_spread.rolling(window=lookback).mean()
    spread_std = log_spread.rolling(window=lookback).std()
    zscore = (log_spread - spread_mean) / spread_std
    df["Zscore"] = zscore
    df["Signal"] = 0
    df.loc[df["Zscore"] < -z_entry, "Signal"] = 1
    df.loc[df["Zscore"] > z_entry, "Signal"] = -1
    df.loc[(df["Zscore"].abs() < z_exit), "Signal"] = 0
    df["Position"] = df["Signal"].replace(to_replace=0, method="ffill").fillna(0)
    # Spread return (long t1, short t2)
    df["Spread_ret"] = np.log(df[t1] / df[t1].shift(1)) - np.log(df[t2] / df[t2].shift(1))
    df["Strategy_ret"] = df["Position"].shift(1) * df["Spread_ret"]
    # Restar comisiones cada vez que la posición cambia
    df["Trade"] = df["Position"].diff().abs()
    df["Strategy_ret"] -= commission * df["Trade"]
    df["Cumulative"] = df["Strategy_ret"].cumsum().apply(np.exp)
    # Métricas
    sharpe = np.nan
    if df["Strategy_ret"].std() > 0:
        sharpe = np.mean(df["Strategy_ret"]) / np.std(df["Strategy_ret"]) * np.sqrt(252)
    n_trades = int(df["Trade"].sum())
    growth = df["Cumulative"].iloc[-1]
    max_dd = ((df["Cumulative"].cummax() - df["Cumulative"]) / df["Cumulative"].cummax()).max()
    return {
        "pair": f"{t1}-{t2}",
        "sharpe": sharpe,
        "growth": growth,
        "drawdown": max_dd,
        "n_trades": n_trades,
        "df": df
    }

# ---- RUN RANKING ----
results = []
for t1, t2 in combinations(top_tickers, 2):
    res = pair_trading(prices, t1, t2, lookback, z_entry, z_exit, commission)
    results.append(res)

results_df = pd.DataFrame(results)
results_df = results_df.sort_values(by="sharpe", ascending=False)

st.subheader("Top 5 pares por Sharpe Ratio (con comisiones)")
st.dataframe(results_df[["pair", "sharpe", "growth", "drawdown", "n_trades"]].head(5))

# Permite elegir un par para análisis
selected_pair = st.selectbox("Selecciona un par para ver detalles", results_df["pair"].head(10))
pair_df = results_df.set_index("pair").loc[selected_pair, "df"]

st.subheader(f"Curva de estrategia para {selected_pair}")
fig, ax = plt.subplots(figsize=(10, 4))
ax.plot(pair_df.index, pair_df["Cumulative"], label="Estrategia Pair Trading")
ax.set_ylabel("Crecimiento acumulado ($ inicial = 1)")
ax.set_title(f"Crecimiento de la estrategia ({selected_pair}, con comisiones)")
ax.legend()
st.pyplot(fig)

st.subheader("Métricas detalladas")
st.write(f"Sharpe Ratio (anualizado): {results_df.set_index('pair').loc[selected_pair, 'sharpe']:.2f}")
st.write(f"Crecimiento total: {results_df.set_index('pair').loc[selected_pair, 'growth']:.2f}x")
st.write(f"Max drawdown: {results_df.set_index('pair').loc[selected_pair, 'drawdown']:.2%}")
st.write(f"Número de señales/operaciones: {results_df.set_index('pair').loc[selected_pair, 'n_trades']}")
