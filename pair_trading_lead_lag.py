import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from itertools import combinations

st.set_page_config(page_title="Pair Trading con Desfase (Lead-Lag)", layout="wide")
st.title("🔁 Pair Trading con Desfase Óptimo entre Acciones (Lead-Lag)")

# ---- CONFIGURACIÓN ----
st.sidebar.header("Parámetros del backtest")
top_tickers = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "META", "NVDA"]
commission = st.sidebar.number_input("Comisión ida+vuelta (ej: 0.001 = 0.1%)", value=0.001, step=0.0001, format="%.4f")
lookback = st.sidebar.slider("Ventana rolling (días)", 20, 120, 60)
z_entry = st.sidebar.slider("Z-score entrada", 1.0, 3.0, 2.0)
z_exit = st.sidebar.slider("Z-score salida", 0.0, 2.0, 0.5)
max_lag = st.sidebar.slider("Desfase máximo a buscar (días)", 0, 10, 5)

@st.cache_data(show_spinner=False)
def load_prices(tickers):
    df = yf.download(tickers, start="2018-01-01", progress=False)["Close"]
    if isinstance(df, pd.Series):
        df = df.to_frame()
    return df

prices = load_prices(top_tickers).dropna()

def pair_trading_lag(prices, t1, t2, lookback, z_entry, z_exit, commission, max_lag):
    best_result = None
    # Probar todos los lags de -max_lag a +max_lag (excluyendo lag=0 para ambos sentidos duplicados)
    for lag in range(-max_lag, max_lag+1):
        if lag == 0:
            continue  # sin desfase ya se prueba por default en otra pasada
        # Alinemos t2: si lag>0, t2 va retrasada respecto a t1
        if lag > 0:
            t2_aligned = prices[t2].shift(lag)
            df = pd.DataFrame({t1: prices[t1], t2: t2_aligned}).dropna()
        else:
            t1_aligned = prices[t1].shift(-lag)
            df = pd.DataFrame({t1: t1_aligned, t2: prices[t2]}).dropna()
        if len(df) < lookback + 10:
            continue  # muy poca data
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
        df["Spread_ret"] = np.log(df[t1] / df[t1].shift(1)) - np.log(df[t2] / df[t2].shift(1))
        df["Strategy_ret"] = df["Position"].shift(1) * df["Spread_ret"]
        df["Trade"] = df["Position"].diff().abs()
        df["Strategy_ret"] -= commission * df["Trade"]
        df["Cumulative"] = df["Strategy_ret"].cumsum().apply(np.exp)
        sharpe = np.nan
        if df["Strategy_ret"].std() > 0:
            sharpe = np.mean(df["Strategy_ret"]) / np.std(df["Strategy_ret"]) * np.sqrt(252)
        n_trades = int(df["Trade"].sum())
        growth = df["Cumulative"].iloc[-1]
        max_dd = ((df["Cumulative"].cummax() - df["Cumulative"]) / df["Cumulative"].cummax()).max()
        result = {
            "pair": f"{t1}-{t2}",
            "lag": lag,
            "sharpe": sharpe,
            "growth": growth,
            "drawdown": max_dd,
            "n_trades": n_trades,
            "df": df
        }
        # Guardar solo el mejor lag para cada par
        if (best_result is None) or (sharpe is not np.nan and (sharpe > best_result["sharpe"])):
            best_result = result
    return best_result

# ---- RUN RANKING ----
results = []
for t1, t2 in combinations(top_tickers, 2):
    res0 = pair_trading_lag(prices, t1, t2, lookback, z_entry, z_exit, commission, 0)  # lag=0 clásico
    if res0 is not None:
        results.append({**res0, "lag": 0})
    res_lag = pair_trading_lag(prices, t1, t2, lookback, z_entry, z_exit, commission, max_lag)
    if res_lag is not None:
        results.append(res_lag)

results_df = pd.DataFrame(results)
results_df = results_df.sort_values(by="sharpe", ascending=False)

st.subheader("Top 5 pares y desfases por Sharpe Ratio (con comisiones)")
st.dataframe(results_df[["pair", "lag", "sharpe", "growth", "drawdown", "n_trades"]].head(5))

selected_row = st.selectbox(
    "Selecciona un par y desfase para ver detalles",
    results_df["pair"] + " (lag=" + results_df["lag"].astype(str) + ")"
)
row = results_df.loc[
    results_df["pair"] + " (lag=" + results_df["lag"].astype(str) + ")" == selected_row
].iloc[0]
pair_df = row["df"]

st.subheader(f"Curva de estrategia para {row['pair']} (lag óptimo = {row['lag']})")
fig, ax = plt.subplots(figsize=(10, 4))
ax.plot(pair_df.index, pair_df["Cumulative"], label="Estrategia Pair Trading (con desfase)")
ax.set_ylabel("Crecimiento acumulado ($ inicial = 1)")
ax.set_title(f"Crecimiento de la estrategia ({row['pair']}, lag={row['lag']}, comisiones incluidas)")
ax.legend()
st.pyplot(fig)

st.subheader("Métricas detalladas")
st.write(f"Sharpe Ratio (anualizado): {row['sharpe']:.2f}")
st.write(f"Crecimiento total: {row['growth']:.2f}x")
st.write(f"Max drawdown: {row['drawdown']:.2%}")
st.write(f"Número de señales/operaciones: {row['n_trades']}")
