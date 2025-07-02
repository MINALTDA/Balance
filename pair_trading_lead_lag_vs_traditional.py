# pair_trading_lead_lag_vs_traditional.py

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from itertools import combinations

st.set_page_config(page_title="Pair Trading: Lead-Lag vs Tradicional", layout="wide")
st.title("🔁 Pair Trading: Alineamiento Temporal (Lead-Lag) vs Tradicional")

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

def pair_trading(prices, t1, t2, lookback, z_entry, z_exit, commission, lag=0):
    # lag > 0: t2 va retrasada respecto a t1
    if lag > 0:
        t2_aligned = prices[t2].shift(lag)
        df = pd.DataFrame({t1: prices[t1], t2: t2_aligned}).dropna()
    elif lag < 0:
        t1_aligned = prices[t1].shift(-lag)
        df = pd.DataFrame({t1: t1_aligned, t2: prices[t2]}).dropna()
    else:
        df = pd.DataFrame({t1: prices[t1], t2: prices[t2]}).dropna()
    if len(df) < lookback + 10:
        return None
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
    return {
        "sharpe": sharpe,
        "growth": growth,
        "drawdown": max_dd,
        "n_trades": n_trades,
        "df": df
    }

# ---- BUSCA EL MEJOR LAG PARA CADA PAR ----
results = []
for t1, t2 in combinations(top_tickers, 2):
    best_sharpe = -np.inf
    best_lag = 0
    best_result = None
    for lag in range(-max_lag, max_lag+1):
        if lag == 0:  # skip, lo probamos después
            continue
        res = pair_trading(prices, t1, t2, lookback, z_entry, z_exit, commission, lag)
        if res and res["sharpe"] > best_sharpe:
            best_sharpe = res["sharpe"]
            best_lag = lag
            best_result = res
    # Calcula también el resultado tradicional (lag=0)
    res_trad = pair_trading(prices, t1, t2, lookback, z_entry, z_exit, commission, lag=0)
    if best_result and res_trad:
        results.append({
            "pair": f"{t1}-{t2}",
            "lag": best_lag,
            "sharpe_lag": best_result["sharpe"],
            "growth_lag": best_result["growth"],
            "drawdown_lag": best_result["drawdown"],
            "n_trades_lag": best_result["n_trades"],
            "df_lag": best_result["df"],
            "sharpe_trad": res_trad["sharpe"],
            "growth_trad": res_trad["growth"],
            "drawdown_trad": res_trad["drawdown"],
            "n_trades_trad": res_trad["n_trades"],
            "df_trad": res_trad["df"],
        })

results_df = pd.DataFrame(results)
results_df = results_df.sort_values(by="sharpe_lag", ascending=False)

st.subheader("Top 5 pares y desfases por Sharpe Ratio (con comisiones)")
st.dataframe(results_df[["pair", "lag", "sharpe_lag", "growth_lag", "sharpe_trad", "growth_trad"]].head(5))

selected_row = st.selectbox(
    "Selecciona un par para comparar estrategias",
    results_df["pair"] + " (lag=" + results_df["lag"].astype(str) + ")"
)
row = results_df.loc[
    results_df["pair"] + " (lag=" + results_df["lag"].astype(str) + ")" == selected_row
].iloc[0]

df_lag = row["df_lag"]
df_trad = row["df_trad"]

# --- GRAFICA COMPARATIVA ---
st.subheader(f"Comparación de estrategias para {row['pair']}")
fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(df_lag.index, df_lag["Cumulative"], label=f"Lead-Lag (lag={row['lag']})")
ax.plot(df_trad.index, df_trad["Cumulative"], label="Tradicional (lag=0)", linestyle='--')
ax.set_ylabel("Crecimiento acumulado ($ inicial = 1)")
ax.set_title(f"Crecimiento acumulado: {row['pair']}")
ax.legend()
st.pyplot(fig)

st.subheader("Métricas detalladas")
st.write(f"**Lead-Lag (lag={row['lag']}):** Sharpe={row['sharpe_lag']:.2f} | Growth={row['growth_lag']:.2f}x | Drawdown={row['drawdown_lag']:.2%} | Operaciones={row['n_trades_lag']}")
st.write(f"**Tradicional (lag=0):** Sharpe={row['sharpe_trad']:.2f} | Growth={row['growth_trad']:.2f}x | Drawdown={row['drawdown_trad']:.2%} | Operaciones={row['n_trades_trad']}")

st.caption("¡Así puedes comparar, para cada par, si el alineamiento (lead-lag) realmente aporta valor respecto al enfoque tradicional!")
