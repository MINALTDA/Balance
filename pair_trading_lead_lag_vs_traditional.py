# # pair_trading_lead_lag_vs_traditional.py

# import streamlit as st
# import yfinance as yf
# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# from itertools import combinations

# st.set_page_config(page_title="Pair Trading: Lead-Lag vs Tradicional", layout="wide")
# st.title("🔁 Pair Trading: Alineamiento Temporal (Lead-Lag) vs Tradicional")

# # ---- CONFIGURACIÓN ----
# st.sidebar.header("Parámetros del backtest")
# top_tickers = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "META", "NVDA"]
# commission = st.sidebar.number_input("Comisión ida+vuelta (ej: 0.001 = 0.1%)", value=0.001, step=0.0001, format="%.4f")
# lookback = st.sidebar.slider("Ventana rolling (días)", 20, 120, 60)
# z_entry = st.sidebar.slider("Z-score entrada", 1.0, 3.0, 2.0)
# z_exit = st.sidebar.slider("Z-score salida", 0.0, 2.0, 0.5)
# max_lag = st.sidebar.slider("Desfase máximo a buscar (días)", 0, 15, 5)

# @st.cache_data(show_spinner=False)
# def load_prices(tickers):
#     df = yf.download(tickers, start="2018-01-01", progress=False)["Close"]
#     if isinstance(df, pd.Series):
#         df = df.to_frame()
#     return df

# prices = load_prices(top_tickers).dropna()

# def pair_trading(prices, t1, t2, lookback, z_entry, z_exit, commission, lag=0):
#     # lag > 0: t2 va retrasada respecto a t1
#     if lag > 0:
#         t2_aligned = prices[t2].shift(lag)
#         df = pd.DataFrame({t1: prices[t1], t2: t2_aligned}).dropna()
#     elif lag < 0:
#         t1_aligned = prices[t1].shift(-lag)
#         df = pd.DataFrame({t1: t1_aligned, t2: prices[t2]}).dropna()
#     else:
#         df = pd.DataFrame({t1: prices[t1], t2: prices[t2]}).dropna()
#     if len(df) < lookback + 10:
#         return None
#     log_spread = np.log(df[t1] / df[t2])
#     spread_mean = log_spread.rolling(window=lookback).mean()
#     spread_std = log_spread.rolling(window=lookback).std()
#     zscore = (log_spread - spread_mean) / spread_std
#     df["Zscore"] = zscore
#     df["Signal"] = 0
#     df.loc[df["Zscore"] < -z_entry, "Signal"] = 1
#     df.loc[df["Zscore"] > z_entry, "Signal"] = -1
#     df.loc[(df["Zscore"].abs() < z_exit), "Signal"] = 0
#     df["Position"] = df["Signal"].replace(to_replace=0, method="ffill").fillna(0)
#     df["Spread_ret"] = np.log(df[t1] / df[t1].shift(1)) - np.log(df[t2] / df[t2].shift(1))
#     df["Strategy_ret"] = df["Position"].shift(1) * df["Spread_ret"]
#     df["Trade"] = df["Position"].diff().abs()
#     df["Strategy_ret"] -= commission * df["Trade"]
#     df["Cumulative"] = df["Strategy_ret"].cumsum().apply(np.exp)
#     sharpe = np.nan
#     if df["Strategy_ret"].std() > 0:
#         sharpe = np.mean(df["Strategy_ret"]) / np.std(df["Strategy_ret"]) * np.sqrt(252)
#     n_trades = int(df["Trade"].sum())
#     growth = df["Cumulative"].iloc[-1]
#     max_dd = ((df["Cumulative"].cummax() - df["Cumulative"]) / df["Cumulative"].cummax()).max()
#     return {
#         "sharpe": sharpe,
#         "growth": growth,
#         "drawdown": max_dd,
#         "n_trades": n_trades,
#         "df": df
#     }

# # ---- BUSCA EL MEJOR LAG PARA CADA PAR ----
# results = []
# for t1, t2 in combinations(top_tickers, 2):
#     best_sharpe = -np.inf
#     best_lag = 0
#     best_result = None
#     for lag in range(-max_lag, max_lag+1):
#         if lag == 0:  # skip, lo probamos después
#             continue
#         res = pair_trading(prices, t1, t2, lookback, z_entry, z_exit, commission, lag)
#         if res and res["sharpe"] > best_sharpe:
#             best_sharpe = res["sharpe"]
#             best_lag = lag
#             best_result = res
#     # Calcula también el resultado tradicional (lag=0)
#     res_trad = pair_trading(prices, t1, t2, lookback, z_entry, z_exit, commission, lag=0)
#     if best_result and res_trad:
#         results.append({
#             "pair": f"{t1}-{t2}",
#             "lag": best_lag,
#             "sharpe_lag": best_result["sharpe"],
#             "growth_lag": best_result["growth"],
#             "drawdown_lag": best_result["drawdown"],
#             "n_trades_lag": best_result["n_trades"],
#             "df_lag": best_result["df"],
#             "sharpe_trad": res_trad["sharpe"],
#             "growth_trad": res_trad["growth"],
#             "drawdown_trad": res_trad["drawdown"],
#             "n_trades_trad": res_trad["n_trades"],
#             "df_trad": res_trad["df"],
#         })

# results_df = pd.DataFrame(results)
# results_df = results_df.sort_values(by="sharpe_lag", ascending=False)

# st.subheader("Top 5 pares y desfases por Sharpe Ratio (con comisiones)")
# st.dataframe(results_df[["pair", "lag", "sharpe_lag", "growth_lag", "sharpe_trad", "growth_trad"]].head(5))

# selected_row = st.selectbox(
#     "Selecciona un par para comparar estrategias",
#     results_df["pair"] + " (lag=" + results_df["lag"].astype(str) + ")"
# )
# row = results_df.loc[
#     results_df["pair"] + " (lag=" + results_df["lag"].astype(str) + ")" == selected_row
# ].iloc[0]

# df_lag = row["df_lag"]
# df_trad = row["df_trad"]

# # --- GRAFICA COMPARATIVA ---
# st.subheader(f"Comparación de estrategias para {row['pair']}")
# fig, ax = plt.subplots(figsize=(10, 5))
# ax.plot(df_lag.index, df_lag["Cumulative"], label=f"Lead-Lag (lag={row['lag']})")
# ax.plot(df_trad.index, df_trad["Cumulative"], label="Tradicional (lag=0)", linestyle='--')
# ax.set_ylabel("Crecimiento acumulado ($ inicial = 1)")
# ax.set_title(f"Crecimiento acumulado: {row['pair']}")
# ax.legend()
# st.pyplot(fig)

# st.subheader("Métricas detalladas")
# st.write(f"**Lead-Lag (lag={row['lag']}):** Sharpe={row['sharpe_lag']:.2f} | Growth={row['growth_lag']:.2f}x | Drawdown={row['drawdown_lag']:.2%} | Operaciones={row['n_trades_lag']}")
# st.write(f"**Tradicional (lag=0):** Sharpe={row['sharpe_trad']:.2f} | Growth={row['growth_trad']:.2f}x | Drawdown={row['drawdown_trad']:.2%} | Operaciones={row['n_trades_trad']}")

# st.caption("¡Así puedes comparar, para cada par, si el alineamiento (lead-lag) realmente aporta valor respecto al enfoque tradicional!")





# # pair_trading_lead_lag_vs_traditional.py

# import streamlit as st
# import yfinance as yf
# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# from itertools import combinations
# from datetime import timedelta

# st.set_page_config(page_title="Pair Trading: Lead-Lag vs Tradicional", layout="wide")
# st.title("🔁 Pair Trading: Alineamiento Temporal (Lead-Lag) vs Tradicional")

# st.sidebar.header("Parámetros del backtest")
# top_tickers = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "META", "NVDA"]
# commission = st.sidebar.number_input("Comisión ida+vuelta (ej: 0.001 = 0.1%)", value=0.001, step=0.0001, format="%.4f")
# lookback = st.sidebar.slider("Ventana rolling (días)", 20, 120, 60)
# z_entry = st.sidebar.slider("Z-score entrada", 1.0, 3.0, 2.0)
# z_exit = st.sidebar.slider("Z-score salida", 0.0, 2.0, 0.5)
# max_lag = st.sidebar.slider("Desfase máximo a buscar (días)", 0, 15, 5)

# @st.cache_data(show_spinner=False)
# def load_prices(tickers):
#     df = yf.download(tickers, start="2018-01-01", progress=False)["Close"]
#     if isinstance(df, pd.Series):
#         df = df.to_frame()
#     return df

# prices = load_prices(top_tickers).dropna()

# def pair_trading(prices, t1, t2, lookback, z_entry, z_exit, commission, lag=0):
#     if lag > 0:
#         t2_aligned = prices[t2].shift(lag)
#         df = pd.DataFrame({t1: prices[t1], t2: t2_aligned}).dropna()
#     elif lag < 0:
#         t1_aligned = prices[t1].shift(-lag)
#         df = pd.DataFrame({t1: t1_aligned, t2: prices[t2]}).dropna()
#     else:
#         df = pd.DataFrame({t1: prices[t1], t2: prices[t2]}).dropna()
#     if len(df) < lookback + 10:
#         return None
#     log_spread = np.log(df[t1] / df[t2])
#     spread_mean = log_spread.rolling(window=lookback).mean()
#     spread_std = log_spread.rolling(window=lookback).std()
#     zscore = (log_spread - spread_mean) / spread_std
#     df["Zscore"] = zscore
#     df["Signal"] = 0
#     df.loc[df["Zscore"] < -z_entry, "Signal"] = 1
#     df.loc[df["Zscore"] > z_entry, "Signal"] = -1
#     df.loc[(df["Zscore"].abs() < z_exit), "Signal"] = 0
#     df["Position"] = df["Signal"].replace(to_replace=0, method="ffill").fillna(0)
#     df["Spread_ret"] = np.log(df[t1] / df[t1].shift(1)) - np.log(df[t2] / df[t2].shift(1))
#     df["Strategy_ret"] = df["Position"].shift(1) * df["Spread_ret"]
#     df["Trade"] = df["Position"].diff().abs()
#     df["Strategy_ret"] -= commission * df["Trade"]
#     df["Cumulative"] = df["Strategy_ret"].cumsum().apply(np.exp)
#     sharpe = np.nan
#     if df["Strategy_ret"].std() > 0:
#         sharpe = np.mean(df["Strategy_ret"]) / np.std(df["Strategy_ret"]) * np.sqrt(252)
#     n_trades = int(df["Trade"].sum())
#     growth = df["Cumulative"].iloc[-1]
#     max_dd = ((df["Cumulative"].cummax() - df["Cumulative"]) / df["Cumulative"].cummax()).max()
#     return {
#         "sharpe": sharpe,
#         "growth": growth,
#         "drawdown": max_dd,
#         "n_trades": n_trades,
#         "df": df,
#         "lag": lag
#     }

# def lag_min_autocorr(prices, t1, t2, max_lag, lookback):
#     # Busca el lag con menor autocorrelación absoluta de retornos del spread
#     lags = list(range(-max_lag, max_lag + 1))
#     lags.remove(0)
#     best_metric = np.inf
#     best_lag = 0
#     for lag in lags:
#         if lag > 0:
#             b_aligned = prices[t2].shift(lag)
#             spread = np.log(prices[t1]) - np.log(b_aligned)
#         elif lag < 0:
#             a_aligned = prices[t1].shift(-lag)
#             spread = np.log(a_aligned) - np.log(prices[t2])
#         else:
#             spread = np.log(prices[t1]) - np.log(prices[t2])
#         ret = spread.diff().dropna()
#         if len(ret) < lookback + 5:
#             continue
#         metric = abs(pd.Series(ret).autocorr(lag=1))
#         if metric < best_metric:
#             best_metric = metric
#             best_lag = lag
#     return best_lag

# # ---- BUSCA EL MEJOR LAG PARA CADA PAR ----
# results = []
# for t1, t2 in combinations(top_tickers, 2):
#     best_sharpe = -np.inf
#     best_lag = 0
#     best_result = None
#     for lag in range(-max_lag, max_lag+1):
#         if lag == 0:
#             continue
#         res = pair_trading(prices, t1, t2, lookback, z_entry, z_exit, commission, lag)
#         if res and res["sharpe"] > best_sharpe:
#             best_sharpe = res["sharpe"]
#             best_lag = lag
#             best_result = res
#     # Tradicional
#     res_trad = pair_trading(prices, t1, t2, lookback, z_entry, z_exit, commission, lag=0)
#     # Lag exploratorio
#     lag_exp = lag_min_autocorr(prices, t1, t2, max_lag, lookback)
#     res_exp = pair_trading(prices, t1, t2, lookback, z_entry, z_exit, commission, lag_exp)
#     if best_result and res_trad and res_exp:
#         results.append({
#             "pair": f"{t1}-{t2}",
#             "lag": best_lag,
#             "sharpe_lag": best_result["sharpe"],
#             "growth_lag": best_result["growth"],
#             "drawdown_lag": best_result["drawdown"],
#             "n_trades_lag": best_result["n_trades"],
#             "df_lag": best_result["df"],
#             "sharpe_trad": res_trad["sharpe"],
#             "growth_trad": res_trad["growth"],
#             "drawdown_trad": res_trad["drawdown"],
#             "n_trades_trad": res_trad["n_trades"],
#             "df_trad": res_trad["df"],
#             "lag_exp": lag_exp,
#             "sharpe_exp": res_exp["sharpe"],
#             "growth_exp": res_exp["growth"],
#             "drawdown_exp": res_exp["drawdown"],
#             "n_trades_exp": res_exp["n_trades"],
#             "df_exp": res_exp["df"]
#         })

# results_df = pd.DataFrame(results)
# results_df = results_df.sort_values(by="sharpe_lag", ascending=False)

# st.subheader("Top 5 pares y desfases por Sharpe Ratio (con comisiones)")
# st.dataframe(results_df[["pair", "lag", "sharpe_lag", "growth_lag", "lag_exp", "sharpe_exp", "growth_exp", "sharpe_trad", "growth_trad"]].head(5))

# selected_row = st.selectbox(
#     "Selecciona un par para comparar estrategias",
#     results_df["pair"] + " (lag={})".format(results_df["lag"])
# )
# row = results_df.loc[
#     results_df["pair"] + " (lag={})".format(results_df["lag"]) == selected_row
# ].iloc[0]

# df_lag = row["df_lag"]
# df_trad = row["df_trad"]
# df_exp = row["df_exp"]

# # --- GRAFICA COMPARATIVA: TODO EL PERÍODO ---
# st.subheader(f"Comparación de estrategias para {row['pair']} (todo el período)")
# fig, ax = plt.subplots(figsize=(10, 5))
# ax.plot(df_lag.index, df_lag["Cumulative"], label=f"Lead-Lag óptimo estrategia (lag={row['lag']})")
# ax.plot(df_exp.index, df_exp["Cumulative"], label=f"Lead-Lag exploratorio (lag={row['lag_exp']})", linestyle='-.')
# ax.plot(df_trad.index, df_trad["Cumulative"], label="Tradicional (lag=0)", linestyle='--')
# ax.set_ylabel("Crecimiento acumulado ($ inicial = 1)")
# ax.set_title(f"Crecimiento acumulado: {row['pair']}")
# ax.legend()
# st.pyplot(fig)

# st.subheader("Métricas detalladas")
# st.write(f"**Lead-Lag óptimo estrategia (lag={row['lag']}):** Sharpe={row['sharpe_lag']:.2f} | Growth={row['growth_lag']:.2f}x | Drawdown={row['drawdown_lag']:.2%} | Operaciones={row['n_trades_lag']}")
# st.write(f"**Lead-Lag exploratorio (lag={row['lag_exp']}):** Sharpe={row['sharpe_exp']:.2f} | Growth={row['growth_exp']:.2f}x | Drawdown={row['drawdown_exp']:.2%} | Operaciones={row['n_trades_exp']}")
# st.write(f"**Tradicional (lag=0):** Sharpe={row['sharpe_trad']:.2f} | Growth={row['growth_trad']:.2f}x | Drawdown={row['drawdown_trad']:.2%} | Operaciones={row['n_trades_trad']}")

# st.caption("Compara cómo cambia la estrategia usando el lag óptimo exploratorio (científico) vs. el óptimo de la estrategia y el enfoque tradicional.")

# # --- TEST FUERA DE MUESTRA: ÚLTIMO AÑO ---
# st.subheader("Comparación fuera de muestra (último año)")

# # 1. Encuentra la fecha de corte (último año completo disponible)
# all_dates = df_lag.index
# if len(all_dates) < 252:
#     st.warning("No hay suficiente historial para realizar test fuera de muestra (último año).")
# else:
#     cutoff_date = all_dates[-252]
#     train_prices = prices.loc[:cutoff_date]
#     test_prices = prices.loc[cutoff_date:]

#     # Lags óptimos sobre el set de entrenamiento
#     lag_trading_train = None
#     lag_exp_train = None
#     # Encuentra lag óptimo de la estrategia en el set de entrenamiento
#     best_sharpe_train = -np.inf
#     for lag in range(-max_lag, max_lag+1):
#         if lag == 0:
#             continue
#         res = pair_trading(train_prices, row["pair"].split("-")[0], row["pair"].split("-")[1], lookback, z_entry, z_exit, commission, lag)
#         if res and res["sharpe"] > best_sharpe_train:
#             best_sharpe_train = res["sharpe"]
#             lag_trading_train = lag
#     lag_exp_train = lag_min_autocorr(train_prices, row["pair"].split("-")[0], row["pair"].split("-")[1], max_lag, lookback)

#     # Simula en el set de test
#     res_lag_test = pair_trading(test_prices, row["pair"].split("-")[0], row["pair"].split("-")[1], lookback, z_entry, z_exit, commission, lag_trading_train)
#     res_exp_test = pair_trading(test_prices, row["pair"].split("-")[0], row["pair"].split("-")[1], lookback, z_entry, z_exit, commission, lag_exp_train)
#     res_trad_test = pair_trading(test_prices, row["pair"].split("-")[0], row["pair"].split("-")[1], lookback, z_entry, z_exit, commission, lag=0)

#     fig2, ax2 = plt.subplots(figsize=(10, 4))
#     if res_lag_test:
#         ax2.plot(res_lag_test["df"].index, res_lag_test["df"]["Cumulative"], label=f"Lead-Lag óptimo estrategia (lag={lag_trading_train})")
#     if res_exp_test:
#         ax2.plot(res_exp_test["df"].index, res_exp_test["df"]["Cumulative"], label=f"Lead-Lag exploratorio (lag={lag_exp_train})", linestyle='-.')
#     if res_trad_test:
#         ax2.plot(res_trad_test["df"].index, res_trad_test["df"]["Cumulative"], label="Tradicional (lag=0)", linestyle='--')
#     ax2.set_ylabel("Crecimiento acumulado ($ inicial = 1)")
#     ax2.set_title(f"Test fuera de muestra - Último año: {row['pair']}")
#     ax2.legend()
#     st.pyplot(fig2)

#     st.write("**Resultados fuera de muestra (último año):**")
#     if res_lag_test:
#         st.write(f"Lead-Lag óptimo estrategia (lag={lag_trading_train}): Sharpe={res_lag_test['sharpe']:.2f} | Growth={res_lag_test['growth']:.2f}x")
#     if res_exp_test:
#         st.write(f"Lead-Lag exploratorio (lag={lag_exp_train}): Sharpe={res_exp_test['sharpe']:.2f} | Growth={res_exp_test['growth']:.2f}x")
#     if res_trad_test:
#         st.write(f"Tradicional (lag=0): Sharpe={res_trad_test['sharpe']:.2f} | Growth={res_trad_test['growth']:.2f}x")







# pair_trading_lead_lag_vs_traditional.py

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from itertools import combinations
from datetime import timedelta
import plotly.graph_objs as go
import matplotlib.pyplot as plt

st.set_page_config(page_title="Pair Trading: Lead-Lag vs Tradicional", layout="wide")
st.title("🔁 Pair Trading: Alineamiento Temporal (Lead-Lag) vs Tradicional")

st.sidebar.header("Parámetros del backtest")
top_tickers = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "META", "NVDA"]
commission = st.sidebar.number_input("Comisión ida+vuelta (ej: 0.001 = 0.1%)", value=0.001, step=0.0001, format="%.4f")
lookback = st.sidebar.slider("Ventana rolling (días)", 20, 120, 60)
z_entry = st.sidebar.slider("Z-score entrada", 1.0, 3.0, 2.0)
z_exit = st.sidebar.slider("Z-score salida", 0.0, 2.0, 0.5)
max_lag = st.sidebar.slider("Desfase máximo a buscar (días)", 0, 15, 5)

@st.cache_data(show_spinner=False)
def load_prices(tickers):
    df = yf.download(tickers, start="2018-01-01", progress=False)["Close"]
    if isinstance(df, pd.Series):
        df = df.to_frame()
    return df

prices = load_prices(top_tickers).dropna()

def pair_trading(prices, t1, t2, lookback, z_entry, z_exit, commission, lag=0):
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
        "df": df,
        "lag": lag
    }

def lag_min_autocorr(prices, t1, t2, max_lag, lookback):
    lags = list(range(-max_lag, max_lag + 1))
    lags.remove(0)
    best_metric = np.inf
    best_lag = 0
    for lag in lags:
        if lag > 0:
            b_aligned = prices[t2].shift(lag)
            spread = np.log(prices[t1]) - np.log(b_aligned)
        elif lag < 0:
            a_aligned = prices[t1].shift(-lag)
            spread = np.log(a_aligned) - np.log(prices[t2])
        else:
            spread = np.log(prices[t1]) - np.log(prices[t2])
        ret = spread.diff().dropna()
        if len(ret) < lookback + 5:
            continue
        metric = abs(pd.Series(ret).autocorr(lag=1))
        if metric < best_metric:
            best_metric = metric
            best_lag = lag
    return best_lag

def get_signals(df, z_entry, z_exit):
    """Identifica puntos de compra/venta (apertura/cierre de posiciones)"""
    signals = []
    position = 0
    for idx, row in df.iterrows():
        if position == 0:
            # No posición
            if row["Zscore"] < -z_entry:
                signals.append(("long_open", idx, row["Cumulative"]))
                position = 1
            elif row["Zscore"] > z_entry:
                signals.append(("short_open", idx, row["Cumulative"]))
                position = -1
        elif position == 1:
            # Long abierta, busca cierre
            if abs(row["Zscore"]) < z_exit:
                signals.append(("long_close", idx, row["Cumulative"]))
                position = 0
        elif position == -1:
            # Short abierto, busca cierre
            if abs(row["Zscore"]) < z_exit:
                signals.append(("short_close", idx, row["Cumulative"]))
                position = 0
    return signals

# ---- BUSCA EL MEJOR LAG PARA CADA PAR ----
results = []
for t1, t2 in combinations(top_tickers, 2):
    best_sharpe = -np.inf
    best_lag = 0
    best_result = None
    for lag in range(-max_lag, max_lag+1):
        if lag == 0:
            continue
        res = pair_trading(prices, t1, t2, lookback, z_entry, z_exit, commission, lag)
        if res and res["sharpe"] > best_sharpe:
            best_sharpe = res["sharpe"]
            best_lag = lag
            best_result = res
    res_trad = pair_trading(prices, t1, t2, lookback, z_entry, z_exit, commission, lag=0)
    lag_exp = lag_min_autocorr(prices, t1, t2, max_lag, lookback)
    res_exp = pair_trading(prices, t1, t2, lookback, z_entry, z_exit, commission, lag_exp)
    if best_result and res_trad and res_exp:
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
            "lag_exp": lag_exp,
            "sharpe_exp": res_exp["sharpe"],
            "growth_exp": res_exp["growth"],
            "drawdown_exp": res_exp["drawdown"],
            "n_trades_exp": res_exp["n_trades"],
            "df_exp": res_exp["df"]
        })

results_df = pd.DataFrame(results)
results_df = results_df.sort_values(by="sharpe_lag", ascending=False)

st.subheader("Top 5 pares y desfases por Sharpe Ratio (con comisiones)")
st.dataframe(results_df[["pair", "lag", "sharpe_lag", "growth_lag", "lag_exp", "sharpe_exp", "growth_exp", "sharpe_trad", "growth_trad"]].head(5))

selected_row = st.selectbox(
    "Selecciona un par para comparar estrategias",
    results_df["pair"] + " (lag={})".format(results_df["lag"])
)
row = results_df.loc[
    results_df["pair"] + " (lag={})".format(results_df["lag"]) == selected_row
].iloc[0]

df_lag = row["df_lag"]
df_trad = row["df_trad"]
df_exp = row["df_exp"]

# --- GRAFICA COMPARATIVA: TODO EL PERÍODO ---
st.subheader(f"Comparación de estrategias para {row['pair']} (todo el período)")
fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(df_lag.index, df_lag["Cumulative"], label=f"Lead-Lag óptimo estrategia (lag={row['lag']})")
ax.plot(df_exp.index, df_exp["Cumulative"], label=f"Lead-Lag exploratorio (lag={row['lag_exp']})", linestyle='-.')
ax.plot(df_trad.index, df_trad["Cumulative"], label="Tradicional (lag=0)", linestyle='--')
ax.set_ylabel("Crecimiento acumulado ($ inicial = 1)")
ax.set_title(f"Crecimiento acumulado: {row['pair']}")
ax.legend()
st.pyplot(fig)

st.subheader("Métricas detalladas")
st.write(f"**Lead-Lag óptimo estrategia (lag={row['lag']}):** Sharpe={row['sharpe_lag']:.2f} | Growth={row['growth_lag']:.2f}x | Drawdown={row['drawdown_lag']:.2%} | Operaciones={row['n_trades_lag']}")
st.write(f"**Lead-Lag exploratorio (lag={row['lag_exp']}):** Sharpe={row['sharpe_exp']:.2f} | Growth={row['growth_exp']:.2f}x | Drawdown={row['drawdown_exp']:.2%} | Operaciones={row['n_trades_exp']}")
st.write(f"**Tradicional (lag=0):** Sharpe={row['sharpe_trad']:.2f} | Growth={row['growth_trad']:.2f}x | Drawdown={row['drawdown_trad']:.2%} | Operaciones={row['n_trades_trad']}")

st.caption("Compara cómo cambia la estrategia usando el lag óptimo exploratorio (científico) vs. el óptimo de la estrategia y el enfoque tradicional.")

# --- TEST FUERA DE MUESTRA: ÚLTIMO AÑO ---
st.subheader("Comparación fuera de muestra (último año)")

all_dates = df_lag.index
if len(all_dates) < 252:
    st.warning("No hay suficiente historial para realizar test fuera de muestra (último año).")
else:
    cutoff_date = all_dates[-252]
    train_prices = prices.loc[:cutoff_date]
    test_prices = prices.loc[cutoff_date:]

    lag_trading_train = None
    lag_exp_train = None
    best_sharpe_train = -np.inf
    for lag in range(-max_lag, max_lag+1):
        if lag == 0:
            continue
        res = pair_trading(train_prices, row["pair"].split("-")[0], row["pair"].split("-")[1], lookback, z_entry, z_exit, commission, lag)
        if res and res["sharpe"] > best_sharpe_train:
            best_sharpe_train = res["sharpe"]
            lag_trading_train = lag
    lag_exp_train = lag_min_autocorr(train_prices, row["pair"].split("-")[0], row["pair"].split("-")[1], max_lag, lookback)

    # Simula en el set de test
    res_lag_test = pair_trading(test_prices, row["pair"].split("-")[0], row["pair"].split("-")[1], lookback, z_entry, z_exit, commission, lag_trading_train)
    res_exp_test = pair_trading(test_prices, row["pair"].split("-")[0], row["pair"].split("-")[1], lookback, z_entry, z_exit, commission, lag_exp_train)
    res_trad_test = pair_trading(test_prices, row["pair"].split("-")[0], row["pair"].split("-")[1], lookback, z_entry, z_exit, commission, lag=0)

    fig2 = go.Figure()

    # --- Lead-Lag óptimo estrategia
    if res_lag_test:
        df = res_lag_test["df"]
        fig2.add_trace(go.Scatter(x=df.index, y=df["Cumulative"], mode='lines', name=f"Lead-Lag óptimo estrategia (lag={lag_trading_train})", line=dict(color="blue")))
        signals = get_signals(df, z_entry, z_exit)
        for typ, idx, val in signals:
            if typ == "long_open":
                fig2.add_trace(go.Scatter(x=[idx], y=[val], mode='markers', marker=dict(symbol="triangle-up", color="green", size=12), name="Abrir Long (estrategia)"))
            elif typ == "short_open":
                fig2.add_trace(go.Scatter(x=[idx], y=[val], mode='markers', marker=dict(symbol="triangle-down", color="red", size=12), name="Abrir Short (estrategia)"))
            elif typ == "long_close":
                fig2.add_trace(go.Scatter(x=[idx], y=[val], mode='markers', marker=dict(symbol="circle", color="green", size=10), name="Cerrar Long (estrategia)"))
            elif typ == "short_close":
                fig2.add_trace(go.Scatter(x=[idx], y=[val], mode='markers', marker=dict(symbol="circle", color="red", size=10), name="Cerrar Short (estrategia)"))

    # --- Lead-Lag exploratorio
    if res_exp_test:
        df = res_exp_test["df"]
        fig2.add_trace(go.Scatter(x=df.index, y=df["Cumulative"], mode='lines', name=f"Lead-Lag exploratorio (lag={lag_exp_train})", line=dict(color="orange")))
        signals = get_signals(df, z_entry, z_exit)
        for typ, idx, val in signals:
            if typ == "long_open":
                fig2.add_trace(go.Scatter(x=[idx], y=[val], mode='markers', marker=dict(symbol="triangle-up", color="lime", size=12), name="Abrir Long (exploratorio)"))
            elif typ == "short_open":
                fig2.add_trace(go.Scatter(x=[idx], y=[val], mode='markers', marker=dict(symbol="triangle-down", color="magenta", size=12), name="Abrir Short (exploratorio)"))
            elif typ == "long_close":
                fig2.add_trace(go.Scatter(x=[idx], y=[val], mode='markers', marker=dict(symbol="circle", color="lime", size=10), name="Cerrar Long (exploratorio)"))
            elif typ == "short_close":
                fig2.add_trace(go.Scatter(x=[idx], y=[val], mode='markers', marker=dict(symbol="circle", color="magenta", size=10), name="Cerrar Short (exploratorio)"))

    # --- Tradicional
    if res_trad_test:
        df = res_trad_test["df"]
        fig2.add_trace(go.Scatter(x=df.index, y=df["Cumulative"], mode='lines', name="Tradicional (lag=0)", line=dict(color="black")))
        signals = get_signals(df, z_entry, z_exit)
        for typ, idx, val in signals:
            if typ == "long_open":
                fig2.add_trace(go.Scatter(x=[idx], y=[val], mode='markers', marker=dict(symbol="triangle-up", color="blue", size=12), name="Abrir Long (tradicional)"))
            elif typ == "short_open":
                fig2.add_trace(go.Scatter(x=[idx], y=[val], mode='markers', marker=dict(symbol="triangle-down", color="red", size=12), name="Abrir Short (tradicional)"))
            elif typ == "long_close":
                fig2.add_trace(go.Scatter(x=[idx], y=[val], mode='markers', marker=dict(symbol="circle", color="blue", size=10), name="Cerrar Long (tradicional)"))
            elif typ == "short_close":
                fig2.add_trace(go.Scatter(x=[idx], y=[val], mode='markers', marker=dict(symbol="circle", color="red", size=10), name="Cerrar Short (tradicional)"))

    fig2.update_layout(
        title=f"Test fuera de muestra - Último año: {row['pair']}",
        yaxis_title="Crecimiento acumulado ($ inicial = 1)",
        legend=dict(itemsizing='constant'),
        height=400
    )
    st.plotly_chart(fig2, use_container_width=True)

    st.write("**Resultados fuera de muestra (último año):**")
    if res_lag_test:
        st.write(f"Lead-Lag óptimo estrategia (lag={lag_trading_train}): Sharpe={res_lag_test['sharpe']:.2f} | Growth={res_lag_test['growth']:.2f}x")
    if res_exp_test:
        st.write(f"Lead-Lag exploratorio (lag={lag_exp_train}): Sharpe={res_exp_test['sharpe']:.2f} | Growth={res_exp_test['growth']:.2f}x")
    if res_trad_test:
        st.write(f"Tradicional (lag=0): Sharpe={res_trad_test['sharpe']:.2f} | Growth={res_trad_test['growth']:.2f}x")
