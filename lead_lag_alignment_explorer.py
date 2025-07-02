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









# # lead_lag_alignment_comparator.py

# import streamlit as st
# import yfinance as yf
# import pandas as pd
# import numpy as np
# import plotly.graph_objs as go

# st.set_page_config(page_title="Comparador Lead-Lag Óptimo", layout="wide")
# st.title("🔎 Comparador Visual de Alineamiento Óptimo Lead-Lag")

# st.sidebar.header("Parámetros")
# tickers = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "META", "NVDA"]
# t1 = st.sidebar.selectbox("Activo líder (A)", tickers, 0)
# t2 = st.sidebar.selectbox("Activo seguidor (B)", tickers, 1)
# max_lag = st.sidebar.slider("Lag máximo a explorar (días)", 1, 15, 7)
# lookback = st.sidebar.slider("Rolling window (días)", 10, 120, 60)
# metric = st.sidebar.selectbox("Criterio para lag óptimo", ["Menor autocorrelación abs. (reversión)", "Máx. Sharpe de spread"], 0)

# st.write(
#     f"""
#     Compara el spread y z-score con **lag=0** y con **lag óptimo** detectado automáticamente
#     ({'mínima autocorrelación' if metric.startswith('Menor') else 'máx. Sharpe'}).
#     - Cambia los parámetros y observa la diferencia.
#     """
# )

# @st.cache_data(show_spinner=False)
# def load_prices(tickers):
#     df = yf.download(tickers, start="2018-01-01", progress=False)["Close"]
#     if isinstance(df, pd.Series):
#         df = df.to_frame()
#     return df

# prices = load_prices([t1, t2]).dropna()

# def calc_spread(prices, lag):
#     # lag>0: retrasar B, lag<0: retrasar A
#     if lag > 0:
#         b_aligned = prices[t2].shift(lag)
#         spread = np.log(prices[t1]) - np.log(b_aligned)
#     elif lag < 0:
#         a_aligned = prices[t1].shift(-lag)
#         spread = np.log(a_aligned) - np.log(prices[t2])
#     else:
#         spread = np.log(prices[t1]) - np.log(prices[t2])
#     return spread.dropna()

# def sharpe_of_spread(spread):
#     ret = spread.diff().dropna()
#     if ret.std() == 0 or np.isnan(ret.std()):
#         return -np.inf
#     return np.mean(ret) / np.std(ret) * np.sqrt(252)

# def min_autocorr_lag(spread):
#     # Retorno del spread (dif1) debe ser menos autocorrelado posible (reversión a la media)
#     ret = spread.diff().dropna()
#     return abs(pd.Series(ret).autocorr(lag=1))

# # --- BUSCA EL LAG ÓPTIMO ---
# lags = list(range(-max_lag, max_lag + 1))
# lags.remove(0)  # ya se compara por defecto
# metrics = []

# for lag in lags:
#     s = calc_spread(prices, lag)
#     if len(s) < lookback + 5:
#         metrics.append(np.nan)
#         continue
#     if metric.startswith("Máx. Sharpe"):
#         val = sharpe_of_spread(s)
#     else:
#         val = -min_autocorr_lag(s)  # Queremos minimizar autocorrelación abs.
#     metrics.append(val)

# if metric.startswith("Máx. Sharpe"):
#     best_idx = np.nanargmax(metrics)
# else:
#     best_idx = np.nanargmax(metrics)  # -autocorr, así máximo es mejor
# lag_opt = lags[best_idx]

# # -- Calcula spreads y z-score para lag=0 y lag óptimo --
# spread0 = calc_spread(prices, 0)
# spread_opt = calc_spread(prices, lag_opt)

# # Alinea longitudes
# common_idx = spread0.index.intersection(spread_opt.index)
# spread0 = spread0.loc[common_idx]
# spread_opt = spread_opt.loc[common_idx]

# spread_mean0 = spread0.rolling(window=lookback).mean()
# spread_std0 = spread0.rolling(window=lookback).std()
# zscore0 = (spread0 - spread_mean0) / spread_std0

# spread_mean_opt = spread_opt.rolling(window=lookback).mean()
# spread_std_opt = spread_opt.rolling(window=lookback).std()
# zscore_opt = (spread_opt - spread_mean_opt) / spread_std_opt

# # -- Plotly comparativo --
# st.subheader(f"Comparativo de Spread logarítmico (lag=0 vs lag óptimo {lag_opt:+d})")
# fig2 = go.Figure()
# fig2.add_trace(go.Scatter(x=spread0.index, y=spread0, name=f"Spread lag=0"))
# fig2.add_trace(go.Scatter(x=spread_opt.index, y=spread_opt, name=f"Spread lag óptimo ({lag_opt:+d})"))
# fig2.add_trace(go.Scatter(x=spread_mean0.index, y=spread_mean0, name="Media rolling lag=0", line=dict(dash='dash')))
# fig2.add_trace(go.Scatter(x=spread_mean_opt.index, y=spread_mean_opt, name="Media rolling lag óptimo", line=dict(dash='dash')))
# fig2.update_layout(height=350, yaxis_title="Spread")
# st.plotly_chart(fig2, use_container_width=True)

# st.subheader(f"Comparativo de Z-score (lag=0 vs lag óptimo {lag_opt:+d})")
# fig3 = go.Figure()
# fig3.add_trace(go.Scatter(x=zscore0.index, y=zscore0, name=f"Z-score lag=0"))
# fig3.add_trace(go.Scatter(x=zscore_opt.index, y=zscore_opt, name=f"Z-score lag óptimo ({lag_opt:+d})"))
# fig3.add_trace(go.Scatter(x=zscore0.index, y=np.zeros_like(zscore0), name="Media", line=dict(dash='dash')))
# fig3.update_layout(height=350, yaxis_title="Z-score")
# st.plotly_chart(fig3, use_container_width=True)

# # -- Información extra --
# st.info(f"""
# **Lag óptimo detectado:** {lag_opt:+d} días  
# **Criterio:** {'Sharpe máximo' if metric.startswith('Máx') else 'Menor autocorrelación absoluta (reversión)'}
# """)

# st.caption(
#     "Observa si con el lag óptimo el spread y/o su z-score muestran reversión a la media más clara o cruces por la media más simétricos. "
#     "Esto respalda visualmente la selección del lag óptimo usado en los modelos de previsión y trading."
# )









# # lead_lag_alignment_comparator.py

# import streamlit as st
# import yfinance as yf
# import pandas as pd
# import numpy as np
# import plotly.graph_objs as go

# st.set_page_config(page_title="Comparador Visual de Alinhamento Lead-Lag", layout="wide")
# st.title("🔎 Comparador Visual do Alinhamento Lead-Lag")

# st.sidebar.header("Parâmetros")
# tickers = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "META", "NVDA"]
# t1 = st.sidebar.selectbox("Ativo líder (A)", tickers, 0)
# t2 = st.sidebar.selectbox("Ativo seguidor (B)", tickers, 1)
# max_lag = st.sidebar.slider("Lag máximo a explorar (dias)", 1, 15, 7)
# lookback = st.sidebar.slider("Janela rolling (dias)", 10, 120, 60)

# st.write(
#     f"""
#     Este painel compara diferentes critérios de alinhamento temporal (lead-lag) entre os ativos selecionados:
#     - **Correlação dos retornos** entre {t1} e {t2} com diferentes lags.
#     - **Sharpe Ratio dos retornos acumulados** ao manter uma posição (buy-and-hold) com o lag aplicado.
#     - **Pontos de inflexão** para cada métrica, mostrando os lags aprimorados.
#     - **Spreads e z-score** para o lag=0 e para os lags aprimorados.
#     Explore diferentes parâmetros para identificar relações de liderança temporal e avaliar possíveis oportunidades de previsão!
#     """
# )

# @st.cache_data(show_spinner=False)
# def load_prices(tickers):
#     df = yf.download(tickers, start="2018-01-01", progress=False)["Close"]
#     if isinstance(df, pd.Series):
#         df = df.to_frame()
#     return df

# prices = load_prices([t1, t2]).dropna()

# def calc_spread(prices, lag):
#     if lag > 0:
#         b_aligned = prices[t2].shift(lag)
#         spread = np.log(prices[t1]) - np.log(b_aligned)
#     elif lag < 0:
#         a_aligned = prices[t1].shift(-lag)
#         spread = np.log(a_aligned) - np.log(prices[t2])
#     else:
#         spread = np.log(prices[t1]) - np.log(prices[t2])
#     return spread.dropna()

# def correlation_returns(prices, lag):
#     if lag > 0:
#         ret_a = np.log(prices[t1] / prices[t1].shift(1))
#         ret_b = np.log(prices[t2].shift(lag) / prices[t2].shift(lag+1))
#     elif lag < 0:
#         ret_a = np.log(prices[t1].shift(-lag) / prices[t1].shift(-lag-1))
#         ret_b = np.log(prices[t2] / prices[t2].shift(1))
#     else:
#         ret_a = np.log(prices[t1] / prices[t1].shift(1))
#         ret_b = np.log(prices[t2] / prices[t2].shift(1))
#     ret_df = pd.DataFrame({"A": ret_a, "B": ret_b}).dropna()
#     return ret_df["A"].corr(ret_df["B"])

# def sharpe_hold(prices, lag):
#     # Sharpe dos retornos cumulativos (buy-and-hold, só para ilustrar efeito do lag)
#     if lag > 0:
#         a = prices[t1]
#         b = prices[t2].shift(lag)
#     elif lag < 0:
#         a = prices[t1].shift(-lag)
#         b = prices[t2]
#     else:
#         a = prices[t1]
#         b = prices[t2]
#     spread = np.log(a) - np.log(b)
#     returns = spread.diff().dropna()
#     if returns.std() == 0 or np.isnan(returns.std()):
#         return -np.inf
#     return np.mean(returns) / np.std(returns) * np.sqrt(252)

# def min_autocorr_lag(prices, lag):
#     # Autocorrelação dos retornos do spread
#     s = calc_spread(prices, lag)
#     ret = s.diff().dropna()
#     return abs(pd.Series(ret).autocorr(lag=1))

# # --- Explora todos os lags ---
# lags = list(range(-max_lag, max_lag + 1))
# lags.remove(0)
# correls = []
# sharpes = []
# autocorrs = []

# for lag in lags:
#     correls.append(correlation_returns(prices, lag))
#     sharpes.append(sharpe_hold(prices, lag))
#     autocorrs.append(-min_autocorr_lag(prices, lag))  # Inverter para maximizar

# # Lags aprimorados
# lag_corr = lags[np.nanargmax(correls)]
# lag_sharpe = lags[np.nanargmax(sharpes)]
# lag_auto = lags[np.nanargmax(autocorrs)]  # Menor autocorrelação

# # --- Gráfico: Correlacao e Sharpe x Lag ---
# st.subheader("Correlação dos retornos e Sharpe Ratio em função do lag")

# fig1 = go.Figure()
# fig1.add_trace(go.Scatter(x=lags, y=correls, mode='lines+markers', name="Correlação dos retornos"))
# fig1.add_trace(go.Scatter(x=lags, y=sharpes, mode='lines+markers', name="Sharpe Ratio (buy-and-hold)"))
# fig1.add_trace(go.Scatter(x=lags, y=autocorrs, mode='lines+markers', name="(-) Autocorrelação abs. do spread"))
# fig1.add_vline(x=lag_corr, line=dict(color="orange", dash="dash"), annotation_text=f"Máx. correlação: {lag_corr}", annotation_position="top left")
# fig1.add_vline(x=lag_sharpe, line=dict(color="purple", dash="dash"), annotation_text=f"Máx. Sharpe: {lag_sharpe}", annotation_position="top right")
# fig1.add_vline(x=lag_auto, line=dict(color="green", dash="dash"), annotation_text=f"Menor autocorr.: {lag_auto}", annotation_position="bottom right")
# fig1.update_layout(
#     xaxis_title="Lag aplicado (dias)",
#     yaxis_title="Valor da métrica",
#     legend=dict(x=0.01, y=0.99),
#     height=350
# )
# st.plotly_chart(fig1, use_container_width=True)

# st.markdown(
#     f"""
#     - **Lag com maior correlação dos retornos:** {lag_corr}
#     - **Lag com maior Sharpe Ratio (retorno acumulado):** {lag_sharpe}
#     - **Lag com menor autocorrelação dos retornos do spread:** {lag_auto}
#     """
# )

# # --- Spreads e z-scores para os lags destacados ---
# def plot_spread_zscore(lag, label):
#     spread = calc_spread(prices, lag)
#     spread_mean = spread.rolling(window=lookback).mean()
#     spread_std = spread.rolling(window=lookback).std()
#     zscore = (spread - spread_mean) / spread_std

#     st.subheader(f"Spread e Z-score para {label} (lag={lag})")
#     fig2 = go.Figure()
#     fig2.add_trace(go.Scatter(x=spread.index, y=spread, name="Spread"))
#     fig2.add_trace(go.Scatter(x=spread_mean.index, y=spread_mean, name="Média rolling", line=dict(dash='dash')))
#     fig2.update_layout(height=250, yaxis_title="Spread")
#     st.plotly_chart(fig2, use_container_width=True)

#     fig3 = go.Figure()
#     fig3.add_trace(go.Scatter(x=zscore.index, y=zscore, name="Z-score"))
#     fig3.add_trace(go.Scatter(x=zscore.index, y=np.zeros_like(zscore), name="Média", line=dict(dash='dash')))
#     fig3.update_layout(height=250, yaxis_title="Z-score")
#     st.plotly_chart(fig3, use_container_width=True)

# st.markdown("## Comparação dos spreads e z-scores para diferentes lags aprimorados")

# st.markdown("### (A) Lag = 0 (sem alinhamento)")
# plot_spread_zscore(0, "Sem alinhamento")

# st.markdown(f"### (B) Lag com maior correlação dos retornos (lag={lag_corr})")
# plot_spread_zscore(lag_corr, "Máx. correlação")

# st.markdown(f"### (C) Lag com maior Sharpe Ratio (lag={lag_sharpe})")
# plot_spread_zscore(lag_sharpe, "Máx. Sharpe Ratio")

# st.markdown(f"### (D) Lag com menor autocorrelação dos retornos do spread (lag={lag_auto})")
# plot_spread_zscore(lag_auto, "Menor autocorrelação")

# st.info(
#     "Explore os gráficos para visualizar como cada critério de alinhamento temporal afeta o comportamento do spread e do z-score. "
#     "Isso pode revelar oportunidades de previsão ou arbitragem estatística entre os ativos selecionados."
# )






# # lead_lag_alignment_comparator.py

# import streamlit as st
# import yfinance as yf
# import pandas as pd
# import numpy as np
# import plotly.graph_objs as go

# st.set_page_config(page_title="Comparador Visual de Alinhamento Lead-Lag", layout="wide")
# st.title("🔎 Comparador Visual de Alinhamento Temporal (Lead-Lag)")

# st.sidebar.header("Parâmetros")
# tickers = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "META", "NVDA"]
# t1 = st.sidebar.selectbox("Ativo líder (A)", tickers, 0)
# t2 = st.sidebar.selectbox("Ativo seguidor (B)", tickers, 1)
# max_lag = st.sidebar.slider("Lag máximo a explorar (dias)", 1, 15, 7)
# lookback = st.sidebar.slider("Janela rolling (dias)", 10, 120, 60)
# z_entry = st.sidebar.slider("Z-score de entrada", 1.0, 3.0, 2.0)
# z_exit = st.sidebar.slider("Z-score de saída", 0.0, 2.0, 0.5)
# commission = st.sidebar.number_input("Comissão ida+volta (ex: 0.001 = 0.1%)", value=0.001, step=0.0001, format="%.4f")

# st.markdown(
#     """
#     <span style='font-size:1.1em'>
#     Este painel compara **dois critérios científicos de alinhamento temporal (lead-lag)**:
#     <ul>
#       <li><b>Lag por reversão à média:</b> selecionado pela menor autocorrelação dos retornos do spread.</li>
#       <li><b>Lag por Sharpe da estratégia:</b> selecionado maximizando o Sharpe anualizado da estratégia de trading, com os mesmos parâmetros do simulador.</li>
#     </ul>
#     O gráfico central mostra as duas métricas para cada lag, com destaque visual para os lags aprimorados. 
#     </span>
#     """,
#     unsafe_allow_html=True
# )

# @st.cache_data(show_spinner=False)
# def load_prices(tickers):
#     df = yf.download(tickers, start="2018-01-01", progress=False)["Close"]
#     if isinstance(df, pd.Series):
#         df = df.to_frame()
#     return df

# prices = load_prices([t1, t2]).dropna()

# def calc_spread(prices, lag):
#     if lag > 0:
#         b_aligned = prices[t2].shift(lag)
#         spread = np.log(prices[t1]) - np.log(b_aligned)
#     elif lag < 0:
#         a_aligned = prices[t1].shift(-lag)
#         spread = np.log(a_aligned) - np.log(prices[t2])
#     else:
#         spread = np.log(prices[t1]) - np.log(prices[t2])
#     return spread.dropna()

# def autocorr_spread(prices, lag, lookback):
#     spread = calc_spread(prices, lag)
#     ret = spread.diff().dropna()
#     if len(ret) < lookback + 5:
#         return np.nan
#     return abs(pd.Series(ret).autocorr(lag=1))

# def sharpe_strategy(prices, lag, lookback, z_entry, z_exit, commission):
#     if lag > 0:
#         t2_aligned = prices[t2].shift(lag)
#         df = pd.DataFrame({t1: prices[t1], t2: t2_aligned}).dropna()
#     elif lag < 0:
#         t1_aligned = prices[t1].shift(-lag)
#         df = pd.DataFrame({t1: t1_aligned, t2: prices[t2]}).dropna()
#     else:
#         df = pd.DataFrame({t1: prices[t1], t2: prices[t2]}).dropna()
#     if len(df) < lookback + 10:
#         return np.nan
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
#     if df["Strategy_ret"].std() > 0:
#         return np.mean(df["Strategy_ret"]) / np.std(df["Strategy_ret"]) * np.sqrt(252)
#     else:
#         return np.nan

# # --- Calcula métricas para cada lag ---
# lags = list(range(-max_lag, max_lag + 1))
# lags.remove(0)
# autocorrs = []
# sharpes = []
# for lag in lags:
#     autocorrs.append(autocorr_spread(prices, lag, lookback))
#     sharpes.append(sharpe_strategy(prices, lag, lookback, z_entry, z_exit, commission))

# # Seleciona os lags aprimorados
# lag_auto = lags[np.nanargmin(autocorrs)]  # menor autocorrelação
# lag_sharpe = lags[np.nanargmax(sharpes)]  # maior Sharpe

# # --- Gráfico duplo eixo y: autocorrelação e Sharpe vs. lag ---
# st.subheader("Evolução das métricas em função do lag (alinhamento temporal)")
# fig = go.Figure()

# fig.add_trace(go.Scatter(x=lags, y=autocorrs, name="Autocorrelação absoluta", yaxis="y1", line=dict(color="green")))
# fig.add_trace(go.Scatter(x=lags, y=sharpes, name="Sharpe Ratio estratégia", yaxis="y2", line=dict(color="purple")))

# fig.add_vline(x=lag_auto, line=dict(color="green", dash="dash"), annotation_text=f"Menor autocorr.: {lag_auto}", annotation_position="top left")
# fig.add_vline(x=lag_sharpe, line=dict(color="purple", dash="dash"), annotation_text=f"Máx. Sharpe: {lag_sharpe}", annotation_position="top right")

# fig.update_layout(
#     xaxis=dict(title="Lag aplicado (dias)"),
#     yaxis=dict(title="Autocorrelação abs.", tickfont=dict(color="green")),
#     yaxis2=dict(title="Sharpe Ratio estratégia", tickfont=dict(color="purple"), anchor="x", overlaying="y", side="right"),
#     legend=dict(x=0.01, y=0.99),
#     height=370
# )
# st.plotly_chart(fig, use_container_width=True)

# st.markdown(
#     f"""
#     - <span style="color:green"><b>Lag com menor autocorrelação:</b> {lag_auto}</span>
#     - <span style="color:purple"><b>Lag com maior Sharpe Ratio:</b> {lag_sharpe}</span>
#     """, unsafe_allow_html=True
# )

# # --- Spreads e z-scores para os lags aprimorados ---
# def plot_spread_zscore(lag, label):
#     spread = calc_spread(prices, lag)
#     spread_mean = spread.rolling(window=lookback).mean()
#     spread_std = spread.rolling(window=lookback).std()
#     zscore = (spread - spread_mean) / spread_std

#     st.subheader(f"Spread e Z-score para {label} (lag={lag})")
#     fig2 = go.Figure()
#     fig2.add_trace(go.Scatter(x=spread.index, y=spread, name="Spread"))
#     fig2.add_trace(go.Scatter(x=spread_mean.index, y=spread_mean, name="Média rolling", line=dict(dash='dash')))
#     fig2.update_layout(height=240, yaxis_title="Spread")
#     st.plotly_chart(fig2, use_container_width=True)

#     fig3 = go.Figure()
#     fig3.add_trace(go.Scatter(x=zscore.index, y=zscore, name="Z-score"))
#     fig3.add_trace(go.Scatter(x=zscore.index, y=np.zeros_like(zscore), name="Média", line=dict(dash='dash')))
#     fig3.update_layout(height=200, yaxis_title="Z-score")
#     st.plotly_chart(fig3, use_container_width=True)

# st.markdown("## Visualização dos spreads e z-scores para os diferentes lags aprimorados")
# st.markdown("### (A) Lag = 0 (sem alinhamento)")
# plot_spread_zscore(0, "Sem alinhamento")

# st.markdown(f"### (B) Lag com menor autocorrelação dos retornos do spread (lag={lag_auto})")
# plot_spread_zscore(lag_auto, "Lag aprimorado por reversão à média")

# st.markdown(f"### (C) Lag com maior Sharpe Ratio da estratégia (lag={lag_sharpe})")
# plot_spread_zscore(lag_sharpe, "Lag aprimorado pelo Sharpe da estratégia")

# st.info(
#     "O gráfico duplo acima mostra claramente os pontos de inflexão para cada critério, destacando visualmente o lag selecionado por reversão à média e o lag selecionado pelo Sharpe da estratégia. "
#     "Os gráficos abaixo permitem comparar o comportamento do spread e do z-score para cada lag de interesse, apoiando decisões científicas e justificadas para sua aplicação em trading."
# )









# comparador_lead_lag.py

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objs as go

st.set_page_config(page_title="Comparador Visual de Alinhamento Temporal (Lead-Lag)", layout="wide")
st.title("🔎 Comparador Visual de Alinhamento Temporal (Lead-Lag)")

st.sidebar.header("Parâmetros")
tickers = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "META", "NVDA"]
t1 = st.sidebar.selectbox("Ativo líder (A)", tickers, 0)
t2 = st.sidebar.selectbox("Ativo seguidor (B)", tickers, 1)
max_lag = st.sidebar.slider("Lag máximo a explorar (dias)", 1, 15, 7)
lookback = st.sidebar.slider("Janela rolling (dias)", 10, 120, 60)
z_entry = st.sidebar.slider("Z-score de entrada", 1.0, 3.0, 2.0)
z_exit = st.sidebar.slider("Z-score de saída", 0.0, 2.0, 0.5)
commission = st.sidebar.number_input("Comissão ida+volta (ex: 0.001 = 0.1%)", value=0.001, step=0.0001, format="%.4f")

st.markdown(
    """
    <span style='font-size:1.1em'>
    Este painel compara **dois critérios científicos de alinhamento temporal (lead-lag)**:
    <ul>
      <li><b>Lag por reversão à média:</b> Seleciona o lag com menor autocorrelação dos retornos do spread.</li>
      <li><b>Lag por Sharpe da estratégia:</b> Seleciona o lag que maximiza o Sharpe anualizado da estratégia de trading, com os parâmetros do simulador.</li>
    </ul>
    O gráfico central mostra as duas métricas para cada lag, com destaque visual para os lags otimizados.
    </span>
    """,
    unsafe_allow_html=True
)

@st.cache_data(show_spinner=False)
def carregar_precos(tickers):
    df = yf.download(tickers, start="2018-01-01", progress=False)["Close"]
    if isinstance(df, pd.Series):
        df = df.to_frame()
    return df

precos = carregar_precos([t1, t2]).dropna()

def calcular_spread(precos, lag):
    if lag > 0:
        b_alinhado = precos[t2].shift(lag)
        spread = np.log(precos[t1]) - np.log(b_alinhado)
    elif lag < 0:
        a_alinhado = precos[t1].shift(-lag)
        spread = np.log(a_alinhado) - np.log(precos[t2])
    else:
        spread = np.log(precos[t1]) - np.log(precos[t2])
    return spread.dropna()

def autocorr_spread(precos, lag, lookback):
    spread = calcular_spread(precos, lag)
    ret = spread.diff().dropna()
    if len(ret) < lookback + 5:
        return np.nan
    return abs(pd.Series(ret).autocorr(lag=1))

def sharpe_estrategia(precos, lag, lookback, z_entry, z_exit, commission):
    if lag > 0:
        t2_alinhado = precos[t2].shift(lag)
        df = pd.DataFrame({t1: precos[t1], t2: t2_alinhado}).dropna()
    elif lag < 0:
        t1_alinhado = precos[t1].shift(-lag)
        df = pd.DataFrame({t1: t1_alinhado, t2: precos[t2]}).dropna()
    else:
        df = pd.DataFrame({t1: precos[t1], t2: precos[t2]}).dropna()
    if len(df) < lookback + 10:
        return np.nan
    log_spread = np.log(df[t1] / df[t2])
    spread_mean = log_spread.rolling(window=lookback).mean()
    spread_std = log_spread.rolling(window=lookback).std()
    zscore = (log_spread - spread_mean) / spread_std
    df["Zscore"] = zscore
    df["Sinal"] = 0
    df.loc[df["Zscore"] < -z_entry, "Sinal"] = 1
    df.loc[df["Zscore"] > z_entry, "Sinal"] = -1
    df.loc[(df["Zscore"].abs() < z_exit), "Sinal"] = 0
    df["Posicao"] = df["Sinal"].replace(to_replace=0, method="ffill").fillna(0)
    df["Retorno_spread"] = np.log(df[t1] / df[t1].shift(1)) - np.log(df[t2] / df[t2].shift(1))
    df["Retorno_estrategia"] = df["Posicao"].shift(1) * df["Retorno_spread"]
    df["Trade"] = df["Posicao"].diff().abs()
    df["Retorno_estrategia"] -= commission * df["Trade"]
    if df["Retorno_estrategia"].std() > 0:
        return np.mean(df["Retorno_estrategia"]) / np.std(df["Retorno_estrategia"]) * np.sqrt(252)
    else:
        return np.nan

# --- Calcula métricas para cada lag ---
lags = list(range(-max_lag, max_lag + 1))
if 0 in lags:
    lags.remove(0)
autocorrs = []
sharpes = []
for lag in lags:
    autocorrs.append(autocorr_spread(precos, lag, lookback))
    sharpes.append(sharpe_estrategia(precos, lag, lookback, z_entry, z_exit, commission))

# Seleciona os lags otimizados
lag_auto = lags[np.nanargmin(autocorrs)]  # menor autocorrelação
lag_sharpe = lags[np.nanargmax(sharpes)]  # maior Sharpe

# --- Gráfico duplo eixo y: autocorrelação e Sharpe vs. lag ---
st.subheader("Evolução das métricas em função do lag (alinhamento temporal)")
fig = go.Figure()

fig.add_trace(go.Scatter(x=lags, y=autocorrs, name="Autocorrelação absoluta", yaxis="y1", line=dict(color="green")))
fig.add_trace(go.Scatter(x=lags, y=sharpes, name="Sharpe Ratio da estratégia", yaxis="y2", line=dict(color="purple")))

fig.add_vline(x=lag_auto, line=dict(color="green", dash="dash"), annotation_text=f"Menor autocorr.: {lag_auto}", annotation_position="top left")
fig.add_vline(x=lag_sharpe, line=dict(color="purple", dash="dash"), annotation_text=f"Máx. Sharpe: {lag_sharpe}", annotation_position="top right")

fig.update_layout(
    xaxis=dict(title="Lag aplicado (dias)"),
    yaxis=dict(title="Autocorrelação abs.", tickfont=dict(color="green")),
    yaxis2=dict(title="Sharpe Ratio estratégia", tickfont=dict(color="purple"), anchor="x", overlaying="y", side="right"),
    legend=dict(x=0.01, y=0.99),
    height=370
)
st.plotly_chart(fig, use_container_width=True)

st.markdown(
    f"""
    - <span style="color:green"><b>Lag com menor autocorrelação:</b> {lag_auto}</span>
    - <span style="color:purple"><b>Lag com maior Sharpe Ratio:</b> {lag_sharpe}</span>
    """, unsafe_allow_html=True
)

# --- Visualização dos spreads e z-scores para os lags otimizados ---
def plot_spread_zscore(lag, label):
    spread = calcular_spread(precos, lag)
    spread_mean = spread.rolling(window=lookback).mean()
    spread_std = spread.rolling(window=lookback).std()
    zscore = (spread - spread_mean) / spread_std

    st.subheader(f"Spread e Z-score para {label} (lag={lag})")
    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(x=spread.index, y=spread, name="Spread"))
    fig2.add_trace(go.Scatter(x=spread_mean.index, y=spread_mean, name="Média móvel", line=dict(dash='dash')))
    fig2.update_layout(height=240, yaxis_title="Spread")
    st.plotly_chart(fig2, use_container_width=True)

    fig3 = go.Figure()
    fig3.add_trace(go.Scatter(x=zscore.index, y=zscore, name="Z-score"))
    fig3.add_trace(go.Scatter(x=zscore.index, y=np.zeros_like(zscore), name="Média", line=dict(dash='dash')))
    fig3.update_layout(height=200, yaxis_title="Z-score")
    st.plotly_chart(fig3, use_container_width=True)

st.markdown("## Visualização dos spreads e z-scores para diferentes lags otimizados")
st.markdown("### (A) Lag = 0 (sem alinhamento)")
plot_spread_zscore(0, "Sem alinhamento")

st.markdown(f"### (B) Lag com menor autocorrelação dos retornos do spread (lag={lag_auto})")
plot_spread_zscore(lag_auto, "Lag otimizado por reversão à média")

st.markdown(f"### (C) Lag com maior Sharpe Ratio da estratégia (lag={lag_sharpe})")
plot_spread_zscore(lag_sharpe, "Lag otimizado pelo Sharpe da estratégia")

st.info(
    "O gráfico duplo acima destaca visualmente os pontos de inflexão para cada critério, mostrando o lag selecionado por reversão à média e o lag selecionado pelo Sharpe da estratégia. "
    "Os gráficos abaixo permitem comparar o comportamento do spread e do z-score para cada lag de interesse, apoiando decisões científicas justificadas para aplicação em trading."
)

