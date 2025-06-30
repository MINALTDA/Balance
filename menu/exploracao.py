# # menu/exploracao.py


# import streamlit as st
# import pandas as pd
# import numpy as np
# import yfinance as yf
# import random
# import plotly.graph_objs as go
# import plotly.colors

# TICKERS_LIST = [
#     "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA", "BRK-B", "JPM", "V",
#     "UNH", "JNJ", "XOM", "PG", "MA", "HD", "CVX", "MRK", "ABBV", "PEP",
#     "LLY", "KO", "AVGO", "COST", "WMT", "DIS", "BAC", "MCD", "ADBE", "PFE",
#     "TMO", "CSCO", "DHR", "ABT", "ACN", "CRM", "TXN", "NKE", "VZ", "NEE",
#     "CMCSA", "WFC", "LIN", "INTC", "QCOM", "MS", "HON", "PM", "UNP", "AMGN",
#     "SBUX", "ORCL", "AMD", "CVS", "ISRG", "GILD", "BKNG", "NOW", "ZTS", "MDLZ"
# ]

# def get_valid_random_stocks(n=50, days=30, max_lag=30, max_attempts=5):
#     tried = set()
#     valid_tickers = []
#     attempt = 0
#     while len(valid_tickers) < n and attempt < max_attempts:
#         tickers_to_try = list(set(TICKERS_LIST) - tried)
#         if not tickers_to_try:
#             break
#         sample_size = min(n - len(valid_tickers) + 10, len(tickers_to_try))
#         tickers_sample = random.sample(tickers_to_try, sample_size)
#         tried.update(tickers_sample)
#         # Descargar con margen extra para los lags
#         data = yf.download(tickers_sample, period=f'{days+max_lag+10}d', interval='1d', progress=False)
#         if 'Adj Close' in data:
#             prices = data['Adj Close']
#         elif 'Close' in data:
#             prices = data['Close']
#         else:
#             prices = pd.DataFrame()
#         filtered = [col for col in prices.columns if prices[col].dropna().shape[0] >= days + max_lag]
#         found = [col for col in filtered if col not in valid_tickers]
#         valid_tickers.extend(found)
#         attempt += 1
#     if not valid_tickers:
#         st.error("Não foi possível encontrar ações com dados válidos. Verifique sua conexão, tente novamente ou diminua o número de ações.")
#         st.stop()
#     if len(valid_tickers) < n:
#         st.warning(f"Apenas {len(valid_tickers)} ações com dados válidos foram encontradas. Usando todas.")
#         n = len(valid_tickers)
#     chosen = valid_tickers[:n]
#     final_data = yf.download(chosen, period=f'{days+max_lag+10}d', interval='1d', progress=False)
#     if 'Adj Close' in final_data:
#         available = [c for c in chosen if c in final_data['Adj Close'].columns]
#         all_prices = final_data['Adj Close'][available]
#     elif 'Close' in final_data:
#         available = [c for c in chosen if c in final_data['Close'].columns]
#         all_prices = final_data['Close'][available]
#     else:
#         st.error("Não foi possível baixar preços finais das ações selecionadas.")
#         st.stop()
#     # Drop any row with missing data (solo fechas donde todas existen)
#     all_prices = all_prices.dropna(axis=0, how='any')
#     return chosen, all_prices

# def find_best_corr(target_series, candidate_series, max_lag=30):
#     best_corr = -np.inf
#     best_lag = None
#     best_aligned = None
#     idx = target_series.index
#     for lag in range(-max_lag, 0):  # solo lags negativos
#         shifted = candidate_series.shift(-lag)
#         # Siempre alinear usando los índices del target
#         shifted = shifted.reindex(idx)
#         valid = target_series.notnull() & shifted.notnull()
#         if valid.sum() < 5:
#             continue
#         corr = target_series[valid].corr(shifted[valid])
#         if np.isnan(corr):
#             continue
#         if abs(corr) > abs(best_corr):
#             best_corr = corr
#             best_lag = lag
#             best_aligned = shifted
#     if best_lag is None or best_aligned is None:
#         return np.nan, np.nan, pd.Series(index=target_series.index, data=np.nan)
#     return best_corr, best_lag, best_aligned

# def show():
#     st.title("Correlações Inteligentes para Previsão de Ações")
#     st.write(
#         "Selecione uma ação de interesse e descubra, entre as demais do mercado, quais séries possuem maior poder preditivo, após alinhar possíveis atrasos temporais. Visualize graficamente a diferença antes e depois do alinhamento e as correlações individuais."
#     )
#     num_stocks = st.slider("Número de ações para explorar", 10, 50, 30)
#     dias_historico = st.slider("Período de dados históricos (dias)", 30, 1000, 750)
#     max_lag = 30
#     tickers, all_prices = get_valid_random_stocks(num_stocks, days=dias_historico, max_lag=max_lag)
#     # Ventana de fechas (últimos N días con datos para todos)
#     final_index = all_prices.index[-dias_historico:]
#     data = all_prices.loc[final_index]
#     st.write(f"Ações válidas selecionadas ({len(tickers)}):", ", ".join(tickers))
#     st.dataframe(data)

#     action_selected = st.selectbox("Selecione a ação de interesse (target)", options=tickers)
#     target_series = data[action_selected]
#     data_no_target = data.drop(columns=[action_selected])

#     st.markdown("#### Exploração das 7 ações mais correlacionadas (com lag)")
#     results = []
#     for ticker in data_no_target.columns:
#         orig_candidate = all_prices[ticker]
#         # Correlación sin desfase (última ventana)
#         candidate_in_window = orig_candidate.reindex(final_index)
#         corr_before = target_series.corr(candidate_in_window)
#         # Mejor lag alineado (compara solo en la ventana final)
#         corr_after, lag, aligned = find_best_corr(target_series, orig_candidate, max_lag=max_lag)
#         results.append({
#             "ticker": ticker,
#             "corr_before": corr_before,
#             "corr_after": corr_after,
#             "lag": lag,
#             "aligned": aligned
#         })

#     top_results = sorted(
#         [res for res in results if not np.isnan(res['corr_after'])],
#         key=lambda x: abs(x['corr_after']),
#         reverse=True
#     )[:7]
#     palette = plotly.colors.qualitative.Set2 * 2

#     # --- Gráficos Interativos ---
#     fig_before = go.Figure()
#     fig_before.add_trace(go.Scatter(x=target_series.index, y=target_series.values,
#                                     mode='lines', name=action_selected,
#                                     line=dict(color='black', width=3)))
#     for i, res in enumerate(top_results):
#         candidate_in_window = all_prices[res['ticker']].reindex(final_index)
#         fig_before.add_trace(go.Scatter(x=candidate_in_window.index, y=candidate_in_window.values,
#                                         mode='lines', name=res['ticker'],
#                                         line=dict(color=palette[i], width=1), opacity=0.7))
#     fig_before.update_layout(title="Séries antes do alinhamento (top 7 + target)",
#                             legend_title="Ações", height=400, template="plotly_white")

#     fig_after = go.Figure()
#     fig_after.add_trace(go.Scatter(x=target_series.index, y=target_series.values,
#                                    mode='lines', name=action_selected,
#                                    line=dict(color='black', width=3)))
#     for i, res in enumerate(top_results):
#         aligned = res["aligned"]
#         fig_after.add_trace(go.Scatter(x=aligned.index, y=aligned.values,
#                                        mode='lines', name=f"{res['ticker']} (lag {res['lag']})",
#                                        line=dict(color=palette[i], width=1), opacity=0.7))
#     fig_after.update_layout(title="Séries após alinhamento ótimo (top 7 + target)",
#                            legend_title="Ações", height=400, template="plotly_white")

#     col1, col2 = st.columns(2)
#     with col1:
#         st.plotly_chart(fig_before, use_container_width=True)
#     with col2:
#         st.plotly_chart(fig_after, use_container_width=True)

#     st.markdown("### Tabela das melhores ações para previsão (top 7)")
#     resumo = pd.DataFrame({
#         "Ticker": [x["ticker"] for x in top_results],
#         "Correl. Antes": [x["corr_before"] for x in top_results],
#         "Correl. Após Alinhamento": [x["corr_after"] for x in top_results],
#         "Lag ótimo": [x["lag"] for x in top_results]
#     })
#     st.dataframe(resumo.style.format({'Correl. Antes': "{:.2f}", 'Correl. Após Alinhamento': "{:.2f}", 'Lag ótimo': "{}"}))

# if __name__ == "__main__":
#     show()






# # # menu/exploracao.py
# import streamlit as st
# import pandas as pd
# import numpy as np
# import yfinance as yf
# import random
# import plotly.graph_objs as go
# import plotly.colors

# TICKERS_LIST = [
#     "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA", "BRK-B", "JPM", "V",
#     "UNH", "JNJ", "XOM", "PG", "MA", "HD", "CVX", "MRK", "ABBV", "PEP",
#     "LLY", "KO", "AVGO", "COST", "WMT", "DIS", "BAC", "MCD", "ADBE", "PFE",
#     "TMO", "CSCO", "DHR", "ABT", "ACN", "CRM", "TXN", "NKE", "VZ", "NEE",
#     "CMCSA", "WFC", "LIN", "INTC", "QCOM", "MS", "HON", "PM", "UNP", "AMGN",
#     "SBUX", "ORCL", "AMD", "CVS", "ISRG", "GILD", "BKNG", "NOW", "ZTS", "MDLZ"
# ]

# def get_valid_random_stocks(n=50, days=30, max_attempts=5):
#     tried = set()
#     valid_tickers = []
#     attempt = 0
#     while len(valid_tickers) < n and attempt < max_attempts:
#         tickers_to_try = list(set(TICKERS_LIST) - tried)
#         if len(tickers_to_try) == 0:
#             break
#         sample_size = min(n - len(valid_tickers) + 10, len(tickers_to_try))
#         tickers_sample = random.sample(tickers_to_try, sample_size)
#         tried.update(tickers_sample)
#         data = yf.download(tickers_sample, period=f'{days+10}d', interval='1d', progress=False)
#         if 'Adj Close' in data:
#             prices = data['Adj Close'].dropna(axis=1, how='any')
#         elif 'Close' in data:
#             prices = data['Close'].dropna(axis=1, how='any')
#         else:
#             prices = pd.DataFrame()
#         found = [col for col in prices.columns if col not in valid_tickers]
#         valid_tickers.extend(found)
#         attempt += 1
#     if len(valid_tickers) == 0:
#         st.error("Não foi possível encontrar ações com dados válidos. Verifique sua conexão, tente novamente ou diminua o número de ações.")
#         st.stop()
#     if len(valid_tickers) < n:
#         st.warning(f"Apenas {len(valid_tickers)} ações com dados válidos foram encontradas. Usando todas.")
#         n = len(valid_tickers)
#     chosen = valid_tickers[:n]
#     final_data = yf.download(chosen, period=f'{days+10}d', interval='1d', progress=False)
#     if 'Adj Close' in final_data:
#         available = [c for c in chosen if c in final_data['Adj Close'].columns]
#         final_prices = final_data['Adj Close'][available].dropna(axis=0, how='any')[-days:]
#     elif 'Close' in final_data:
#         available = [c for c in chosen if c in final_data['Close'].columns]
#         final_prices = final_data['Close'][available].dropna(axis=0, how='any')[-days:]
#     else:
#         st.error("Não foi possível baixar preços finais das ações selecionadas.")
#         st.stop()
#     return chosen, final_prices

# def find_best_corr(target_series, candidate_series, max_lag=30):
#     best_corr = -np.inf
#     best_lag = None
#     best_aligned = None
#     # Solo lags negativos (lag < 0) para predicción futura
#     for lag in range(-max_lag, 0):
#         shifted = candidate_series.shift(-lag)
#         aligned = pd.concat([target_series, shifted], axis=1).dropna()
#         if len(aligned) < 5:
#             continue
#         corr = aligned.iloc[:, 0].corr(aligned.iloc[:, 1])
#         if np.isnan(corr):
#             continue
#         if abs(corr) > abs(best_corr):
#             best_corr = corr
#             best_lag = lag
#             best_aligned = shifted
#     if best_lag is None or best_aligned is None:
#         return np.nan, np.nan, pd.Series(index=target_series.index, data=np.nan)
#     return best_corr, best_lag, best_aligned

# def show():
#     st.title("Correlações Inteligentes para Previsão de Ações")
#     st.write(
#         "Selecione uma ação de interesse e descubra, entre as demais do mercado, quais séries possuem maior poder preditivo, após alinhar possíveis atrasos temporais. Visualize graficamente a diferença antes e depois do alinhamento e as correlações individuais."
#     )
#     num_stocks = st.slider("Número de ações para explorar", 10, 50, 30)
#     dias_historico = st.slider("Período de dados históricos (dias)", 30, 1000, 750)
#     tickers, data = get_valid_random_stocks(num_stocks, days=dias_historico)
#     st.write(f"Ações válidas selecionadas ({len(tickers)}):", ", ".join(tickers))
#     st.dataframe(data)

#     # Selector da ação de interesse
#     action_selected = st.selectbox("Selecione a ação de interesse (target)", options=tickers)
#     data_no_target = data.drop(columns=[action_selected])
#     target_series = data[action_selected]

#     st.markdown("#### Exploração das 7 ações mais correlacionadas (com lag)")
#     results = []
#     for ticker in data_no_target.columns:
#         candidate_series = data_no_target[ticker]
#         corr_before = target_series.corr(candidate_series)
#         corr_after, lag, aligned = find_best_corr(target_series, candidate_series, max_lag=30)
#         results.append({
#             "ticker": ticker,
#             "corr_before": corr_before,
#             "corr_after": corr_after,
#             "lag": lag,
#             "aligned": aligned
#         })

#     # Top 7 melhores, excluindo correlaciones NaN
#     top_results = sorted(
#         [res for res in results if not np.isnan(res['corr_after'])],
#         key=lambda x: abs(x['corr_after']),
#         reverse=True
#     )[:7]
#     top_tickers = [res['ticker'] for res in top_results]

#     # Paleta de cores para Plotly (distintas das 7 séries)
#     palette = plotly.colors.qualitative.Set2 * 2

#     # --- Gráfico Interativo Antes do alinhamento ---
#     fig_before = go.Figure()
#     fig_before.add_trace(go.Scatter(x=target_series.index, y=target_series.values,
#                                     mode='lines', name=action_selected,
#                                     line=dict(color='black', width=3)))
#     for i, res in enumerate(top_results):
#         serie = data_no_target[res['ticker']]
#         fig_before.add_trace(go.Scatter(x=serie.index, y=serie.values,
#                                         mode='lines', name=res['ticker'],
#                                         line=dict(color=palette[i], width=1), opacity=0.7))
#     fig_before.update_layout(title="Séries antes do alinhamento (top 7 + target)",
#                             legend_title="Ações", height=400, template="plotly_white")

#     # --- Gráfico Interativo Depois do alinhamento ---
#     fig_after = go.Figure()
#     fig_after.add_trace(go.Scatter(x=target_series.index, y=target_series.values,
#                                    mode='lines', name=action_selected,
#                                    line=dict(color='black', width=3)))
#     for i, res in enumerate(top_results):
#         aligned = res["aligned"]
#         fig_after.add_trace(go.Scatter(x=aligned.index, y=aligned.values,
#                                        mode='lines', name=f"{res['ticker']} (lag {res['lag']})",
#                                        line=dict(color=palette[i], width=1), opacity=0.7))
#     fig_after.update_layout(title="Séries após alinhamento ótimo (top 7 + target)",
#                            legend_title="Ações", height=400, template="plotly_white")

#     # Mostrar gráficos em duas colunas
#     col1, col2 = st.columns(2)
#     with col1:
#         st.plotly_chart(fig_before, use_container_width=True)
#     with col2:
#         st.plotly_chart(fig_after, use_container_width=True)

#     # Tabela resumo para investidores
#     st.markdown("### Tabela das melhores ações para previsão (top 7)")
#     resumo = pd.DataFrame({
#         "Ticker": [x["ticker"] for x in top_results],
#         "Correl. Antes": [x["corr_before"] for x in top_results],
#         "Correl. Após Alinhamento": [x["corr_after"] for x in top_results],
#         "Lag ótimo": [x["lag"] for x in top_results]
#     })
#     st.dataframe(resumo.style.format({'Correl. Antes': "{:.2f}", 'Correl. Após Alinhamento': "{:.2f}", 'Lag ótimo': "{}"}))

# if __name__ == "__main__":
#     show()









# menu/exploracao.py
# # menu/exploracao.py
# import streamlit as st
# import pandas as pd
# import numpy as np
# import yfinance as yf
# import random
# import plotly.graph_objs as go
# import plotly.colors

# TICKERS_LIST = [
#     "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA", "BRK-B", "JPM", "V",
#     "UNH", "JNJ", "XOM", "PG", "MA", "HD", "CVX", "MRK", "ABBV", "PEP",
#     "LLY", "KO", "AVGO", "COST", "WMT", "DIS", "BAC", "MCD", "ADBE", "PFE",
#     "TMO", "CSCO", "DHR", "ABT", "ACN", "CRM", "TXN", "NKE", "VZ", "NEE",
#     "CMCSA", "WFC", "LIN", "INTC", "QCOM", "MS", "HON", "PM", "UNP", "AMGN",
#     "SBUX", "ORCL", "AMD", "CVS", "ISRG", "GILD", "BKNG", "NOW", "ZTS", "MDLZ"
# ]

# def get_valid_random_stocks(n=50, days=30, max_lag=30, max_attempts=5):
#     tried = set()
#     valid_tickers = []
#     attempt = 0
#     while len(valid_tickers) < n and attempt < max_attempts:
#         tickers_to_try = list(set(TICKERS_LIST) - tried)
#         if not tickers_to_try:
#             break
#         sample_size = min(n - len(valid_tickers) + 10, len(tickers_to_try))
#         tickers_sample = random.sample(tickers_to_try, sample_size)
#         tried.update(tickers_sample)
#         data = yf.download(tickers_sample, period=f'{days+max_lag+10}d', interval='1d', progress=False)
#         if 'Adj Close' in data:
#             prices = data['Adj Close']
#         elif 'Close' in data:
#             prices = data['Close']
#         else:
#             prices = pd.DataFrame()
#         filtered = [col for col in prices.columns if prices[col].dropna().shape[0] >= days + max_lag]
#         found = [col for col in filtered if col not in valid_tickers]
#         valid_tickers.extend(found)
#         attempt += 1
#     if not valid_tickers:
#         st.error("Não foi possível encontrar ações com dados válidos. Verifique sua conexão, tente novamente ou diminua o número de ações.")
#         st.stop()
#     if len(valid_tickers) < n:
#         st.warning(f"Apenas {len(valid_tickers)} ações com dados válidos foram encontradas. Usando todas.")
#         n = len(valid_tickers)
#     chosen = valid_tickers[:n]
#     final_data = yf.download(chosen, period=f'{days+max_lag+10}d', interval='1d', progress=False)
#     if 'Adj Close' in final_data:
#         available = [c for c in chosen if c in final_data['Adj Close'].columns]
#         all_prices = final_data['Adj Close'][available]
#     elif 'Close' in final_data:
#         available = [c for c in chosen if c in final_data['Close'].columns]
#         all_prices = final_data['Close'][available]
#     else:
#         st.error("Não foi possível baixar preços finais das ações selecionadas.")
#         st.stop()
#     return chosen, all_prices

# def show():
#     st.title("Correlações Inteligentes para Previsão de Ações")
#     st.write(
#         "Selecione uma ação de interesse e descubra, entre as demais do mercado, quais séries possuem maior poder preditivo, após alinhar possíveis atrasos temporais. Visualize graficamente a diferença antes e depois do alinhamento e as correlações individuais."
#     )
#     num_stocks = st.slider("Número de ações para explorar", 10, 50, 30)
#     dias_historico = st.slider("Período de dados históricos (dias)", 30, 1000, 750)
#     max_lag = 30
#     tickers, all_prices = get_valid_random_stocks(num_stocks, days=dias_historico, max_lag=max_lag)

#     # Ventana final de datas para análise (as últimas N datas com dados completos)
#     final_index = all_prices.index[-dias_historico:]
#     data = all_prices.loc[final_index]

#     st.write(f"Ações válidas selecionadas ({len(tickers)}):", ", ".join(tickers))
#     st.dataframe(data)

#     action_selected = st.selectbox("Selecione a ação de interesse (target)", options=tickers)
#     target_series = data[action_selected]
#     target_full = all_prices[action_selected]
#     data_no_target = data.drop(columns=[action_selected])

#     st.markdown("#### Exploração das 7 ações mais correlacionadas (com lag)")
#     results = []
#     diagnostico_valids = {}

#     for ticker in data_no_target.columns:
#         orig_candidate = all_prices[ticker]
#         corr_before = target_series.corr(orig_candidate.reindex(final_index))
#         best_corr = -np.inf
#         best_lag = None
#         best_aligned = None
#         max_valid = 0
#         for lag in range(-max_lag, 0):  # Solo lags negativos para prever futuro
#             shifted = orig_candidate.shift(-lag).reindex(final_index)
#             valid = target_series.notnull() & shifted.notnull()
#             n_valid = valid.sum()
#             if n_valid < 5:
#                 continue
#             corr = target_series[valid].corr(shifted[valid])
#             if np.isnan(corr):
#                 continue
#             if abs(corr) > abs(best_corr):
#                 best_corr = corr
#                 best_lag = lag
#                 best_aligned = shifted
#                 max_valid = n_valid
#         diagnostico_valids[ticker] = max_valid
#         results.append({
#             "ticker": ticker,
#             "corr_before": corr_before,
#             "corr_after": best_corr,
#             "lag": best_lag,
#             "aligned": best_aligned,
#             "valid_points": max_valid
#         })

#     st.write("Diagnóstico: quantidade de pontos válidos por ação após shift ótimo")
#     st.dataframe(pd.Series(diagnostico_valids, name='valores válidos após shift'))

#     # Solo incluimos resultados con serie alineada válida
#     top_results = [
#         res for res in results
#         if (
#             (not np.isnan(res['corr_after'])) and 
#             (res['aligned'] is not None) and 
#             (isinstance(res['aligned'], pd.Series)) and
#             (res['aligned'].notnull().sum() > 0)
#         )
#     ]
#     top_results = sorted(top_results, key=lambda x: abs(x['corr_after']), reverse=True)[:7]

#     palette = plotly.colors.qualitative.Set2 * 2

#     # --- Gráfico Interativo Antes do alinhamento ---
#     fig_before = go.Figure()
#     fig_before.add_trace(go.Scatter(x=target_series.index, y=target_series.values,
#                                     mode='lines', name=action_selected,
#                                     line=dict(color='black', width=3)))
#     for i, res in enumerate(top_results):
#         serie = data_no_target[res['ticker']]
#         fig_before.add_trace(go.Scatter(x=serie.index, y=serie.values,
#                                         mode='lines', name=res['ticker'],
#                                         line=dict(color=palette[i], width=1), opacity=0.7))
#     fig_before.update_layout(title="Séries antes do alinhamento (top 7 + target)",
#                             legend_title="Ações", height=400, template="plotly_white")

#     # --- Gráfico Interativo Depois do alinhamento ---
#     fig_after = go.Figure()
#     fig_after.add_trace(go.Scatter(x=target_series.index, y=target_series.values,
#                                    mode='lines', name=action_selected,
#                                    line=dict(color='black', width=3)))
#     for i, res in enumerate(top_results):
#         aligned = res["aligned"]
#         if isinstance(aligned, pd.Series) and aligned.notnull().sum() > 0:
#             fig_after.add_trace(go.Scatter(x=aligned.index, y=aligned.values,
#                                            mode='lines', name=f"{res['ticker']} (lag {res['lag']})",
#                                            line=dict(color=palette[i], width=1), opacity=0.7))
#     fig_after.update_layout(title="Séries após alinhamento ótimo (top 7 + target)",
#                            legend_title="Ações", height=400, template="plotly_white")

#     # Mostrar gráficos em duas colunas
#     col1, col2 = st.columns(2)
#     with col1:
#         st.plotly_chart(fig_before, use_container_width=True)
#     with col2:
#         st.plotly_chart(fig_after, use_container_width=True)

#     # Tabela resumo para investidores
#     st.markdown("### Tabela das melhores ações para previsão (top 7)")
#     resumo = pd.DataFrame({
#         "Ticker": [x["ticker"] for x in top_results],
#         "Correl. Antes": [x["corr_before"] for x in top_results],
#         "Correl. Após Alinhamento": [x["corr_after"] for x in top_results],
#         "Lag ótimo": [x["lag"] for x in top_results],
#         "Pontos válidos": [x["valid_points"] for x in top_results]
#     })
#     st.dataframe(resumo.style.format({'Correl. Antes': "{:.2f}", 'Correl. Após Alinhamento': "{:.2f}", 'Lag ótimo': "{}", 'Pontos válidos': "{}"}))

# if __name__ == "__main__":
#     show()




# import streamlit as st
# import pandas as pd
# import numpy as np
# import yfinance as yf
# import random
# import plotly.graph_objs as go
# import plotly.colors

# TICKERS_LIST = [
#     "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA", "BRK-B", "JPM", "V",
#     "UNH", "JNJ", "XOM", "PG", "MA", "HD", "CVX", "MRK", "ABBV", "PEP",
#     "LLY", "KO", "AVGO", "COST", "WMT", "DIS", "BAC", "MCD", "ADBE", "PFE",
#     "TMO", "CSCO", "DHR", "ABT", "ACN", "CRM", "TXN", "NKE", "VZ", "NEE",
#     "CMCSA", "WFC", "LIN", "INTC", "QCOM", "MS", "HON", "PM", "UNP", "AMGN",
#     "SBUX", "ORCL", "AMD", "CVS", "ISRG", "GILD", "BKNG", "NOW", "ZTS", "MDLZ"
# ]

# def download_stocks(tickers, days, max_lag):
#     # Descargar un periodo mayor para cubrir todos los lags posibles
#     period = days + max_lag + 10
#     df = yf.download(tickers, period=f"{period}d", interval="1d", progress=False)
#     if 'Adj Close' in df:
#         data = df['Adj Close']
#     elif 'Close' in df:
#         data = df['Close']
#     else:
#         data = pd.DataFrame()
#     return data

# def prepare_data(n=20, days=750, max_lag=25):
#     # Selecciona n tickers al azar
#     tickers = random.sample(TICKERS_LIST, n)
#     data = download_stocks(tickers, days, max_lag)
#     # Elimina columnas (acciones) con NaN en la ventana final requerida
#     min_len = days + max_lag
#     cols_validas = []
#     for col in data.columns:
#         serie = data[col].dropna()
#         if len(serie) < min_len:
#             continue
#         ultimos = serie.iloc[-min_len:]
#         if ultimos.isnull().any():
#             continue
#         cols_validas.append(col)
#     if not cols_validas or len(cols_validas) < 2:
#         st.error("Poucas ações com dados completos para análise. Tente diminuir dias históricos ou número de ações.")
#         st.stop()
#     # Genera dataframe final SOLO con acciones y filas completas
#     data_valid = data[cols_validas].iloc[-min_len:]
#     return data_valid

# def show():
#     st.title("Correlações Inteligentes para Previsão de Ações (Eficiência Máxima)")
#     st.write(
#         "Explora apenas ações com dados 100% completos na janela analisada, com seleção flexível da ação de interesse e comparação dos melhores pares correlacionados após alinhamento de lags (por posição)."
#     )
#     num_stocks = st.slider("Número de ações para explorar", 10, 50, 20)
#     dias_historico = st.slider("Período de dados históricos (dias)", 30, 1000, 750)
#     max_lag = 25

#     data = prepare_data(num_stocks, dias_historico, max_lag)
#     st.write(f"Ações disponíveis (com dados completos):", ", ".join(data.columns))
#     st.dataframe(data)

#     action_selected = st.selectbox("Selecione a ação de interesse (target)", options=data.columns)
#     idx_target = data.columns.get_loc(action_selected)
#     target = data.iloc[-dias_historico:, idx_target].values

#     results = []
#     palette = plotly.colors.qualitative.Set2 * 2

#     for ticker in data.columns:
#         if ticker == action_selected:
#             continue
#         series = data[ticker].values
#         max_corr = -np.inf
#         best_lag = None
#         best_slice = None
#         for lag in range(1, max_lag+1):
#             if len(series) < dias_historico + lag:
#                 continue
#             shifted = series[-dias_historico - lag:-lag]
#             if len(shifted) != dias_historico:
#                 continue
#             corr = np.corrcoef(target, shifted)[0,1]
#             if np.isnan(corr):
#                 continue
#             if abs(corr) > abs(max_corr):
#                 max_corr = corr
#                 best_lag = lag
#                 best_slice = shifted
#         if len(series) >= dias_historico:
#             corr_before = np.corrcoef(target, series[-dias_historico:])[0,1]
#         else:
#             corr_before = np.nan
#         results.append({
#             "ticker": ticker,
#             "corr_before": corr_before,
#             "corr_after": max_corr,
#             "lag": best_lag,
#             "aligned": best_slice,
#             "original": series[-dias_historico:]
#         })

#     # Top 7 melhores
#     top_results = sorted(
#         [res for res in results if not np.isnan(res['corr_after']) and res['lag'] is not None],
#         key=lambda x: abs(x['corr_after']),
#         reverse=True
#     )[:7]

#     # --- Gráficos Plotly ---
#     fig_before = go.Figure()
#     fig_before.add_trace(go.Scatter(
#         x=np.arange(dias_historico), y=target,
#         mode='lines', name=action_selected,
#         line=dict(color='black', width=3)
#     ))
#     for i, res in enumerate(top_results):
#         fig_before.add_trace(go.Scatter(
#             x=np.arange(dias_historico), y=res["original"],
#             mode='lines', name=res["ticker"],
#             line=dict(color=palette[i], width=1), opacity=0.7
#         ))
#     fig_before.update_layout(title="Séries antes do alinhamento (top 7 + target)",
#                             legend_title="Ações", height=400, template="plotly_white")

#     fig_after = go.Figure()
#     fig_after.add_trace(go.Scatter(
#         x=np.arange(dias_historico), y=target,
#         mode='lines', name=action_selected,
#         line=dict(color='black', width=3)
#     ))
#     for i, res in enumerate(top_results):
#         if res["aligned"] is not None:
#             fig_after.add_trace(go.Scatter(
#                 x=np.arange(dias_historico), y=res["aligned"],
#                 mode='lines', name=f"{res['ticker']} (lag {res['lag']})",
#                 line=dict(color=palette[i], width=1), opacity=0.7
#             ))
#     fig_after.update_layout(title="Séries após alinhamento ótimo (top 7 + target)",
#                            legend_title="Ações", height=400, template="plotly_white")

#     col1, col2 = st.columns(2)
#     with col1:
#         st.plotly_chart(fig_before, use_container_width=True)
#     with col2:
#         st.plotly_chart(fig_after, use_container_width=True)

#     st.markdown("### Tabela das melhores ações para previsão (top 7)")
#     resumo = pd.DataFrame({
#         "Ticker": [x["ticker"] for x in top_results],
#         "Correl. Antes": [x["corr_before"] for x in top_results],
#         "Correl. Após Alinhamento": [x["corr_after"] for x in top_results],
#         "Lag ótimo": [x["lag"] for x in top_results]
#     })
#     st.dataframe(resumo.style.format({'Correl. Antes': "{:.2f}", 'Correl. Após Alinhamento': "{:.2f}", 'Lag ótimo': "{}"}))

# if __name__ == "__main__":
#     show()







import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import random
import plotly.graph_objs as go

TICKERS_LIST = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA", "BRK-B", "JPM", "V",
    "UNH", "JNJ", "XOM", "PG", "MA", "HD", "CVX", "MRK", "ABBV", "PEP",
    "LLY", "KO", "AVGO", "COST", "WMT", "DIS", "BAC", "MCD", "ADBE", "PFE",
    "TMO", "CSCO", "DHR", "ABT", "ACN", "CRM", "TXN", "NKE", "VZ", "NEE",
    "CMCSA", "WFC", "LIN", "INTC", "QCOM", "MS", "HON", "PM", "UNP", "AMGN",
    "SBUX", "ORCL", "AMD", "CVS", "ISRG", "GILD", "BKNG", "NOW", "ZTS", "MDLZ"
]

@st.cache_data(show_spinner=True)
def download_stocks(tickers, days, max_lag):
    period = days + max_lag + 10
    df = yf.download(tickers, period=f"{period}d", interval="1d", progress=False)
    data = df['Adj Close'] if 'Adj Close' in df else df['Close']
    return data

def get_valid_data(n=20, days=750, max_lag=30):
    tickers = random.sample(TICKERS_LIST, n)
    data = download_stocks(tickers, days, max_lag)
    min_len = days + max_lag
    valid_cols = [c for c in data.columns if data[c].dropna().shape[0] >= min_len]
    # Filtrar sólo filas completas al final
    data_valid = data[valid_cols].iloc[-min_len:]
    data_valid = data_valid.dropna(axis=1)
    return data_valid

def calc_best_lag(target, candidate, dias, max_lag=30):
    """
    Retorna la mejor correlación (en valor absoluto) y el lag asociado
    """
    best_corr = None
    best_lag = None
    best_slice = None
    for lag in range(1, max_lag + 1):
        shifted = candidate[-dias - lag : -lag]
        if len(shifted) != dias:
            continue
        corr = np.corrcoef(target, shifted)[0, 1]
        if np.isnan(corr):
            continue
        if (best_corr is None) or (abs(corr) > abs(best_corr)):
            best_corr = corr
            best_lag = lag
            best_slice = shifted
    return best_corr, best_lag, best_slice

def show():
    st.title("Descubrimiento de Acciones Altamente Correlacionadas con Lag")
    st.write("Selecciona una acción objetivo y descubre las 7 que más la siguen, considerando *lags* de hasta 30 días.")

    n_stocks = st.slider("Número de acciones a analizar", 10, 50, 20)
    dias_historico = st.slider("Ventana de días para comparar", 100, 1000, 750)
    max_lag = st.slider("Máximo desfase (lag, en días)", 5, 30, 25)

    data = get_valid_data(n_stocks, dias_historico, max_lag)
    st.write("Acciones disponibles (con datos completos):", ", ".join(data.columns))
    st.dataframe(data)

    action_selected = st.selectbox("Selecciona la acción objetivo", options=data.columns)
    target = data[action_selected].iloc[-dias_historico:].values

    resultados = []
    for ticker in data.columns:
        if ticker == action_selected:
            continue
        serie = data[ticker].values
        # correlación original (sin desfase)
        corr_0 = np.corrcoef(target, serie[-dias_historico:])[0, 1]
        # buscar lag óptimo
        corr_best, lag_best, serie_lag = calc_best_lag(target, serie, dias_historico, max_lag)
        resultados.append({
            "ticker": ticker,
            "corr_0": corr_0,
            "corr_best": corr_best,
            "lag_best": lag_best,
            "serie_0": serie[-dias_historico:],
            "serie_lag": serie_lag
        })

    # Selecciona los 7 mejores (por valor absoluto de correlación con lag)
    top7 = sorted(
        [r for r in resultados if r["corr_best"] is not None],
        key=lambda r: abs(r["corr_best"]),
        reverse=True
    )[:7]

    # Gráfico original
    fig_ori = go.Figure()
    fig_ori.add_trace(go.Scatter(
        x=np.arange(dias_historico), y=target,
        mode='lines', name=action_selected, line=dict(color='black', width=3)
    ))
    for i, r in enumerate(top7):
        fig_ori.add_trace(go.Scatter(
            x=np.arange(dias_historico), y=r["serie_0"],
            mode='lines', name=r["ticker"], opacity=0.6
        ))
    fig_ori.update_layout(title="Series originales (sin desfase)", height=400)

    # Gráfico alineadas por lag
    fig_lag = go.Figure()
    fig_lag.add_trace(go.Scatter(
        x=np.arange(dias_historico), y=target,
        mode='lines', name=action_selected, line=dict(color='black', width=3)
    ))
    for i, r in enumerate(top7):
        fig_lag.add_trace(go.Scatter(
            x=np.arange(dias_historico), y=r["serie_lag"],
            mode='lines', name=f"{r['ticker']} (lag {r['lag_best']})", opacity=0.7
        ))
    fig_lag.update_layout(title="Series alineadas por lag óptimo", height=400)

    col1, col2 = st.columns(2)
    with col1:
        st.plotly_chart(fig_ori, use_container_width=True)
    with col2:
        st.plotly_chart(fig_lag, use_container_width=True)

    st.markdown("### Tabla de los 7 mejores emparejamientos (por correlación absoluta tras desfase)")
    tabla = pd.DataFrame({
        "Ticker": [x["ticker"] for x in top7],
        "Correlación sin desfase": [x["corr_0"] for x in top7],
        "Correlación máxima (con lag)": [x["corr_best"] for x in top7],
        "Lag óptimo (días)": [x["lag_best"] for x in top7]
    })
    st.dataframe(tabla.style.format({"Correlación sin desfase": "{:.2f}", "Correlación máxima (con lag)": "{:.2f}"}))

if __name__ == "__main__":
    main()
