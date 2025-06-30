# # #menu/exploracao.py
# import streamlit as st
# import pandas as pd
# import numpy as np
# import yfinance as yf
# import random
# import plotly.graph_objs as go
# from sklearn.ensemble import RandomForestRegressor
# from sklearn.metrics import mean_squared_error, r2_score

# TICKERS_LIST = [
#     "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA", "BRK-B", "JPM", "V",
#     "UNH", "JNJ", "XOM", "PG", "MA", "HD", "CVX", "MRK", "ABBV", "PEP",
#     "LLY", "KO", "AVGO", "COST", "WMT", "DIS", "BAC", "MCD", "ADBE", "PFE",
#     "TMO", "CSCO", "DHR", "ABT", "ACN", "CRM", "TXN", "NKE", "VZ", "NEE",
#     "CMCSA", "WFC", "LIN", "INTC", "QCOM", "MS", "HON", "PM", "UNP", "AMGN",
#     "SBUX", "ORCL", "AMD", "CVS", "ISRG", "GILD", "BKNG", "NOW", "ZTS", "MDLZ"
# ]

# # --- Mantener tickers seleccionados y target en session_state ---
# def init_session_state(n_stocks, seed):
#     if 'tickers' not in st.session_state or st.session_state['tickers_seed'] != seed or st.session_state['tickers_n'] != n_stocks:
#         random.seed(seed)
#         st.session_state['tickers'] = random.sample(TICKERS_LIST, n_stocks)
#         st.session_state['tickers_seed'] = seed
#         st.session_state['tickers_n'] = n_stocks

# @st.cache_data(show_spinner=True)
# def download_stocks(tickers, days, max_lag):
#     period = days + max_lag + 40  # +40 para asegurar ventana
#     df = yf.download(tickers, period=f"{period}d", interval="1d", progress=False)
#     data = df['Adj Close'] if 'Adj Close' in df else df['Close']
#     return data

# def get_valid_data(tickers, days=750, max_lag=30):
#     data = download_stocks(tickers, days, max_lag)
#     min_len = days + max_lag + 30  # +30 por ventana temporal
#     valid_cols = [c for c in data.columns if data[c].dropna().shape[0] >= min_len]
#     data_valid = data[valid_cols].iloc[-min_len:]
#     data_valid = data_valid.dropna(axis=1)
#     return data_valid

# def calc_best_lag(target, candidate, dias, max_lag=30):
#     best_corr = None
#     best_lag = None
#     best_slice = None
#     for lag in range(1, max_lag + 1):
#         shifted = candidate[-dias - lag : -lag]
#         if len(shifted) != dias:
#             continue
#         corr = np.corrcoef(target, shifted)[0, 1]
#         if np.isnan(corr):
#             continue
#         if (best_corr is None) or (abs(corr) > abs(best_corr)):
#             best_corr = corr
#             best_lag = lag
#             best_slice = shifted
#     return best_corr, best_lag, best_slice

# def create_sliding_window_features(target, correlated_list, window_size=30):
#     num_samples = len(target) - window_size
#     X = []
#     y = []
#     for i in range(num_samples):
#         features = []
#         features.extend(target[i:i+window_size])
#         for corr in correlated_list:
#             features.extend(corr[i:i+window_size])
#         X.append(features)
#         y.append(target[i+window_size])
#     return np.array(X), np.array(y)

# def show():
#     st.title("Correlación y Previsión de Series Temporales con Sliding Window")
#     st.write("Previsión basada en ventana de los últimos 30 días de la acción objetivo y de las 7 acciones más correlacionadas (alineadas).")
#     seed = st.number_input("Semilla para selección de acciones (fijo)", value=42, min_value=0, max_value=9999, step=1)
#     n_stocks = st.slider("Número de acciones a analizar", 10, 50, 20)
#     dias_historico = st.slider("Ventana de días para comparar", 100, 1000, 750)
#     max_lag = st.slider("Máximo desfase (lag, en días)", 5, 30, 25)

#     # Inicializar tickers únicos para la sesión
#     init_session_state(n_stocks, seed)
#     tickers = st.session_state['tickers']

#     data = get_valid_data(tickers, dias_historico, max_lag)
#     st.write("Acciones disponibles (con datos completos):", ", ".join(data.columns))
#     st.dataframe(data)

#     # Mantener el target fijo si posible
#     if 'target_selected' not in st.session_state or st.session_state['target_selected'] not in data.columns:
#         st.session_state['target_selected'] = data.columns[0]
#     action_selected = st.selectbox("Selecciona la acción objetivo", options=data.columns, index=list(data.columns).index(st.session_state['target_selected']))
#     st.session_state['target_selected'] = action_selected

#     target = data[action_selected].iloc[-dias_historico:].values

#     resultados = []
#     for ticker in data.columns:
#         if ticker == action_selected:
#             continue
#         serie = data[ticker].values
#         corr_0 = np.corrcoef(target, serie[-dias_historico:])[0, 1]
#         corr_best, lag_best, serie_lag = calc_best_lag(target, serie, dias_historico, max_lag)
#         resultados.append({
#             "ticker": ticker,
#             "corr_0": corr_0,
#             "corr_best": corr_best,
#             "lag_best": lag_best,
#             "serie_0": serie[-dias_historico:],
#             "serie_lag": serie_lag
#         })

#     # Selecciona los 7 mejores
#     top7 = sorted(
#         [r for r in resultados if r["corr_best"] is not None],
#         key=lambda r: abs(r["corr_best"]),
#         reverse=True
#     )[:7]

#     # Gráficos antes y después del lag
#     fig_ori = go.Figure()
#     fig_ori.add_trace(go.Scatter(
#         x=np.arange(dias_historico), y=target,
#         mode='lines', name=action_selected, line=dict(color='black', width=3)
#     ))
#     for i, r in enumerate(top7):
#         fig_ori.add_trace(go.Scatter(
#             x=np.arange(dias_historico), y=r["serie_0"],
#             mode='lines', name=r["ticker"], opacity=0.6
#         ))
#     fig_ori.update_layout(title="Series originales (sin desfase)", height=400)

#     fig_lag = go.Figure()
#     fig_lag.add_trace(go.Scatter(
#         x=np.arange(dias_historico), y=target,
#         mode='lines', name=action_selected, line=dict(color='black', width=3)
#     ))
#     for i, r in enumerate(top7):
#         fig_lag.add_trace(go.Scatter(
#             x=np.arange(dias_historico), y=r["serie_lag"],
#             mode='lines', name=f"{r['ticker']} (lag {r['lag_best']})", opacity=0.7
#         ))
#     fig_lag.update_layout(title="Series alineadas por lag óptimo", height=400)

#     col1, col2 = st.columns(2)
#     with col1:
#         st.plotly_chart(fig_ori, use_container_width=True)
#     with col2:
#         st.plotly_chart(fig_lag, use_container_width=True)

#     st.markdown("### Tabla de los 7 mejores emparejamientos (por correlación absoluta tras desfase)")
#     tabla = pd.DataFrame({
#         "Ticker": [x["ticker"] for x in top7],
#         "Correlación sin desfase": [x["corr_0"] for x in top7],
#         "Correlación máxima (con lag)": [x["corr_best"] for x in top7],
#         "Lag óptimo (días)": [x["lag_best"] for x in top7]
#     })
#     st.dataframe(tabla.style.format({"Correlación sin desfase": "{:.2f}", "Correlación máxima (con lag)": "{:.2f}"}))

#     # --- PREVISIÓN RANDOM FOREST ---
#     st.markdown("## Previsión de la Acción Objetivo usando Random Forest y ventana de 30 días")
#     dias_pred = st.slider("¿Cuántos días futuros predecir? (test set)", 1, 15, 7)
#     window_size = 30

#     # Construcción de las matrices bien alineadas y del mismo tamaño
#     target_full = data[action_selected].iloc[-dias_historico - window_size:].values
#     min_len = len(target_full)
#     correlated_full = []
#     for r in top7:
#         serie = r["serie_lag"]
#         if len(serie) > min_len:
#             serie = serie[-min_len:]
#         elif len(serie) < min_len:
#             serie = np.concatenate([np.full(min_len - len(serie), np.nan), serie])
#         correlated_full.append(serie)

#     # Elimina cualquier fila donde haya NaN en alguna serie
#     mask = ~np.isnan(target_full)
#     for serie in correlated_full:
#         mask &= ~np.isnan(serie)
#     target_full = target_full[mask]
#     correlated_full = [serie[mask] for serie in correlated_full]

#     # Construir X, y usando ventana de 30 días para todo el período
#     X_all, y_all = create_sliding_window_features(target_full, correlated_full, window_size=window_size)
#     total_samples = len(y_all)
#     dias_graf = dias_pred + 15

#     if total_samples < dias_pred + 30:
#         st.warning("No hay suficientes datos para la ventana seleccionada. Ajusta los parámetros.")
#         return

#     # División entrenamiento-validación-test
#     X_train = X_all[:-dias_pred]
#     y_train = y_all[:-dias_pred]
#     X_test = X_all[-dias_pred:]
#     y_test = y_all[-dias_pred:]

#     # Validación interna (con split 70/30)
#     split = int(0.7 * len(X_train))
#     X_tr, X_val = X_train[:split], X_train[split:]
#     y_tr, y_val = y_train[:split], y_train[split:]

#     rf = RandomForestRegressor(n_estimators=200, random_state=42)
#     rf.fit(X_tr, y_tr)

#     y_val_pred = rf.predict(X_val)
#     mse = mean_squared_error(y_val, y_val_pred)
#     r2_val = r2_score(y_val, y_val_pred)
#     r2_val_disp = f"{r2_val:.2f}" if r2_val >= 0 else "—"

#     y_pred = rf.predict(X_test)
#     mse_test = mean_squared_error(y_test, y_pred)
#     r2_test = r2_score(y_test, y_pred)
#     r2_test_disp = f"{r2_test:.2f}" if r2_test >= 0 else "—"

#     # Para graficar: últimos (dias_pred + 15) días reales y previstos
#     real_graf = y_all[-dias_graf:]
#     pred_graf = np.concatenate([np.full(dias_graf - dias_pred, np.nan), y_pred]) if dias_pred < dias_graf else y_pred[-dias_graf:]

#     fig_pred = go.Figure()
#     fig_pred.add_trace(go.Scatter(
#         x=np.arange(len(real_graf)), y=real_graf,
#         mode='lines+markers', name="Real", line=dict(color='blue')
#     ))
#     fig_pred.add_trace(go.Scatter(
#         x=np.arange(len(pred_graf)), y=pred_graf,
#         mode='lines+markers', name="Previsto RF", line=dict(color='orange')
#     ))
#     fig_pred.update_layout(title=f"Previsión Random Forest ({dias_pred} días + 15 días extra, ventana=30)",
#                           xaxis_title="Día",
#                           yaxis_title="Precio",
#                           height=450)

#     st.markdown(
#         f"""**MSE validación:** {mse:.2f} | **R² validación:** {r2_val_disp}  
# **MSE test:** {mse_test:.2f} | **R² test:** {r2_test_disp}"""
#     )
#     st.plotly_chart(fig_pred, use_container_width=True)
#     st.markdown(
#         f"""
#         - **Azul:** valores reales de la acción objetivo.
#         - **Naranja:** previsión de Random Forest para los {dias_pred} días seleccionados.
#         - Se muestran también los 15 días anteriores para contexto.
#         - El modelo utiliza la ventana de los últimos 30 días de la acción objetivo y las 7 correlacionadas, alineadas.
#         """
#     )

# if __name__ == "__main__":
#     show()








# import streamlit as st
# import pandas as pd
# import numpy as np
# import yfinance as yf
# import random
# import plotly.graph_objs as go
# from sklearn.ensemble import RandomForestRegressor
# from sklearn.metrics import mean_squared_error, r2_score

# TICKERS_LIST = [
#     "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA", "BRK-B", "JPM", "V",
#     "UNH", "JNJ", "XOM", "PG", "MA", "HD", "CVX", "MRK", "ABBV", "PEP",
#     "LLY", "KO", "AVGO", "COST", "WMT", "DIS", "BAC", "MCD", "ADBE", "PFE",
#     "TMO", "CSCO", "DHR", "ABT", "ACN", "CRM", "TXN", "NKE", "VZ", "NEE",
#     "CMCSA", "WFC", "LIN", "INTC", "QCOM", "MS", "HON", "PM", "UNP", "AMGN",
#     "SBUX", "ORCL", "AMD", "CVS", "ISRG", "GILD", "BKNG", "NOW", "ZTS", "MDLZ"
# ]

# def init_session_state(n_stocks, seed):
#     if 'tickers' not in st.session_state or st.session_state['tickers_seed'] != seed or st.session_state['tickers_n'] != n_stocks:
#         random.seed(seed)
#         st.session_state['tickers'] = random.sample(TICKERS_LIST, n_stocks)
#         st.session_state['tickers_seed'] = seed
#         st.session_state['tickers_n'] = n_stocks

# @st.cache_data(show_spinner=True)
# def download_stocks(tickers, days, max_lag):
#     period = days + max_lag + 40
#     df = yf.download(tickers, period=f"{period}d", interval="1d", progress=False)
#     data = df['Adj Close'] if 'Adj Close' in df else df['Close']
#     return data

# def get_valid_data(tickers, days=750, max_lag=30):
#     data = download_stocks(tickers, days, max_lag)
#     min_len = days + max_lag + 30
#     valid_cols = [c for c in data.columns if data[c].dropna().shape[0] >= min_len]
#     data_valid = data[valid_cols].iloc[-min_len:]
#     data_valid = data_valid.dropna(axis=1)
#     return data_valid

# def calc_best_lag(target, candidate, dias, max_lag=30):
#     best_corr = None
#     best_lag = None
#     best_slice = None
#     for lag in range(1, max_lag + 1):
#         shifted = candidate[-dias - lag : -lag]
#         if len(shifted) != dias:
#             continue
#         corr = np.corrcoef(target, shifted)[0, 1]
#         if np.isnan(corr):
#             continue
#         if (best_corr is None) or (abs(corr) > abs(best_corr)):
#             best_corr = corr
#             best_lag = lag
#             best_slice = shifted
#     return best_corr, best_lag, best_slice

# def create_sliding_window_features(target, correlated_list, window_size=30):
#     num_samples = len(target) - window_size
#     X = []
#     y = []
#     for i in range(num_samples):
#         features = []
#         features.extend(target[i:i+window_size])
#         for corr in correlated_list:
#             features.extend(corr[i:i+window_size])
#         X.append(features)
#         y.append(target[i+window_size])
#     return np.array(X), np.array(y)

# def show():
#     st.title("Correlación y Previsión de Series Temporales con Sliding Window")
#     st.write("Previsión basada en ventana de los últimos 30 días de la acción objetivo y de las 7 acciones más correlacionadas (alineadas).")
#     seed = st.number_input("Semilla para selección de acciones (fijo)", value=42, min_value=0, max_value=9999, step=1)
#     n_stocks = st.slider("Número de acciones a analizar", 10, 50, 20)
#     dias_historico = st.slider("Ventana de días para comparar", 100, 1000, 750)
#     max_lag = st.slider("Máximo desfase (lag, en días)", 5, 30, 25)

#     # Inicializar tickers únicos para la sesión
#     init_session_state(n_stocks, seed)
#     tickers = st.session_state['tickers']

#     data = get_valid_data(tickers, dias_historico, max_lag)
#     st.write("Acciones disponibles (con datos completos):", ", ".join(data.columns))
#     st.dataframe(data)

#     # Solo inicializa valor por defecto, no sobreescribe en cada ciclo
#     if 'target_selected' not in st.session_state or st.session_state['target_selected'] not in data.columns:
#         st.session_state['target_selected'] = data.columns[0]
#     action_selected = st.selectbox(
#         "Selecciona la acción objetivo",
#         options=data.columns,
#         index=list(data.columns).index(st.session_state['target_selected']),
#         key='selectbox_target'
#     )
#     if action_selected != st.session_state['target_selected']:
#         st.session_state['target_selected'] = action_selected

#     target = data[action_selected].iloc[-dias_historico:].values

#     resultados = []
#     for ticker in data.columns:
#         if ticker == action_selected:
#             continue
#         serie = data[ticker].values
#         corr_0 = np.corrcoef(target, serie[-dias_historico:])[0, 1]
#         corr_best, lag_best, serie_lag = calc_best_lag(target, serie, dias_historico, max_lag)
#         resultados.append({
#             "ticker": ticker,
#             "corr_0": corr_0,
#             "corr_best": corr_best,
#             "lag_best": lag_best,
#             "serie_0": serie[-dias_historico:],
#             "serie_lag": serie_lag
#         })

#     top7 = sorted(
#         [r for r in resultados if r["corr_best"] is not None],
#         key=lambda r: abs(r["corr_best"]),
#         reverse=True
#     )[:7]

#     fig_ori = go.Figure()
#     fig_ori.add_trace(go.Scatter(
#         x=np.arange(dias_historico), y=target,
#         mode='lines', name=action_selected, line=dict(color='black', width=3)
#     ))
#     for i, r in enumerate(top7):
#         fig_ori.add_trace(go.Scatter(
#             x=np.arange(dias_historico), y=r["serie_0"],
#             mode='lines', name=r["ticker"], opacity=0.6
#         ))
#     fig_ori.update_layout(title="Series originales (sin desfase)", height=400)

#     fig_lag = go.Figure()
#     fig_lag.add_trace(go.Scatter(
#         x=np.arange(dias_historico), y=target,
#         mode='lines', name=action_selected, line=dict(color='black', width=3)
#     ))
#     for i, r in enumerate(top7):
#         fig_lag.add_trace(go.Scatter(
#             x=np.arange(dias_historico), y=r["serie_lag"],
#             mode='lines', name=f"{r['ticker']} (lag {r['lag_best']})", opacity=0.7
#         ))
#     fig_lag.update_layout(title="Series alineadas por lag óptimo", height=400)

#     col1, col2 = st.columns(2)
#     with col1:
#         st.plotly_chart(fig_ori, use_container_width=True)
#     with col2:
#         st.plotly_chart(fig_lag, use_container_width=True)

#     st.markdown("### Tabla de los 7 mejores emparejamientos (por correlación absoluta tras desfase)")
#     tabla = pd.DataFrame({
#         "Ticker": [x["ticker"] for x in top7],
#         "Correlación sin desfase": [x["corr_0"] for x in top7],
#         "Correlación máxima (con lag)": [x["corr_best"] for x in top7],
#         "Lag óptimo (días)": [x["lag_best"] for x in top7]
#     })
#     st.dataframe(tabla.style.format({"Correlación sin desfase": "{:.2f}", "Correlación máxima (con lag)": "{:.2f}"}))

#     st.markdown("## Previsión de la Acción Objetivo usando Random Forest y ventana de 30 días")
#     dias_pred = st.slider("¿Cuántos días futuros predecir? (test set)", 1, 15, 7)
#     window_size = 30

#     target_full = data[action_selected].iloc[-dias_historico - window_size:].values
#     min_len = len(target_full)
#     correlated_full = []
#     for r in top7:
#         serie = r["serie_lag"]
#         if len(serie) > min_len:
#             serie = serie[-min_len:]
#         elif len(serie) < min_len:
#             serie = np.concatenate([np.full(min_len - len(serie), np.nan), serie])
#         correlated_full.append(serie)

#     mask = ~np.isnan(target_full)
#     for serie in correlated_full:
#         mask &= ~np.isnan(serie)
#     target_full = target_full[mask]
#     correlated_full = [serie[mask] for serie in correlated_full]

#     X_all, y_all = create_sliding_window_features(target_full, correlated_full, window_size=window_size)
#     total_samples = len(y_all)
#     dias_graf = dias_pred + 15

#     if total_samples < dias_pred + 30:
#         st.warning("No hay suficientes datos para la ventana seleccionada. Ajusta los parámetros.")
#         return

#     X_train = X_all[:-dias_pred]
#     y_train = y_all[:-dias_pred]
#     X_test = X_all[-dias_pred:]
#     y_test = y_all[-dias_pred:]

#     split = int(0.7 * len(X_train))
#     X_tr, X_val = X_train[:split], X_train[split:]
#     y_tr, y_val = y_train[:split], y_train[split:]

#     rf = RandomForestRegressor(n_estimators=200, random_state=42)
#     rf.fit(X_tr, y_tr)

#     y_val_pred = rf.predict(X_val)
#     mse = mean_squared_error(y_val, y_val_pred)
#     r2_val = r2_score(y_val, y_val_pred)
#     r2_val_disp = f"{r2_val:.2f}" if r2_val >= 0 else "—"

#     y_pred = rf.predict(X_test)
#     mse_test = mean_squared_error(y_test, y_pred)
#     r2_test = r2_score(y_test, y_pred)
#     r2_test_disp = f"{r2_test:.2f}" if r2_test >= 0 else "—"

#     real_graf = y_all[-dias_graf:]
#     pred_graf = np.concatenate([np.full(dias_graf - dias_pred, np.nan), y_pred]) if dias_pred < dias_graf else y_pred[-dias_graf:]

#     fig_pred = go.Figure()
#     fig_pred.add_trace(go.Scatter(
#         x=np.arange(len(real_graf)), y=real_graf,
#         mode='lines+markers', name="Real", line=dict(color='blue')
#     ))
#     fig_pred.add_trace(go.Scatter(
#         x=np.arange(len(pred_graf)), y=pred_graf,
#         mode='lines+markers', name="Previsto RF", line=dict(color='orange')
#     ))
#     fig_pred.update_layout(title=f"Previsión Random Forest ({dias_pred} días + 15 días extra, ventana=30)",
#                           xaxis_title="Día",
#                           yaxis_title="Precio",
#                           height=450)

#     st.markdown(
#         f"""**MSE validación:** {mse:.2f} | **R² validación:** {r2_val_disp}  
# **MSE test:** {mse_test:.2f} | **R² test:** {r2_test_disp}"""
#     )
#     st.plotly_chart(fig_pred, use_container_width=True)
#     st.markdown(
#         f"""
#         - **Azul:** valores reales de la acción objetivo.
#         - **Naranja:** previsión de Random Forest para los {dias_pred} días seleccionados.
#         - Se muestran también los 15 días anteriores para contexto.
#         - El modelo utiliza la ventana de los últimos 30 días de la acción objetivo y las 7 correlacionadas, alineadas.
#         """
#     )

# if __name__ == "__main__":
#     show()









# import streamlit as st
# import pandas as pd
# import numpy as np
# import yfinance as yf
# import random
# import plotly.graph_objs as go
# from sklearn.ensemble import RandomForestRegressor
# from sklearn.metrics import mean_squared_error, r2_score

# TICKERS_LIST = [
#     "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA", "BRK-B", "JPM", "V",
#     "UNH", "JNJ", "XOM", "PG", "MA", "HD", "CVX", "MRK", "ABBV", "PEP",
#     "LLY", "KO", "AVGO", "COST", "WMT", "DIS", "BAC", "MCD", "ADBE", "PFE",
#     "TMO", "CSCO", "DHR", "ABT", "ACN", "CRM", "TXN", "NKE", "VZ", "NEE",
#     "CMCSA", "WFC", "LIN", "INTC", "QCOM", "MS", "HON", "PM", "UNP", "AMGN",
#     "SBUX", "ORCL", "AMD", "CVS", "ISRG", "GILD", "BKNG", "NOW", "ZTS", "MDLZ"
# ]

# def init_session_state(n_stocks, seed):
#     if 'tickers' not in st.session_state or st.session_state['tickers_seed'] != seed or st.session_state['tickers_n'] != n_stocks:
#         random.seed(seed)
#         st.session_state['tickers'] = random.sample(TICKERS_LIST, n_stocks)
#         st.session_state['tickers_seed'] = seed
#         st.session_state['tickers_n'] = n_stocks

# @st.cache_data(show_spinner=True)
# def download_stocks(tickers, days, max_lag):
#     period = days + max_lag + 40
#     df = yf.download(tickers, period=f"{period}d", interval="1d", progress=False)
#     data = df['Adj Close'] if 'Adj Close' in df else df['Close']
#     return data

# def get_valid_data(tickers, days=750, max_lag=30):
#     data = download_stocks(tickers, days, max_lag)
#     min_len = days + max_lag + 30
#     valid_cols = [c for c in data.columns if data[c].dropna().shape[0] >= min_len]
#     data_valid = data[valid_cols].iloc[-min_len:]
#     data_valid = data_valid.dropna(axis=1)
#     return data_valid

# def calc_best_lag(target, candidate, dias, max_lag=30):
#     best_corr = None
#     best_lag = None
#     best_slice = None
#     for lag in range(1, max_lag + 1):
#         shifted = candidate[-dias - lag : -lag]
#         if len(shifted) != dias:
#             continue
#         corr = np.corrcoef(target, shifted)[0, 1]
#         if np.isnan(corr):
#             continue
#         if (best_corr is None) or (abs(corr) > abs(best_corr)):
#             best_corr = corr
#             best_lag = lag
#             best_slice = shifted
#     return best_corr, best_lag, best_slice

# def create_sliding_window_features(target, correlated_list, window_size=30):
#     num_samples = len(target) - window_size
#     X = []
#     y = []
#     for i in range(num_samples):
#         features = []
#         features.extend(target[i:i+window_size])
#         for corr in correlated_list:
#             features.extend(corr[i:i+window_size])
#         X.append(features)
#         y.append(target[i+window_size])
#     return np.array(X), np.array(y)

# def show():
#     st.title("Correlación y Previsión de Series Temporales con Sliding Window")
#     st.write("Previsión basada en ventana de los últimos 30 días de la acción objetivo y de las 7 acciones más correlacionadas (alineadas).")
#     seed = st.number_input("Semilla para selección de acciones (fijo)", value=42, min_value=0, max_value=9999, step=1)
#     n_stocks = st.slider("Número de acciones a analizar", 10, 50, 20)
#     dias_historico = st.slider("Ventana de días para comparar", 100, 1000, 750)
#     max_lag = st.slider("Máximo desfase (lag, en días)", 5, 30, 25)

#     # Inicializar tickers únicos para la sesión
#     init_session_state(n_stocks, seed)
#     tickers = st.session_state['tickers']

#     data = get_valid_data(tickers, dias_historico, max_lag)
#     st.write("Acciones disponibles (con datos completos):", ", ".join(data.columns))
#     st.dataframe(data)

#     # --- Inicializar valor por defecto SOLO al principio, nunca dentro del ciclo
#     default_target = st.session_state.get('target_selected', data.columns[0] if len(data.columns) > 0 else "")
#     action_selected = st.selectbox(
#         "Selecciona la acción objetivo",
#         options=data.columns,
#         index=list(data.columns).index(default_target) if default_target in data.columns else 0,
#         key='selectbox_target'
#     )
#     # Guardar selección solo si el usuario cambió
#     if st.session_state.get('target_selected', None) != action_selected:
#         st.session_state['target_selected'] = action_selected

#     target = data[action_selected].iloc[-dias_historico:].values

#     resultados = []
#     for ticker in data.columns:
#         if ticker == action_selected:
#             continue
#         serie = data[ticker].values
#         corr_0 = np.corrcoef(target, serie[-dias_historico:])[0, 1]
#         corr_best, lag_best, serie_lag = calc_best_lag(target, serie, dias_historico, max_lag)
#         resultados.append({
#             "ticker": ticker,
#             "corr_0": corr_0,
#             "corr_best": corr_best,
#             "lag_best": lag_best,
#             "serie_0": serie[-dias_historico:],
#             "serie_lag": serie_lag
#         })

#     top7 = sorted(
#         [r for r in resultados if r["corr_best"] is not None],
#         key=lambda r: abs(r["corr_best"]),
#         reverse=True
#     )[:7]

#     fig_ori = go.Figure()
#     fig_ori.add_trace(go.Scatter(
#         x=np.arange(dias_historico), y=target,
#         mode='lines', name=action_selected, line=dict(color='black', width=3)
#     ))
#     for i, r in enumerate(top7):
#         fig_ori.add_trace(go.Scatter(
#             x=np.arange(dias_historico), y=r["serie_0"],
#             mode='lines', name=r["ticker"], opacity=0.6
#         ))
#     fig_ori.update_layout(title="Series originales (sin desfase)", height=400)

#     fig_lag = go.Figure()
#     fig_lag.add_trace(go.Scatter(
#         x=np.arange(dias_historico), y=target,
#         mode='lines', name=action_selected, line=dict(color='black', width=3)
#     ))
#     for i, r in enumerate(top7):
#         fig_lag.add_trace(go.Scatter(
#             x=np.arange(dias_historico), y=r["serie_lag"],
#             mode='lines', name=f"{r['ticker']} (lag {r['lag_best']})", opacity=0.7
#         ))
#     fig_lag.update_layout(title="Series alineadas por lag óptimo", height=400)

#     col1, col2 = st.columns(2)
#     with col1:
#         st.plotly_chart(fig_ori, use_container_width=True)
#     with col2:
#         st.plotly_chart(fig_lag, use_container_width=True)

#     st.markdown("### Tabla de los 7 mejores emparejamientos (por correlación absoluta tras desfase)")
#     tabla = pd.DataFrame({
#         "Ticker": [x["ticker"] for x in top7],
#         "Correlación sin desfase": [x["corr_0"] for x in top7],
#         "Correlación máxima (con lag)": [x["corr_best"] for x in top7],
#         "Lag óptimo (días)": [x["lag_best"] for x in top7]
#     })
#     st.dataframe(tabla.style.format({"Correlación sin desfase": "{:.2f}", "Correlación máxima (con lag)": "{:.2f}"}))

#     st.markdown("## Previsión de la Acción Objetivo usando Random Forest y ventana de 30 días")
#     dias_pred = st.slider("¿Cuántos días futuros predecir? (test set)", 1, 15, 7)
#     window_size = 30

#     target_full = data[action_selected].iloc[-dias_historico - window_size:].values
#     min_len = len(target_full)
#     correlated_full = []
#     for r in top7:
#         serie = r["serie_lag"]
#         if len(serie) > min_len:
#             serie = serie[-min_len:]
#         elif len(serie) < min_len:
#             serie = np.concatenate([np.full(min_len - len(serie), np.nan), serie])
#         correlated_full.append(serie)

#     mask = ~np.isnan(target_full)
#     for serie in correlated_full:
#         mask &= ~np.isnan(serie)
#     target_full = target_full[mask]
#     correlated_full = [serie[mask] for serie in correlated_full]

#     X_all, y_all = create_sliding_window_features(target_full, correlated_full, window_size=window_size)
#     total_samples = len(y_all)
#     dias_graf = dias_pred + 15

#     if total_samples < dias_pred + 30:
#         st.warning("No hay suficientes datos para la ventana seleccionada. Ajusta los parámetros.")
#         return

#     X_train = X_all[:-dias_pred]
#     y_train = y_all[:-dias_pred]
#     X_test = X_all[-dias_pred:]
#     y_test = y_all[-dias_pred:]

#     split = int(0.7 * len(X_train))
#     X_tr, X_val = X_train[:split], X_train[split:]
#     y_tr, y_val = y_train[:split], y_train[split:]

#     rf = RandomForestRegressor(n_estimators=200, random_state=42)
#     rf.fit(X_tr, y_tr)

#     y_val_pred = rf.predict(X_val)
#     mse = mean_squared_error(y_val, y_val_pred)
#     r2_val = r2_score(y_val, y_val_pred)
#     r2_val_disp = f"{r2_val:.2f}" if r2_val >= 0 else "—"

#     y_pred = rf.predict(X_test)
#     mse_test = mean_squared_error(y_test, y_pred)
#     r2_test = r2_score(y_test, y_pred)
#     r2_test_disp = f"{r2_test:.2f}" if r2_test >= 0 else "—"

#     # -------------------- GRAFICAR PREVISIÓN SIN DESFASE -----------------------
#     # Los últimos dias_graf días de y_all, pero la previsión inicia en el punto correcto
#     x_real = np.arange(total_samples - dias_graf, total_samples)
#     x_pred = np.arange(total_samples - dias_pred, total_samples)

#     fig_pred = go.Figure()
#     fig_pred.add_trace(go.Scatter(
#         x=x_real, y=y_all[-dias_graf:],
#         mode='lines+markers', name="Real", line=dict(color='blue')
#     ))
#     fig_pred.add_trace(go.Scatter(
#         x=x_pred, y=y_pred,
#         mode='lines+markers', name="Previsto RF", line=dict(color='orange')
#     ))
#     # Línea vertical de corte (donde empieza la previsión)
#     fig_pred.add_shape(type="line",
#         x0=total_samples - dias_pred - 0.5, y0=min(y_all[-dias_graf:]), x1=total_samples - dias_pred - 0.5, y1=max(y_all[-dias_graf:]),
#         line=dict(color="gray", width=1, dash="dot"))
#     fig_pred.update_layout(
#         title=f"Previsión Random Forest ({dias_pred} días + 15 días extra, ventana=30)",
#         xaxis_title="Día",
#         yaxis_title="Precio",
#         height=450,
#         showlegend=True
#     )

#     st.markdown(
#         f"""**MSE validación:** {mse:.2f} | **R² validación:** {r2_val_disp}  
# **MSE test:** {mse_test:.2f} | **R² test:** {r2_test_disp}"""
#     )
#     st.plotly_chart(fig_pred, use_container_width=True)
#     st.markdown(
#         f"""
#         - **Azul:** valores reales de la acción objetivo.
#         - **Naranja:** previsión de Random Forest para los {dias_pred} días seleccionados.
#         - Línea punteada gris: inicio de la previsión.
#         - El modelo utiliza la ventana de los últimos 30 días de la acción objetivo y las 7 correlacionadas, alineadas.
#         """
#     )

# if __name__ == "__main__":
#     show()







import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import random
import plotly.graph_objs as go
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

TICKERS_LIST = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA", "BRK-B", "JPM", "V",
    "UNH", "JNJ", "XOM", "PG", "MA", "HD", "CVX", "MRK", "ABBV", "PEP",
    "LLY", "KO", "AVGO", "COST", "WMT", "DIS", "BAC", "MCD", "ADBE", "PFE",
    "TMO", "CSCO", "DHR", "ABT", "ACN", "CRM", "TXN", "NKE", "VZ", "NEE",
    "CMCSA", "WFC", "LIN", "INTC", "QCOM", "MS", "HON", "PM", "UNP", "AMGN",
    "SBUX", "ORCL", "AMD", "CVS", "ISRG", "GILD", "BKNG", "NOW", "ZTS", "MDLZ"
]

def init_session_state(n_stocks, seed):
    if 'tickers' not in st.session_state or st.session_state['tickers_seed'] != seed or st.session_state['tickers_n'] != n_stocks:
        random.seed(seed)
        st.session_state['tickers'] = random.sample(TICKERS_LIST, n_stocks)
        st.session_state['tickers_seed'] = seed
        st.session_state['tickers_n'] = n_stocks

@st.cache_data(show_spinner=True)
def download_stocks(tickers, days, max_lag):
    period = days + max_lag + 40
    df = yf.download(tickers, period=f"{period}d", interval="1d", progress=False)
    data = df['Adj Close'] if 'Adj Close' in df else df['Close']
    return data

def get_valid_data(tickers, days=750, max_lag=30):
    data = download_stocks(tickers, days, max_lag)
    min_len = days + max_lag + 30
    valid_cols = [c for c in data.columns if data[c].dropna().shape[0] >= min_len]
    data_valid = data[valid_cols].iloc[-min_len:]
    data_valid = data_valid.dropna(axis=1)
    return data_valid

def calc_best_lag(target, candidate, dias, max_lag=30):
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

def create_sliding_window_features(target, correlated_list, window_size=30):
    num_samples = len(target) - window_size
    X = []
    y = []
    for i in range(num_samples):
        features = []
        features.extend(target[i:i+window_size])
        for corr in correlated_list:
            features.extend(corr[i:i+window_size])
        X.append(features)
        y.append(target[i+window_size])
    return np.array(X), np.array(y)

def show():
    st.title("Correlación y Previsión de Series Temporales con Sliding Window")
    st.write("Previsión basada en ventana de los últimos 30 días de la acción objetivo y de las 7 acciones más correlacionadas (alineadas).")
    seed = st.number_input("Semilla para selección de acciones (fijo)", value=42, min_value=0, max_value=9999, step=1)
    n_stocks = st.slider("Número de acciones a analizar", 10, 50, 20)
    dias_historico = st.slider("Ventana de días para comparar", 100, 1000, 750)
    max_lag = st.slider("Máximo desfase (lag, en días)", 5, 30, 25)

    init_session_state(n_stocks, seed)
    tickers = st.session_state['tickers']

    data = get_valid_data(tickers, dias_historico, max_lag)
    st.write("Acciones disponibles (con datos completos):", ", ".join(data.columns))
    st.dataframe(data)

    # --- INICIALIZAR EL KEY SOLO SI CAMBIA LA LISTA ---
    if 'target_selected' not in st.session_state or st.session_state.get('tickers_snapshot') != tuple(data.columns):
        st.session_state['target_selected'] = data.columns[0]
        st.session_state['tickers_snapshot'] = tuple(data.columns)

    # USAR key y que sea el widget quien gestiona la selección
    st.selectbox(
        "Selecciona la acción objetivo",
        options=data.columns,
        key="target_selected"
    )
    action_selected = st.session_state["target_selected"]

    target = data[action_selected].iloc[-dias_historico:].values

    resultados = []
    for ticker in data.columns:
        if ticker == action_selected:
            continue
        serie = data[ticker].values
        corr_0 = np.corrcoef(target, serie[-dias_historico:])[0, 1]
        corr_best, lag_best, serie_lag = calc_best_lag(target, serie, dias_historico, max_lag)
        resultados.append({
            "ticker": ticker,
            "corr_0": corr_0,
            "corr_best": corr_best,
            "lag_best": lag_best,
            "serie_0": serie[-dias_historico:],
            "serie_lag": serie_lag
        })

    top7 = sorted(
        [r for r in resultados if r["corr_best"] is not None],
        key=lambda r: abs(r["corr_best"]),
        reverse=True
    )[:7]

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

    st.markdown("## Previsión de la Acción Objetivo usando Random Forest y ventana de 30 días")
    dias_pred = st.slider("¿Cuántos días futuros predecir? (test set)", 1, 15, 7)
    window_size = 30

    target_full = data[action_selected].iloc[-dias_historico - window_size:].values
    min_len = len(target_full)
    correlated_full = []
    for r in top7:
        serie = r["serie_lag"]
        if len(serie) > min_len:
            serie = serie[-min_len:]
        elif len(serie) < min_len:
            serie = np.concatenate([np.full(min_len - len(serie), np.nan), serie])
        correlated_full.append(serie)

    mask = ~np.isnan(target_full)
    for serie in correlated_full:
        mask &= ~np.isnan(serie)
    target_full = target_full[mask]
    correlated_full = [serie[mask] for serie in correlated_full]

    X_all, y_all = create_sliding_window_features(target_full, correlated_full, window_size=window_size)
    total_samples = len(y_all)
    dias_graf = dias_pred + 15

    if total_samples < dias_pred + 30:
        st.warning("No hay suficientes datos para la ventana seleccionada. Ajusta los parámetros.")
        return

    X_train = X_all[:-dias_pred]
    y_train = y_all[:-dias_pred]
    X_test = X_all[-dias_pred:]
    y_test = y_all[-dias_pred:]

    split = int(0.7 * len(X_train))
    X_tr, X_val = X_train[:split], X_train[split:]
    y_tr, y_val = y_train[:split], y_train[split:]

    rf = RandomForestRegressor(n_estimators=200, random_state=42)
    rf.fit(X_tr, y_tr)

    y_val_pred = rf.predict(X_val)
    mse = mean_squared_error(y_val, y_val_pred)
    r2_val = r2_score(y_val, y_val_pred)
    r2_val_disp = f"{r2_val:.2f}" if r2_val >= 0 else "—"

    y_pred = rf.predict(X_test)
    mse_test = mean_squared_error(y_test, y_pred)
    r2_test = r2_score(y_test, y_pred)
    r2_test_disp = f"{r2_test:.2f}" if r2_test >= 0 else "—"

    x_real = np.arange(total_samples - dias_graf, total_samples)
    x_pred = x_real[-dias_pred:]

    fig_pred = go.Figure()
    fig_pred.add_trace(go.Scatter(
        x=x_real, y=y_all[-dias_graf:],
        mode='lines+markers', name="Real", line=dict(color='blue')
    ))
    fig_pred.add_trace(go.Scatter(
        x=x_pred, y=y_pred,
        mode='lines+markers', name="Previsto RF", line=dict(color='orange')
    ))
    fig_pred.add_shape(type="line",
        x0=x_pred[0] - 0.5, y0=min(y_all[-dias_graf:]), x1=x_pred[0] - 0.5, y1=max(y_all[-dias_graf:]),
        line=dict(color="gray", width=1, dash="dot"))
    fig_pred.update_layout(
        title=f"Previsión Random Forest ({dias_pred} días + 15 días extra, ventana=30)",
        xaxis_title="Día",
        yaxis_title="Precio",
        height=450,
        showlegend=True
    )

    st.markdown(
        f"""**MSE validación:** {mse:.2f} | **R² validación:** {r2_val_disp}  
**MSE test:** {mse_test:.2f} | **R² test:** {r2_test_disp}"""
    )
    st.plotly_chart(fig_pred, use_container_width=True)
    st.markdown(
        f"""
        - **Azul:** valores reales de la acción objetivo.
        - **Naranja:** previsión de Random Forest para los {dias_pred} días seleccionados.
        - Línea punteada gris: inicio de la previsión.
        - El modelo utiliza la ventana de los últimos 30 días de la acción objetivo y las 7 correlacionadas, alineadas.
        """
    )

if __name__ == "__main__":
    show()
