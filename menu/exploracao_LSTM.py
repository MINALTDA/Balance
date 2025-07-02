# # # # #menu/exploracao.py
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

#     init_session_state(n_stocks, seed)
#     tickers = st.session_state['tickers']

#     data = get_valid_data(tickers, dias_historico, max_lag)
#     st.write("Acciones disponibles (con datos completos):", ", ".join(data.columns))
#     st.dataframe(data)

#     # --- INICIALIZAR EL KEY SOLO SI CAMBIA LA LISTA ---
#     if 'target_selected' not in st.session_state or st.session_state.get('tickers_snapshot') != tuple(data.columns):
#         st.session_state['target_selected'] = data.columns[0]
#         st.session_state['tickers_snapshot'] = tuple(data.columns)

#     # USAR key y que sea el widget quien gestiona la selección
#     st.selectbox(
#         "Selecciona la acción objetivo",
#         options=data.columns,
#         key="target_selected"
#     )
#     action_selected = st.session_state["target_selected"]

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

#     rf = RandomForestRegressor(n_estimators=100, random_state=42)
#     rf.fit(X_tr, y_tr)

#     y_val_pred = rf.predict(X_val)
#     mse = mean_squared_error(y_val, y_val_pred)
#     r2_val = r2_score(y_val, y_val_pred)
#     r2_val_disp = f"{r2_val:.2f}" if r2_val >= 0 else "—"

#     y_pred = rf.predict(X_test)
#     mse_test = mean_squared_error(y_test, y_pred)
#     r2_test = r2_score(y_test, y_pred)
#     r2_test_disp = f"{r2_test:.2f}" if r2_test >= 0 else "—"

#     x_real = np.arange(total_samples - dias_graf, total_samples)
#     x_pred = x_real[-dias_pred:]

#     fig_pred = go.Figure()
#     fig_pred.add_trace(go.Scatter(
#         x=x_real, y=y_all[-dias_graf:],
#         mode='lines+markers', name="Real", line=dict(color='blue')
#     ))
#     fig_pred.add_trace(go.Scatter(
#         x=x_pred, y=y_pred,
#         mode='lines+markers', name="Previsto RF", line=dict(color='orange')
#     ))
#     fig_pred.add_shape(type="line",
#         x0=x_pred[0] - 0.5, y0=min(y_all[-dias_graf:]), x1=x_pred[0] - 0.5, y1=max(y_all[-dias_graf:]),
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

# def zscore_normalize(arr):
#     mu = np.mean(arr)
#     sigma = np.std(arr)
#     if sigma == 0: sigma = 1
#     arr_norm = (arr - mu) / sigma
#     return arr_norm, mu, sigma

# def zscore_denormalize(arr_norm, mu, sigma):
#     return arr_norm * sigma + mu

# def show():
#     st.title("Correlación y Previsión de Series Temporales con Sliding Window (Top 7 alineado, autoregresivo)")
#     st.write("Previsión basada en ventana de los últimos 30 días de la acción objetivo y de las 7 acciones más correlacionadas (alineadas, sin fuga de datos futuros).")
#     seed = st.number_input("Semilla para selección de acciones (fijo)", value=42, min_value=0, max_value=9999, step=1)
#     n_stocks = st.slider("Número de acciones a analizar", 10, 50, 20)
#     dias_historico = st.slider("Ventana de días para comparar", 100, 1000, 750)
#     max_lag = st.slider("Máximo desfase (lag, en días)", 5, 30, 25)

#     init_session_state(n_stocks, seed)
#     tickers = st.session_state['tickers']

#     data = get_valid_data(tickers, dias_historico, max_lag)
#     st.write("Acciones disponibles (con datos completos):", ", ".join(data.columns))
#     st.dataframe(data)

#     # --- INICIALIZAR EL KEY SOLO SI CAMBIA LA LISTA ---
#     if 'target_selected' not in st.session_state or st.session_state.get('tickers_snapshot') != tuple(data.columns):
#         st.session_state['target_selected'] = data.columns[0]
#         st.session_state['tickers_snapshot'] = tuple(data.columns)

#     st.selectbox(
#         "Selecciona la acción objetivo",
#         options=data.columns,
#         key="target_selected"
#     )
#     action_selected = st.session_state["target_selected"]

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

#     # División de entrenamiento/test igual que antes:
#     X_train = X_all[:-dias_pred]
#     y_train = y_all[:-dias_pred]
#     X_test = X_all[-dias_pred:]
#     y_test = y_all[-dias_pred:]

#     # Normalización SOLO sobre train:
#     X_train_norm, mu_X, std_X = zscore_normalize(X_train)
#     y_train_norm, mu_y, std_y = zscore_normalize(y_train)
#     X_test_norm = (X_test - mu_X) / std_X

#     split = int(0.7 * len(X_train_norm))
#     X_tr, X_val = X_train_norm[:split], X_train_norm[split:]
#     y_tr, y_val = y_train_norm[:split], y_train_norm[split:]

#     rf = RandomForestRegressor(n_estimators=100, random_state=42)
#     rf.fit(X_tr, y_tr)

#     y_val_pred = rf.predict(X_val)
#     mse = mean_squared_error(zscore_denormalize(y_val, mu_y, std_y), zscore_denormalize(y_val_pred, mu_y, std_y))
#     r2_val = r2_score(zscore_denormalize(y_val, mu_y, std_y), zscore_denormalize(y_val_pred, mu_y, std_y))
#     r2_val_disp = f"{r2_val:.2f}" if r2_val >= 0 else "—"

#     # Modelos para las correlacionadas (autoregresivo)
#     corrs_rfs = []
#     mu_corrs, std_corrs = [], []
#     for i in range(len(top7)):
#         corr_series = correlated_full[i]
#         y_corr = []
#         X_corr = []
#         for j in range(len(corr_series) - window_size - dias_pred):
#             X_corr.append(corr_series[j:j+window_size])
#             y_corr.append(corr_series[j+window_size])
#         X_corr = np.array(X_corr)
#         y_corr = np.array(y_corr)
#         X_corr_norm, mu_corr, std_corr = zscore_normalize(X_corr)
#         y_corr_norm, mu_y_corr, std_y_corr = zscore_normalize(y_corr)
#         rf_corr = RandomForestRegressor(n_estimators=50, random_state=42)
#         rf_corr.fit(X_corr_norm, y_corr_norm)
#         corrs_rfs.append(rf_corr)
#         mu_corrs.append(mu_corr)
#         std_corrs.append(std_corr)

#     # --- Previsión autoregresiva multi-step ---
#     ult_idx = -dias_pred
#     last_target_window = target_full[ult_idx - window_size:ult_idx].tolist()
#     last_corr_windows = [corr[ult_idx - window_size:ult_idx].tolist() for corr in correlated_full]
#     last_target_window_norm = [(x - mu_y) / std_y for x in last_target_window]
#     last_corr_windows_norm = []
#     for i, win in enumerate(last_corr_windows):
#         mu_corr, std_corr = mu_corrs[i], std_corrs[i]
#         last_corr_windows_norm.append([(x - mu_corr) / std_corr for x in win])

#     y_pred_ar_norm = []
#     corr_pred_ar_norm = [win.copy() for win in last_corr_windows_norm]
#     for step in range(dias_pred):
#         features = []
#         features.extend(last_target_window_norm)
#         for win in corr_pred_ar_norm:
#             features.extend(win)
#         pred_norm = rf.predict([features])[0]
#         y_pred_ar_norm.append(pred_norm)
#         last_target_window_norm.pop(0)
#         last_target_window_norm.append(pred_norm)
#         # Autoregresivo en correlacionadas
#         for i in range(len(corr_pred_ar_norm)):
#             rf_corr = corrs_rfs[i]
#             mu_corr, std_corr = mu_corrs[i], std_corrs[i]
#             feat_corr = corr_pred_ar_norm[i][-window_size:]
#             pred_corr_norm = rf_corr.predict([feat_corr])[0]
#             corr_pred_ar_norm[i].pop(0)
#             corr_pred_ar_norm[i].append(pred_corr_norm)

#     y_pred_ar = zscore_denormalize(np.array(y_pred_ar_norm), mu_y, std_y)
#     mse_test = mean_squared_error(y_test, y_pred_ar)
#     r2_test = r2_score(y_test, y_pred_ar)
#     r2_test_disp = f"{r2_test:.2f}" if r2_test >= 0 else "—"

#     x_real = np.arange(total_samples - dias_graf, total_samples)
#     x_pred = x_real[-dias_pred:]

#     fig_pred = go.Figure()
#     fig_pred.add_trace(go.Scatter(
#         x=x_real, y=y_all[-dias_graf:],
#         mode='lines+markers', name="Real", line=dict(color='blue')
#     ))
#     fig_pred.add_trace(go.Scatter(
#         x=x_pred, y=y_pred_ar,
#         mode='lines+markers', name="Previsto RF autoregresivo", line=dict(color='orange')
#     ))
#     fig_pred.add_shape(type="line",
#         x0=x_pred[0] - 0.5, y0=min(y_all[-dias_graf:]), x1=x_pred[0] - 0.5, y1=max(y_all[-dias_graf:]),
#         line=dict(color="gray", width=1, dash="dot"))
#     fig_pred.update_layout(
#         title=f"Previsión Random Forest Multi-step Autoregresivo NORMALIZADO ({dias_pred} días + 15 días extra, ventana=30)",
#         xaxis_title="Día",
#         yaxis_title="Precio",
#         height=450,
#         showlegend=True
#     )

#     st.markdown(
#         f"""**MSE validación:** {mse:.2f} | **R² validación:** {r2_val_disp}  
# **MSE test (autoregresivo):** {mse_test:.2f} | **R² test:** {r2_test_disp}"""
#     )
#     st.plotly_chart(fig_pred, use_container_width=True)
#     st.markdown(
#         f"""
#         - **Azul:** valores reales de la acción objetivo.
#         - **Naranja:** previsión Random Forest para los {dias_pred} días seleccionados (autoregresivo).
#         - Línea punteada gris: inicio de la previsión.
#         - El modelo utiliza la ventana de los últimos 30 días de la acción objetivo y las 7 correlacionadas, alineadas.
#         - Las correlacionadas también son previstas autoregresivamente.
#         """
#     )

# if __name__ == "__main__":
#     show()













# # menu/exploracao.py
# import streamlit as st
# import pandas as pd
# import numpy as np
# import yfinance as yf
# import random
# import plotly.graph_objs as go
# from sklearn.metrics import mean_squared_error, r2_score

# import tensorflow as tf
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import LSTM, Dense, Dropout
# from tensorflow.keras.callbacks import EarlyStopping

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
#     for lag in range(3, max_lag + 3):
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

# def zscore_normalize(train_series, test_series=None):
#     mu = np.mean(train_series)
#     sigma = np.std(train_series)
#     if sigma == 0:
#         sigma = 1
#     norm_train = (train_series - mu) / sigma
#     if test_series is not None:
#         norm_test = (test_series - mu) / sigma
#         return norm_train, norm_test, mu, sigma
#     return norm_train, mu, sigma

# def zscore_denormalize(norm_values, mu, sigma):
#     return norm_values * sigma + mu

# def create_sliding_window_multivariate(target, correlated_list, window_size=30):
#     # Output shape: (samples, window, features)
#     num_samples = len(target) - window_size
#     X = []
#     y = []
#     for i in range(num_samples):
#         features = []
#         for offset in range(window_size):
#             day_feats = [target[i + offset]]
#             for corr in correlated_list:
#                 day_feats.append(corr[i + offset])
#             features.append(day_feats)
#         X.append(features)
#         y.append(target[i + window_size])
#     return np.array(X), np.array(y)

# def show():
#     st.title("Previsión de Series Temporales con LSTM autoregresivo + Top 7 correlacionadas")
#     seed = st.number_input("Semilla para selección de acciones (fijo)", value=42, min_value=0, max_value=9999, step=1)
#     n_stocks = st.slider("Número de acciones a analizar", 10, 50, 20)
#     dias_historico = st.slider("Ventana de días para comparar", 100, 1000, 750)
#     max_lag = st.slider("Máximo desfase (lag, en días)", 5, 30, 25)

#     # random.seed(seed)
#     np.random.seed(seed)
#     tf.random.set_seed(seed)

#     init_session_state(n_stocks, seed)
#     tickers = st.session_state['tickers']

#     data = get_valid_data(tickers, dias_historico, max_lag)
#     st.write("Acciones disponibles (con datos completos):", ", ".join(data.columns))
#     st.dataframe(data)

#     if 'target_selected' not in st.session_state or st.session_state.get('tickers_snapshot') != tuple(data.columns):
#         st.session_state['target_selected'] = data.columns[0]
#         st.session_state['tickers_snapshot'] = tuple(data.columns)

#     st.selectbox(
#         "Selecciona la acción objetivo",
#         options=data.columns,
#         key="target_selected"
#     )
#     action_selected = st.session_state["target_selected"]

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

#     # -------- Visualizaciones originales --------
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

#     st.markdown("## Previsión de la Acción Objetivo usando LSTM autoregresivo + normalización")
#     dias_pred = st.slider("¿Cuántos días futuros predecir? (test set)", 1, 15, 7)
#     window_size = 30

#     # Preparamos los datos para normalización y ventanas
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

#     # Normalización robusta
#     train_idx = -(dias_pred + 30) if (dias_pred + 30) < len(target_full) else 0
#     mu_y = np.mean(target_full[:train_idx])
#     std_y = np.std(target_full[:train_idx]) if np.std(target_full[:train_idx]) != 0 else 1
#     target_full_norm = (target_full - mu_y) / std_y
#     correlated_full_norm = []
#     for corr in correlated_full:
#         mu_corr = np.mean(corr[:train_idx])
#         std_corr = np.std(corr[:train_idx]) if np.std(corr[:train_idx]) != 0 else 1
#         correlated_full_norm.append((corr - mu_corr) / std_corr)

#     # Crear ventanas multivariadas (target + correlacionadas como features)
#     X_all, y_all = create_sliding_window_multivariate(target_full_norm, correlated_full_norm, window_size=window_size)
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

#     # Modelo LSTM robusto
#     n_feats = X_train.shape[2]
#     model = Sequential([
#         LSTM(64, input_shape=(window_size, n_feats), return_sequences=False),
#         Dropout(0.3),
#         Dense(32, activation='relu'),
#         Dense(1)
#     ])
#     model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss='mse')
#     es = EarlyStopping(monitor='val_loss', patience=8, restore_best_weights=True)
#     model.fit(X_tr, y_tr, validation_data=(X_val, y_val), epochs=120, batch_size=16, verbose=0, callbacks=[es])

#     # Validación clásica (no autoregresiva)
#     y_val_pred = model.predict(X_val, verbose=0).flatten()
#     mse = mean_squared_error(y_val, y_val_pred)
#     r2_val = r2_score(y_val, y_val_pred)
#     r2_val_disp = f"{r2_val:.2f}" if r2_val >= 0 else "—"

#     # --- Previsión multi-step autoregresiva
#     ult_idx = -dias_pred
#     last_window = X_all[ult_idx - 1].copy()  # shape (window, n_feats)
#     y_pred_ar = []
#     for step in range(dias_pred):
#         # Modelo espera shape (1, window, features)
#         pred_norm = model.predict(last_window[np.newaxis, ...], verbose=0)[0, 0]
#         y_pred_ar.append(pred_norm)
#         # Shift ventana: actualiza solo target en la ventana, correlacionadas se mantienen fijas en test (no se autoregresionan)
#         next_window = np.roll(last_window, -1, axis=0)
#         next_window[-1, 0] = pred_norm
#         last_window = next_window

#     y_pred_ar_real = zscore_denormalize(np.array(y_pred_ar), mu_y, std_y)
#     y_test_real = zscore_denormalize(y_test, mu_y, std_y)
#     mse_test = mean_squared_error(y_test_real, y_pred_ar_real)
#     r2_test = r2_score(y_test_real, y_pred_ar_real)
#     r2_test_disp = f"{r2_test:.2f}" if r2_test >= 0 else "—"

#     x_real = np.arange(total_samples - dias_graf, total_samples)
#     x_pred = x_real[-dias_pred:]

#     fig_pred = go.Figure()
#     fig_pred.add_trace(go.Scatter(
#         x=x_real, y=zscore_denormalize(y_all[-dias_graf:], mu_y, std_y),
#         mode='lines+markers', name="Real", line=dict(color='blue')
#     ))
#     fig_pred.add_trace(go.Scatter(
#         x=x_pred, y=y_pred_ar_real,
#         mode='lines+markers', name="Previsto LSTM autoregresivo", line=dict(color='orange')
#     ))
#     fig_pred.add_shape(type="line",
#         x0=x_pred[0] - 0.5, y0=min(zscore_denormalize(y_all[-dias_graf:], mu_y, std_y)),
#         x1=x_pred[0] - 0.5, y1=max(zscore_denormalize(y_all[-dias_graf:], mu_y, std_y)),
#         line=dict(color="gray", width=1, dash="dot"))
#     fig_pred.update_layout(
#         title=f"Previsión LSTM Multi-step Autoregresivo NORMALIZADO ({dias_pred} días + 15 días extra, ventana=30)",
#         xaxis_title="Día",
#         yaxis_title="Precio",
#         height=450,
#         showlegend=True
#     )

#     st.markdown(
#         f"""**MSE validación:** {mse:.2f} | **R² validación:** {r2_val_disp}  
# **MSE test (autoregresivo):** {mse_test:.2f} | **R² test:** {r2_test_disp}"""
#     )
#     st.plotly_chart(fig_pred, use_container_width=True)
#     st.markdown(
#         f"""
#         - **Azul:** valores reales de la acción objetivo.
#         - **Naranja:** previsión LSTM autoregresivo puro.
#         - Línea punteada gris: inicio de la previsión.
#         - Entradas futuras usan solo valores previstos, nunca reales.
#         - Cada ventana contiene la acción objetivo y las 7 más correlacionadas (alineadas y normalizadas).
#         """
#     )

# if __name__ == "__main__":
#     show()













# menu/exploracao.py
import os
import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import random
import plotly.graph_objs as go
from sklearn.metrics import mean_squared_error, r2_score

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

# Opciones para reproducibilidad máxima
os.environ["PYTHONHASHSEED"] = "42"
os.environ["TF_DETERMINISTIC_OPS"] = "1"

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
    for lag in range(5, max_lag + 5):
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

def zscore_normalize(train_series, test_series=None):
    mu = np.mean(train_series)
    sigma = np.std(train_series)
    if sigma == 0:
        sigma = 1
    norm_train = (train_series - mu) / sigma
    if test_series is not None:
        norm_test = (test_series - mu) / sigma
        return norm_train, norm_test, mu, sigma
    return norm_train, mu, sigma

def zscore_denormalize(norm_values, mu, sigma):
    return norm_values * sigma + mu

def create_sliding_window_multivariate(target, correlated_list, window_size=30):
    # Output shape: (samples, window, features)
    num_samples = len(target) - window_size
    X = []
    y = []
    for i in range(num_samples):
        features = []
        for offset in range(window_size):
            day_feats = [target[i + offset]]
            for corr in correlated_list:
                day_feats.append(corr[i + offset])
            features.append(day_feats)
        X.append(features)
        y.append(target[i + window_size])
    return np.array(X), np.array(y)

def show():
    st.title("Previsión de Series Temporales con LSTM autoregresivo + Top correlacionadas")
    seed = st.number_input("Semilla para selección de acciones (fijo)", value=42, min_value=0, max_value=9999, step=1)
    n_stocks = st.slider("Número de acciones a analizar", 10, 50, 20)
    dias_historico = st.slider("Ventana de días para comparar", 100, 1000, 750)
    max_lag = st.slider("Máximo desfase (lag, en días)", 5, 30, 25)

    # Fijar todas las semillas para máxima reproducibilidad
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)

    init_session_state(n_stocks, seed)
    tickers = st.session_state['tickers']

    data = get_valid_data(tickers, dias_historico, max_lag)
    st.write("Acciones disponibles (con datos completos):", ", ".join(data.columns))
    st.dataframe(data)

    if 'target_selected' not in st.session_state or st.session_state.get('tickers_snapshot') != tuple(data.columns):
        st.session_state['target_selected'] = data.columns[0]
        st.session_state['tickers_snapshot'] = tuple(data.columns)

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

    # Top 7 por valor absoluto de correlación máxima con lag
    top7 = sorted(
        [r for r in resultados if r["corr_best"] is not None],
        key=lambda r: abs(r["corr_best"]),
        reverse=True
    )[:7]

    # --------- FILTRO POR UMBRAL DE CORRELACIÓN -------------
    # Umbral de correlación máxima absoluta (puedes ajustar aquí)
    CORR_THRESHOLD = 0.7  # <-- Edita este valor para aflojar/ajustar el filtro

    top_corr = [r for r in top7 if abs(r["corr_best"]) >= CORR_THRESHOLD]

    st.markdown(
        f"**Solo se usan en el modelo las correlacionadas con |correlación máxima (con lag)| ≥ {CORR_THRESHOLD}. "
        f"Seleccionadas: {len(top_corr)} de 7 posibles.**"
    )

    # -------- Visualizaciones originales --------
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

    st.markdown("## Previsión de la Acción Objetivo usando LSTM autoregresivo + normalización")
    dias_pred = st.slider("¿Cuántos días futuros predecir? (test set)", 1, 15, 7)
    window_size = 30

    # Preparamos los datos para normalización y ventanas
    target_full = data[action_selected].iloc[-dias_historico - window_size:].values
    min_len = len(target_full)
    correlated_full = []
    for r in top_corr:  # <-- SOLO LAS CORRELACIONADAS QUE PASAN EL FILTRO
        serie = r["serie_lag"]
        if len(serie) > min_len:
            serie = serie[-min_len:]
        elif len(serie) < min_len:
            serie = np.concatenate([np.full(min_len - len(serie), np.nan), serie])
        correlated_full.append(serie)

    # Si no hay correlacionadas que pasen el filtro, solo se usará la serie objetivo
    mask = ~np.isnan(target_full)
    for serie in correlated_full:
        mask &= ~np.isnan(serie)
    target_full = target_full[mask]
    correlated_full = [serie[mask] for serie in correlated_full]

    # Normalización robusta
    train_idx = -(dias_pred + 30) if (dias_pred + 30) < len(target_full) else 0
    mu_y = np.mean(target_full[:train_idx])
    std_y = np.std(target_full[:train_idx]) if np.std(target_full[:train_idx]) != 0 else 1
    target_full_norm = (target_full - mu_y) / std_y
    correlated_full_norm = []
    for corr in correlated_full:
        mu_corr = np.mean(corr[:train_idx])
        std_corr = np.std(corr[:train_idx]) if np.std(corr[:train_idx]) != 0 else 1
        correlated_full_norm.append((corr - mu_corr) / std_corr)

    # Crear ventanas multivariadas (target + correlacionadas como features)
    X_all, y_all = create_sliding_window_multivariate(target_full_norm, correlated_full_norm, window_size=window_size)
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

    # Modelo LSTM robusto
    n_feats = X_train.shape[2]
    model = Sequential([
        LSTM(64, input_shape=(window_size, n_feats), return_sequences=False),
        Dropout(0.3),
        Dense(32, activation='relu'),
        Dense(1)
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss='mse')
    es = EarlyStopping(monitor='val_loss', patience=8, restore_best_weights=True)
    model.fit(X_tr, y_tr, validation_data=(X_val, y_val), epochs=120, batch_size=16, verbose=0, callbacks=[es])

    # Validación clásica (no autoregresiva)
    y_val_pred = model.predict(X_val, verbose=0).flatten()
    mse = mean_squared_error(y_val, y_val_pred)
    r2_val = r2_score(y_val, y_val_pred)
    r2_val_disp = f"{r2_val:.2f}" if r2_val >= 0 else "—"

    # --- Previsión multi-step autoregresiva
    ult_idx = -dias_pred
    last_window = X_all[ult_idx - 1].copy()  # shape (window, n_feats)
    y_pred_ar = []
    for step in range(dias_pred):
        # Modelo espera shape (1, window, features)
        pred_norm = model.predict(last_window[np.newaxis, ...], verbose=0)[0, 0]
        y_pred_ar.append(pred_norm)
        # Shift ventana: actualiza solo target en la ventana, correlacionadas se mantienen fijas en test (no se autoregresionan)
        next_window = np.roll(last_window, -1, axis=0)
        next_window[-1, 0] = pred_norm
        last_window = next_window

    y_pred_ar_real = zscore_denormalize(np.array(y_pred_ar), mu_y, std_y)
    y_test_real = zscore_denormalize(y_test, mu_y, std_y)
    mse_test = mean_squared_error(y_test_real, y_pred_ar_real)
    r2_test = r2_score(y_test_real, y_pred_ar_real)
    r2_test_disp = f"{r2_test:.2f}" if r2_test >= 0 else "—"

    x_real = np.arange(total_samples - dias_graf, total_samples)
    x_pred = x_real[-dias_pred:]

    fig_pred = go.Figure()
    fig_pred.add_trace(go.Scatter(
        x=x_real, y=zscore_denormalize(y_all[-dias_graf:], mu_y, std_y),
        mode='lines+markers', name="Real", line=dict(color='blue')
    ))
    fig_pred.add_trace(go.Scatter(
        x=x_pred, y=y_pred_ar_real,
        mode='lines+markers', name="Previsto LSTM autoregresivo", line=dict(color='orange')
    ))
    fig_pred.add_shape(type="line",
        x0=x_pred[0] - 0.5, y0=min(zscore_denormalize(y_all[-dias_graf:], mu_y, std_y)),
        x1=x_pred[0] - 0.5, y1=max(zscore_denormalize(y_all[-dias_graf:], mu_y, std_y)),
        line=dict(color="gray", width=1, dash="dot"))
    fig_pred.update_layout(
        title=f"Previsión LSTM Multi-step Autoregresivo NORMALIZADO ({dias_pred} días + 15 días extra, ventana=30)",
        xaxis_title="Día",
        yaxis_title="Precio",
        height=450,
        showlegend=True
    )

    st.markdown(
        f"""**MSE validación:** {mse:.2f} | **R² validación:** {r2_val_disp}  
**MSE test (autoregresivo):** {mse_test:.2f} | **R² test:** {r2_test_disp}"""
    )
    st.plotly_chart(fig_pred, use_container_width=True)
    st.markdown(
        f"""
        - **Azul:** valores reales de la acción objetivo.
        - **Naranja:** previsión LSTM autoregresivo puro.
        - Línea punteada gris: inicio de la previsión.
        - Entradas futuras usan solo valores previstos, nunca reales.
        - Cada ventana contiene la acción objetivo y las correlacionadas con |corr|≥{CORR_THRESHOLD} (alineadas y normalizadas).
        """
    )

if __name__ == "__main__":
    show()
