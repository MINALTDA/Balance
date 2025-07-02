# # menu/exploracao.py
# import os
# import streamlit as st
# import pandas as pd
# import numpy as np
# import yfinance as yf
# import random
# import plotly.graph_objs as go
# from sklearn.ensemble import RandomForestRegressor
# from sklearn.metrics import mean_squared_error, r2_score
# from sklearn.model_selection import TimeSeriesSplit, GridSearchCV

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
#     num_samples = len(target) - window_size
#     X = []
#     y = []
#     for i in range(num_samples):
#         features = []
#         for offset in range(window_size):
#             day_feats = [target[i + offset]]
#             for corr in correlated_list:
#                 day_feats.append(corr[i + offset])
#             features.extend(day_feats)
#         X.append(features)
#         y.append(target[i + window_size])
#     return np.array(X), np.array(y)

# def show():
#     st.title("Previsión con Random Forest + GridSearchCV y TimeSeriesSplit")
#     seed = st.number_input("Semilla para selección de acciones (fijo)", value=42, min_value=0, max_value=9999, step=1)
#     n_stocks = st.slider("Número de acciones a analizar", 10, 50, 20)
#     dias_historico = st.slider("Ventana de días para comparar", 100, 1000, 750)
#     max_lag = st.slider("Máximo desfase (lag, en días)", 5, 30, 25)

#     random.seed(seed)
#     np.random.seed(seed)

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

#     # --------- FILTRO POR UMBRAL DE CORRELACIÓN -------------
#     CORR_THRESHOLD = 0.5  # <-- Edita este valor para aflojar/ajustar el filtro
#     top_corr = [r for r in top7 if abs(r["corr_best"]) >= CORR_THRESHOLD]

#     st.markdown(
#         f"**Solo se usan en el modelo las correlacionadas con |correlación máxima (con lag)| ≥ {CORR_THRESHOLD}. "
#         f"Seleccionadas: {len(top_corr)} de 7 posibles.**"
#     )

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

#     st.markdown("## Previsión de la Acción Objetivo usando Random Forest + tuning")
#     dias_pred = st.slider("¿Cuántos días futuros predecir? (test set)", 1, 15, 7)
#     window_size = 30

#     target_full = data[action_selected].iloc[-dias_historico - window_size:].values
#     min_len = len(target_full)
#     correlated_full = []
#     for r in top_corr:
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

#     # Normalización robusta usando solo el set de entrenamiento
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

#     # --- GridSearchCV con TimeSeriesSplit ---
#     param_grid = {
#         "n_estimators": [50, 100, 200],
#         "max_depth": [3, 5, 10, None],
#         "min_samples_split": [2, 5],
#         "min_samples_leaf": [1, 2, 4],
#     }
#     tscv = TimeSeriesSplit(n_splits=4)
#     rf = RandomForestRegressor(random_state=seed)
#     grid = GridSearchCV(rf, param_grid, cv=tscv, scoring="neg_mean_squared_error", n_jobs=-1)
#     with st.spinner("Ajustando hiperparámetros (puede tardar un poco)..."):
#         grid.fit(X_train, y_train)
#     best_rf = grid.best_estimator_

#     st.success(f"Mejores hiperparámetros: {grid.best_params_}")

#     # Validación clásica (no autoregresiva)
#     split = int(0.7 * len(X_train))
#     X_tr, X_val = X_train[:split], X_train[split:]
#     y_tr, y_val = y_train[:split], y_train[split:]
#     best_rf.fit(X_tr, y_tr)
#     y_val_pred = best_rf.predict(X_val)
#     mse = mean_squared_error(y_val, y_val_pred)
#     r2_val = r2_score(y_val, y_val_pred)
#     r2_val_disp = f"{r2_val:.2f}" if r2_val >= 0 else "—"

#     # --- Previsión multi-step autoregresiva (realista)
#     ult_idx = -dias_pred
#     last_window = X_all[ult_idx - 1].copy()  # shape (window * n_feats,)
#     y_pred_ar = []
#     for step in range(dias_pred):
#         pred_norm = best_rf.predict(last_window.reshape(1, -1))[0]
#         y_pred_ar.append(pred_norm)
#         # Shift ventana: actualiza solo target, correlacionadas se mantienen fijas (no autoregresión)
#         next_window = np.roll(last_window, -1)
#         next_window[(window_size - 1) * (1 + len(correlated_full))] = pred_norm
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
#         mode='lines+markers', name="Previsto RF autoregresivo", line=dict(color='orange')
#     ))
#     fig_pred.add_shape(type="line",
#         x0=x_pred[0] - 0.5, y0=min(zscore_denormalize(y_all[-dias_graf:], mu_y, std_y)),
#         x1=x_pred[0] - 0.5, y1=max(zscore_denormalize(y_all[-dias_graf:], mu_y, std_y)),
#         line=dict(color="gray", width=1, dash="dot"))
#     fig_pred.update_layout(
#         title=f"Previsión Random Forest Multi-step Autoregresivo ({dias_pred} días + 15 días extra, ventana=30)",
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
#         - **Naranja:** previsión Random Forest autoregresivo.
#         - Línea punteada gris: inicio de la previsión.
#         - Entradas futuras usan solo valores previstos, nunca reales.
#         - Cada ventana contiene la acción objetivo y las correlacionadas con |corr|≥{CORR_THRESHOLD} (alineadas y normalizadas).
#         - Tuning automático con GridSearchCV + TimeSeriesSplit.
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
# from sklearn.ensemble import RandomForestRegressor
# from sklearn.metrics import mean_squared_error, r2_score
# from sklearn.model_selection import TimeSeriesSplit, GridSearchCV

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
#     num_samples = len(target) - window_size
#     X = []
#     y = []
#     for i in range(num_samples):
#         features = []
#         for offset in range(window_size):
#             day_feats = [target[i + offset]]
#             for corr in correlated_list:
#                 day_feats.append(corr[i + offset])
#             features.extend(day_feats)
#         X.append(features)
#         y.append(target[i + window_size])
#     return np.array(X), np.array(y)

# def show():
#     st.title("Alineamiento y Previsión con Random Forest (GridSearch + TimeSeriesSplit)")
#     seed = st.number_input("Semilla para selección de acciones (fijo)", value=42, min_value=0, max_value=9999, step=1)
#     n_stocks = st.slider("Número de acciones a analizar", 10, 50, 20)
#     dias_historico = st.slider("Ventana de días para comparar", 100, 1000, 750)
#     max_lag = st.slider("Máximo desfase (lag, en días)", 5, 30, 25)

#     random.seed(seed)
#     np.random.seed(seed)

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

#     # --- FILTRO POR CORRELACIÓN --- (ajusta aquí)
#     CORR_THRESHOLD = 0.7
#     top_corr = [r for r in top7 if abs(r["corr_best"]) >= CORR_THRESHOLD]
#     st.markdown(
#         f"**Solo se usan en el modelo las correlacionadas con |correlación máxima (con lag)| ≥ {CORR_THRESHOLD}. "
#         f"Seleccionadas: {len(top_corr)} de 7 posibles.**"
#     )

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

#     st.markdown("## Previsión de la Acción Objetivo usando Random Forest + tuning")
#     dias_pred = st.slider("¿Cuántos días futuros predecir? (test set)", 1, 15, 7)
#     window_size = 30

#     target_full = data[action_selected].iloc[-dias_historico - window_size:].values
#     min_len = len(target_full)
#     correlated_full = []
#     for r in top_corr:
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

#     # --- Normalización robusta usando solo el set de entrenamiento
#     train_idx = -(dias_pred + 30) if (dias_pred + 30) < len(target_full) else 0
#     mu_y = np.mean(target_full[:train_idx])
#     std_y = np.std(target_full[:train_idx]) if np.std(target_full[:train_idx]) != 0 else 1
#     target_full_norm = (target_full - mu_y) / std_y
#     correlated_full_norm = []
#     for corr in correlated_full:
#         mu_corr = np.mean(corr[:train_idx])
#         std_corr = np.std(corr[:train_idx]) if np.std(corr[:train_idx]) != 0 else 1
#         correlated_full_norm.append((corr - mu_corr) / std_corr)

#     # --- Crear ventanas multivariadas (target + correlacionadas como features)
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

#     # --- GridSearchCV con TimeSeriesSplit para robustez ---
#     param_grid = {
#         "n_estimators": [50, 100, 200],
#         "max_depth": [3, 5, 10, None],
#         "min_samples_split": [2, 5],
#         "min_samples_leaf": [1, 2, 4],
#     }
#     tscv = TimeSeriesSplit(n_splits=4)
#     rf = RandomForestRegressor(random_state=seed)
#     grid = GridSearchCV(rf, param_grid, cv=tscv, scoring="neg_mean_squared_error", n_jobs=-1)
#     with st.spinner("Ajustando hiperparámetros (puede tardar un poco)..."):
#         grid.fit(X_train, y_train)
#     best_rf = grid.best_estimator_

#     st.success(f"Mejores hiperparámetros: {grid.best_params_}")

#     # --- Validación clásica (no autoregresiva)
#     split = int(0.7 * len(X_train))
#     X_tr, X_val = X_train[:split], X_train[split:]
#     y_tr, y_val = y_train[:split], y_train[split:]
#     best_rf.fit(X_tr, y_tr)
#     y_val_pred = best_rf.predict(X_val)
#     mse = mean_squared_error(y_val, y_val_pred)
#     r2_val = r2_score(y_val, y_val_pred)
#     r2_val_disp = f"{r2_val:.2f}" if r2_val >= 0 else "—"

#     # --- Previsión multi-step autoregresiva (realista)
#     ult_idx = -dias_pred
#     last_window = X_all[ult_idx - 1].copy()  # shape (window * n_feats,)
#     y_pred_ar = []
#     for step in range(dias_pred):
#         pred_norm = best_rf.predict(last_window.reshape(1, -1))[0]
#         y_pred_ar.append(pred_norm)
#         # Shift ventana: actualiza solo target, correlacionadas se mantienen fijas
#         next_window = np.roll(last_window, -1)
#         next_window[(window_size - 1) * (1 + len(correlated_full))] = pred_norm
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
#         mode='lines+markers', name="Previsto RF autoregresivo", line=dict(color='orange')
#     ))
#     fig_pred.add_shape(type="line",
#         x0=x_pred[0] - 0.5, y0=min(zscore_denormalize(y_all[-dias_graf:], mu_y, std_y)),
#         x1=x_pred[0] - 0.5, y1=max(zscore_denormalize(y_all[-dias_graf:], mu_y, std_y)),
#         line=dict(color="gray", width=1, dash="dot"))
#     fig_pred.update_layout(
#         title=f"Previsión Random Forest Multi-step Autoregresivo ({dias_pred} días + 15 días extra, ventana=30)",
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
#         - **Naranja:** previsión Random Forest autoregresivo.
#         - Línea punteada gris: inicio de la previsión.
#         - Entradas futuras usan solo valores previstos, nunca reales.
#         - Cada ventana contiene la acción objetivo y las correlacionadas con |corr|≥{CORR_THRESHOLD} (alineadas y normalizadas).
#         - Tuning automático con GridSearchCV + TimeSeriesSplit.
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
#     # output: (samples, window, features)
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
#     st.title("Alineamiento y Previsión con LSTM autoregresivo (multi-step)")
#     seed = st.number_input("Semilla para selección de acciones (fijo)", value=42, min_value=0, max_value=9999, step=1)
#     n_stocks = st.slider("Número de acciones a analizar", 10, 50, 20)
#     dias_historico = st.slider("Ventana de días para comparar", 100, 1000, 750)
#     max_lag = st.slider("Máximo desfase (lag, en días)", 5, 30, 25)

#     # Fijar semilla para reproducibilidad en LSTM
#     random.seed(seed)
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

#     # --- FILTRO POR CORRELACIÓN --- (ajusta aquí si lo deseas)
#     CORR_THRESHOLD = 0.7
#     top_corr = [r for r in top7 if abs(r["corr_best"]) >= CORR_THRESHOLD]
#     st.markdown(
#         f"**Solo se usan en el modelo las correlacionadas con |correlación máxima (con lag)| ≥ {CORR_THRESHOLD}. "
#         f"Seleccionadas: {len(top_corr)} de 7 posibles.**"
#     )

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
#     for r in top_corr:
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

#     # --- Normalización robusta solo con entrenamiento
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

#     # --- Modelo LSTM robusto
#     n_feats = X_train.shape[2]
#     model = Sequential([
#         LSTM(64, input_shape=(window_size, n_feats), return_sequences=False),
#         Dropout(0.3),
#         Dense(32, activation='relu'),
#         Dense(1)
#     ])
#     model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss='mse')
#     es = EarlyStopping(monitor='val_loss', patience=8, restore_best_weights=True)
#     model.fit(X_tr, y_tr, validation_data=(X_val, y_val), epochs=100, batch_size=16, verbose=0, callbacks=[es])

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
#         pred_norm = model.predict(last_window[np.newaxis, ...], verbose=0)[0, 0]
#         y_pred_ar.append(pred_norm)
#         # Shift ventana: solo target se actualiza, correlacionadas se mantienen fijas en test
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
#         title=f"Previsión LSTM Multi-step Autoregresivo ({dias_pred} días + 15 días extra, ventana=30)",
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
#         - Cada ventana contiene la acción objetivo y las correlacionadas con |corr|≥{CORR_THRESHOLD} (alineadas y normalizadas).
#         - LSTM robusta, entrenada con EarlyStopping y batch de 16.
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
#     st.title("Alineamiento y Previsión con LSTM autoregresivo (multi-step) recortando últimos 15 datos")
#     seed = st.number_input("Semilla para selección de acciones (fijo)", value=42, min_value=0, max_value=9999, step=1)
#     n_stocks = st.slider("Número de acciones a analizar", 10, 50, 20)
#     dias_historico = st.slider("Ventana de días para comparar", 100, 1000, 750)
#     max_lag = st.slider("Máximo desfase (lag, en días)", 5, 30, 25)

#     # Fijar semilla para reproducibilidad en LSTM
#     random.seed(seed)
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

#     CORR_THRESHOLD = 0.7
#     top_corr = [r for r in top7 if abs(r["corr_best"]) >= CORR_THRESHOLD]
#     st.markdown(
#         f"**Solo se usan en el modelo las correlacionadas con |correlación máxima (con lag)| ≥ {CORR_THRESHOLD}. "
#         f"Seleccionadas: {len(top_corr)} de 7 posibles.**"
#     )

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

#     st.markdown("## Previsión de la Acción Objetivo usando LSTM autoregresivo + normalización (recortando últimos 15 datos)")
#     dias_pred = st.slider("¿Cuántos días futuros predecir? (test set)", 1, 15, 7)
#     window_size = 30

#     # Preparamos los datos para normalización y ventanas
#     target_full = data[action_selected].iloc[-dias_historico - window_size:].values
#     min_len = len(target_full)
#     correlated_full = []
#     for r in top_corr:
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

#     # -------- RECORTA LOS ÚLTIMOS 15 DATOS DE TODAS LAS SERIES ---------
#     n_cut = 15
#     if len(target_full) > n_cut:
#         target_full = target_full[:-n_cut]
#         correlated_full = [c[:-n_cut] for c in correlated_full]

#     # --- Normalización robusta solo con entrenamiento
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

#     # --- Modelo LSTM robusto
#     n_feats = X_train.shape[2]
#     model = Sequential([
#         LSTM(64, input_shape=(window_size, n_feats), return_sequences=False),
#         Dropout(0.3),
#         Dense(32, activation='relu'),
#         Dense(1)
#     ])
#     model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss='mse')
#     es = EarlyStopping(monitor='val_loss', patience=8, restore_best_weights=True)
#     model.fit(X_tr, y_tr, validation_data=(X_val, y_val), epochs=100, batch_size=16, verbose=0, callbacks=[es])

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
#         pred_norm = model.predict(last_window[np.newaxis, ...], verbose=0)[0, 0]
#         y_pred_ar.append(pred_norm)
#         # Shift ventana: solo target se actualiza, correlacionadas se mantienen fijas en test
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
#         title=f"Previsión LSTM Multi-step Autoregresivo ({dias_pred} días + 15 días extra, ventana=30)",
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
#         - Cada ventana contiene la acción objetivo y las correlacionadas con |corr|≥{CORR_THRESHOLD} (alineadas y normalizadas).
#         - LSTM robusta, entrenada con EarlyStopping y batch de 16.
#         - *Se han recortado los últimos {n_cut} datos de todas las series antes de modelar*.
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
#     st.title("Previsión con LSTM y Validación Temporal Realista")
#     seed = st.number_input("Semilla para selección de acciones (fijo)", value=42, min_value=0, max_value=9999, step=1)
#     n_stocks = st.slider("Número de acciones a analizar", 10, 50, 20)
#     dias_historico = st.slider("Ventana de días para comparar", 100, 1000, 750)
#     max_lag = st.slider("Máximo desfase (lag, en días)", 5, 30, 25)

#     np.random.seed(seed)
#     tf.random.set_seed(seed)
#     random.seed(seed)
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

#     st.markdown("## Previsión usando LSTM con validación temporal")

#     dias_pred = st.slider("¿Cuántos días futuros predecir? (test set)", 1, 15, 7)
#     window_size = 30

#     # Eliminar últimos 15 datos si se desea simular un escenario realista
#     N_CUT = 15
#     target_full = data[action_selected].iloc[-dias_historico - window_size:-N_CUT].values if dias_historico + window_size + N_CUT <= len(data) else data[action_selected].iloc[-dias_historico - window_size:].values
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

#     # Normalización robusta usando solo el set de entrenamiento
#     # Reserva última parte para val+test
#     total_samples = len(target_full) - window_size
#     dias_val = max(10, int(0.15 * total_samples))

#     # Asegurar que hay suficiente data
#     if total_samples < dias_pred + dias_val + 30:
#         st.warning("No hay suficientes datos para la ventana seleccionada. Ajusta los parámetros.")
#         return

#     train_idx = total_samples - (dias_val + dias_pred)
#     mu_y = np.mean(target_full[:train_idx + window_size])
#     std_y = np.std(target_full[:train_idx + window_size]) if np.std(target_full[:train_idx + window_size]) != 0 else 1
#     target_full_norm = (target_full - mu_y) / std_y
#     correlated_full_norm = []
#     for corr in correlated_full:
#         mu_corr = np.mean(corr[:train_idx + window_size])
#         std_corr = np.std(corr[:train_idx + window_size]) if np.std(corr[:train_idx + window_size]) != 0 else 1
#         correlated_full_norm.append((corr - mu_corr) / std_corr)

#     X_all, y_all = create_sliding_window_multivariate(target_full_norm, correlated_full_norm, window_size=window_size)
#     total_samples = len(y_all)

#     # División temporal
#     X_train = X_all[:train_idx]
#     y_train = y_all[:train_idx]
#     X_val = X_all[train_idx:train_idx + dias_val]
#     y_val = y_all[train_idx:train_idx + dias_val]
#     X_test = X_all[-dias_pred:]
#     y_test = y_all[-dias_pred:]

#     # Modelo LSTM
#     n_feats = X_train.shape[2]
#     model = Sequential([
#         LSTM(64, input_shape=(window_size, n_feats), return_sequences=False),
#         Dropout(0.3),
#         Dense(32, activation='relu'),
#         Dense(1)
#     ])
#     model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss='mse')
#     es = EarlyStopping(monitor='val_loss', patience=8, restore_best_weights=True)
#     model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=100, batch_size=16, verbose=0, callbacks=[es])

#     # Métricas de validación
#     y_val_pred = model.predict(X_val, verbose=0).flatten()
#     mse_val = mean_squared_error(y_val, y_val_pred)
#     r2_val = r2_score(y_val, y_val_pred)
#     r2_val_disp = f"{r2_val:.2f}" if r2_val >= 0 else "—"

#     # Previsión multi-step autoregresiva en test
#     ult_idx = -dias_pred
#     last_window = X_all[ult_idx - 1].copy()
#     y_pred_ar = []
#     for step in range(dias_pred):
#         pred_norm = model.predict(last_window[np.newaxis, ...], verbose=0)[0, 0]
#         y_pred_ar.append(pred_norm)
#         # Actualiza solo target
#         next_window = np.roll(last_window, -1, axis=0)
#         next_window[-1, 0] = pred_norm
#         last_window = next_window

#     y_pred_ar_real = zscore_denormalize(np.array(y_pred_ar), mu_y, std_y)
#     y_test_real = zscore_denormalize(y_test, mu_y, std_y)
#     mse_test = mean_squared_error(y_test_real, y_pred_ar_real)
#     r2_test = r2_score(y_test_real, y_pred_ar_real)
#     r2_test_disp = f"{r2_test:.2f}" if r2_test >= 0 else "—"

#     x_real = np.arange(total_samples - (dias_val + dias_pred + 15), total_samples)
#     x_pred = np.arange(total_samples - dias_pred, total_samples)

#     fig_pred = go.Figure()
#     fig_pred.add_trace(go.Scatter(
#         x=x_real, y=zscore_denormalize(y_all[-(dias_val + dias_pred + 15):-dias_pred], mu_y, std_y),
#         mode='lines+markers', name="Real (train/val)", line=dict(color='blue')
#     ))
#     fig_pred.add_trace(go.Scatter(
#         x=x_pred, y=y_pred_ar_real,
#         mode='lines+markers', name="Previsto LSTM autoregresivo", line=dict(color='orange')
#     ))
#     fig_pred.add_trace(go.Scatter(
#         x=x_pred, y=y_test_real,
#         mode='lines+markers', name="Real (test)", line=dict(color='green', dash='dash')
#     ))
#     fig_pred.add_shape(type="line",
#         x0=x_pred[0] - 0.5, y0=min(zscore_denormalize(y_all[-(dias_val + dias_pred + 15):], mu_y, std_y)),
#         x1=x_pred[0] - 0.5, y1=max(zscore_denormalize(y_all[-(dias_val + dias_pred + 15):], mu_y, std_y)),
#         line=dict(color="gray", width=1, dash="dot"))
#     fig_pred.update_layout(
#         title=f"Previsión LSTM Multi-step Autoregresivo VALIDACIÓN TEMPORAL ({dias_pred} días test, {dias_val} val, ventana=30)",
#         xaxis_title="Día",
#         yaxis_title="Precio",
#         height=450,
#         showlegend=True
#     )

#     st.markdown(
#         f"""**MSE validación:** {mse_val:.2f} | **R² validación:** {r2_val_disp}  
# **MSE test (autoregresivo):** {mse_test:.2f} | **R² test:** {r2_test_disp}"""
#     )
#     st.plotly_chart(fig_pred, use_container_width=True)
#     st.markdown(
#         f"""
#         - **Azul:** valores reales en entrenamiento/validación.
#         - **Naranja:** previsión LSTM autoregresivo puro.
#         - **Verde dashed:** reales en test.
#         - Línea punteada gris: inicio de la previsión.
#         - Split temporal: train – val – test en ese orden, sin usar datos del futuro para entrenar.
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
#     st.title("Previsión con LSTM y Validación Temporal Realista")
#     seed = st.number_input("Semilla para selección de acciones (fijo)", value=42, min_value=0, max_value=9999, step=1)
#     n_stocks = st.slider("Número de acciones a analizar", 10, 50, 20)
#     dias_historico = st.slider("Ventana de días para comparar", 100, 1000, 750)
#     max_lag = st.slider("Máximo desfase (lag, en días)", 5, 30, 25)

#     np.random.seed(seed)
#     tf.random.set_seed(seed)
#     random.seed(seed)
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

#     st.markdown("## Previsión usando LSTM con validación temporal")

#     dias_pred = st.slider("¿Cuántos días futuros predecir? (test set)", 1, 15, 7)
#     window_size = 30

#     # División temporal realista: train (70%), val (15%), test (15%)
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

#     total_samples = len(target_full) - window_size
#     val_size = max(10, int(0.15 * total_samples))
#     test_size = dias_pred  # siempre = días predicción del slider
#     train_size = total_samples - val_size - test_size

#     if train_size <= 0 or val_size <= 0 or total_samples < (test_size + val_size + 30):
#         st.warning("No hay suficientes datos para la ventana seleccionada. Ajusta los parámetros.")
#         return

#     mu_y = np.mean(target_full[:train_size + window_size])
#     std_y = np.std(target_full[:train_size + window_size]) if np.std(target_full[:train_size + window_size]) != 0 else 1
#     target_full_norm = (target_full - mu_y) / std_y
#     correlated_full_norm = []
#     for corr in correlated_full:
#         mu_corr = np.mean(corr[:train_size + window_size])
#         std_corr = np.std(corr[:train_size + window_size]) if np.std(corr[:train_size + window_size]) != 0 else 1
#         correlated_full_norm.append((corr - mu_corr) / std_corr)

#     X_all, y_all = create_sliding_window_multivariate(target_full_norm, correlated_full_norm, window_size=window_size)

#     # Real temporal split
#     X_train = X_all[:train_size]
#     y_train = y_all[:train_size]
#     X_val = X_all[train_size:train_size + val_size]
#     y_val = y_all[train_size:train_size + val_size]
#     X_test = X_all[-test_size:]
#     y_test = y_all[-test_size:]

#     # Modelo LSTM
#     n_feats = X_train.shape[2]
#     model = Sequential([
#         LSTM(64, input_shape=(window_size, n_feats), return_sequences=False),
#         Dropout(0.3),
#         Dense(32, activation='relu'),
#         Dense(1)
#     ])
#     model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss='mse')
#     es = EarlyStopping(monitor='val_loss', patience=8, restore_best_weights=True)
#     model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=100, batch_size=16, verbose=0, callbacks=[es])

#     # Validación
#     y_val_pred = model.predict(X_val, verbose=0).flatten()
#     mse_val = mean_squared_error(y_val, y_val_pred)
#     r2_val = r2_score(y_val, y_val_pred)
#     r2_val_disp = f"{r2_val:.2f}" if r2_val >= 0 else "—"

#     # Test autoregresivo multi-step
#     ult_idx = -test_size
#     last_window = X_all[ult_idx - 1].copy()
#     y_pred_ar = []
#     for step in range(test_size):
#         pred_norm = model.predict(last_window[np.newaxis, ...], verbose=0)[0, 0]
#         y_pred_ar.append(pred_norm)
#         next_window = np.roll(last_window, -1, axis=0)
#         next_window[-1, 0] = pred_norm
#         last_window = next_window

#     y_pred_ar_real = zscore_denormalize(np.array(y_pred_ar), mu_y, std_y)
#     y_test_real = zscore_denormalize(y_test, mu_y, std_y)
#     mse_test = mean_squared_error(y_test_real, y_pred_ar_real)
#     r2_test = r2_score(y_test_real, y_pred_ar_real)
#     r2_test_disp = f"{r2_test:.2f}" if r2_test >= 0 else "—"

#     x_real = np.arange(len(y_all) - (val_size + test_size + 15), len(y_all))
#     x_pred = np.arange(len(y_all) - test_size, len(y_all))

#     fig_pred = go.Figure()
#     fig_pred.add_trace(go.Scatter(
#         x=x_real, y=zscore_denormalize(y_all[-(val_size + test_size + 15):-test_size], mu_y, std_y),
#         mode='lines+markers', name="Real (train/val)", line=dict(color='blue')
#     ))
#     fig_pred.add_trace(go.Scatter(
#         x=x_pred, y=y_pred_ar_real,
#         mode='lines+markers', name="Previsto LSTM autoregresivo", line=dict(color='orange')
#     ))
#     fig_pred.add_trace(go.Scatter(
#         x=x_pred, y=y_test_real,
#         mode='lines+markers', name="Real (test)", line=dict(color='green', dash='dash')
#     ))
#     fig_pred.add_shape(type="line",
#         x0=x_pred[0] - 0.5, y0=min(zscore_denormalize(y_all[-(val_size + test_size + 15):], mu_y, std_y)),
#         x1=x_pred[0] - 0.5, y1=max(zscore_denormalize(y_all[-(val_size + test_size + 15):], mu_y, std_y)),
#         line=dict(color="gray", width=1, dash="dot"))
#     fig_pred.update_layout(
#         title=f"Previsión LSTM Multi-step Autoregresivo VALIDACIÓN TEMPORAL ({test_size} días test, {val_size} val, ventana=30)",
#         xaxis_title="Día",
#         yaxis_title="Precio",
#         height=450,
#         showlegend=True
#     )

#     st.markdown(
#         f"""**MSE validación:** {mse_val:.2f} | **R² validación:** {r2_val_disp}  
# **MSE test (autoregresivo):** {mse_test:.2f} | **R² test:** {r2_test_disp}"""
#     )
#     st.plotly_chart(fig_pred, use_container_width=True)
#     st.markdown(
#         f"""
#         - **Azul:** valores reales en entrenamiento/validación.
#         - **Naranja:** previsión LSTM autoregresivo puro.
#         - **Verde dashed:** reales en test.
#         - Línea punteada gris: inicio de la previsión.
#         - Split temporal: train – val – test en ese orden, sin usar datos del futuro para entrenar.
#         """
#     )

# if __name__ == "__main__":
#     show()












# # menu/exploracao_ajustado.py
# import streamlit as st
# import pandas as pd
# import numpy as np
# import yfinance as yf
# import random
# import plotly.graph_objs as go
# from sklearn.ensemble import RandomForestRegressor
# from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit
# from sklearn.metrics import mean_squared_error, r2_score
# from typing import List, Dict, Any, Tuple

# # --- Constantes y Configuración ---
# TICKERS_LIST = [
#     "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA", "BRK-B", "JPM", "V",
#     "UNH", "JNJ", "XOM", "PG", "MA", "HD", "CVX", "MRK", "ABBV", "PEP",
#     "LLY", "KO", "AVGO", "COST", "WMT", "DIS", "BAC", "MCD", "ADBE", "PFE",
#     "TMO", "CSCO", "DHR", "ABT", "ACN", "CRM", "TXN", "NKE", "VZ", "NEE",
#     "CMCSA", "WFC", "LIN", "INTC", "QCOM", "MS", "HON", "PM", "UNP", "AMGN",
#     "SBUX", "ORCL", "AMD", "CVS", "ISRG", "GILD", "BKNG", "NOW", "ZTS", "MDLZ"
# ]

# # --- Funciones de Datos ---

# def init_session_state(n_stocks: int, seed: int):
#     """Inicializa o actualiza los tickers en el estado de la sesión."""
#     if 'tickers' not in st.session_state or st.session_state.get('tickers_seed') != seed or st.session_state.get('tickers_n') != n_stocks:
#         random.seed(seed)
#         st.session_state['tickers'] = random.sample(TICKERS_LIST, n_stocks)
#         st.session_state['tickers_seed'] = seed
#         st.session_state['tickers_n'] = n_stocks

# @st.cache_data(show_spinner="Descargando datos de acciones...")
# def download_stocks(tickers: List[str], days: int) -> pd.DataFrame:
#     """Descarga datos históricos de cierre ajustado para una lista de tickers."""
#     period = days + 100  # Búfer para asegurar suficientes datos
#     df = yf.download(tickers, period=f"{period}d", interval="1d", progress=False)
#     data = df['Adj Close'] if 'Adj Close' in df else df['Close']
#     return data

# def get_valid_data(tickers: List[str], days: int) -> pd.DataFrame:
#     """Filtra y limpia los datos para asegurar que no haya NaNs y tengan la longitud requerida."""
#     data = download_stocks(tickers, days)
#     min_len = days + 50
    
#     if isinstance(data, pd.Series):
#         data = data.to_frame(tickers[0])
        
#     valid_cols = [c for c in data.columns if data[c].dropna().shape[0] >= min_len]
#     if not valid_cols:
#         return pd.DataFrame()

#     data_valid = data[valid_cols].iloc[-min_len:]
#     data_valid = data_valid.dropna(axis=1)
#     return data_valid

# # --- Funciones de Análisis y Modelado ---

# def create_sliding_window_features(target: np.ndarray, window_size: int) -> Tuple[np.ndarray, np.ndarray]:
#     """Crea características y etiquetas de una serie temporal para un modelo autorregresivo."""
#     num_samples = len(target) - window_size
#     X = np.array([target[i:i + window_size] for i in range(num_samples)])
#     y = np.array([target[i + window_size] for i in range(num_samples)])
#     return X, y

# def perform_autoregressive_forecast(model, initial_window: np.ndarray, steps: int) -> np.ndarray:
#     """Realiza un pronóstico autorregresivo multi-step."""
#     forecast = []
#     current_window = initial_window.copy()
#     for _ in range(steps):
#         pred = model.predict(current_window.reshape(1, -1))[0]
#         forecast.append(pred)
#         current_window = np.roll(current_window, -1)
#         current_window[-1] = pred
#     return np.array(forecast)

# def train_and_evaluate_model(
#     target_series: np.ndarray,
#     window_size: int,
#     pred_days: int,
#     seed: int
# ) -> Dict[str, Any]:
#     """
#     Función completa para entrenar, optimizar automáticamente y evaluar el modelo.
#     """
#     # 1. Crear dataset
#     X_all, y_all = create_sliding_window_features(target_series, window_size)
    
#     if len(y_all) < pred_days * 2 + window_size:
#         return {"error": "No hay suficientes datos para la ventana y días de predicción seleccionados."}

#     # 2. División Walk-Forward
#     X_train, y_train = X_all[:-2 * pred_days], y_all[:-2 * pred_days]
#     X_val, y_val = X_all[-2 * pred_days:-pred_days], y_all[-2 * pred_days:-pred_days]
#     X_test, y_test = X_all[-pred_days:], y_all[-pred_days:]

#     # 3. ### MEJORA: Definición de la parrilla de búsqueda de hiperparámetros ###
#     param_dist = {
#         "n_estimators": [50, 100, 150, 200, 250, 350, 500, 700, 900],
#         "max_depth": [None, 10, 20, 30],
#         "min_samples_leaf": [1, 2, 4, 6, 8, 10],
#         "min_samples_split": [2, 5, 10]
#     }
#     tscv = TimeSeriesSplit(n_splits=3)
    
#     # ### MEJORA: Instancia de RandomizedSearchCV para el ajuste automático ###
#     random_search = RandomizedSearchCV(
#         RandomForestRegressor(random_state=seed),
#         param_distributions=param_dist,
#         n_iter=10,  # Número de combinaciones a probar
#         cv=tscv,
#         scoring="neg_mean_squared_error",
#         n_jobs=-1,
#         random_state=seed
#     )
#     random_search.fit(X_train, y_train)
    
#     # El mejor modelo encontrado
#     rf_best = random_search.best_estimator_
#     best_params = random_search.best_params_

#     # 4. Evaluación Walk-Forward en el set de Prueba
#     y_test_pred = perform_autoregressive_forecast(rf_best, X_test[0], pred_days)
#     mse_test = mean_squared_error(y_test, y_test_pred)
#     r2_test = r2_score(y_test, y_test_pred)

#     # 5. Evaluación contra un baseline
#     last_val_before_test = y_all[-pred_days - 1]
#     persistence_preds = np.insert(y_test[:-1], 0, last_val_before_test)
#     mse_persistence = mean_squared_error(y_test, persistence_preds)

#     return {
#         "error": None,
#         "best_params": best_params,
#         "predictions": y_test_pred,
#         "ground_truth_series": y_all,
#         "metrics": {
#             "mse_test": mse_test, "r2_test": r2_test,
#             "mse_persistence": mse_persistence
#         },
#         "params": {
#             "window_size": window_size,
#             "pred_days": pred_days,
#             "total_samples": len(y_all)
#         }
#     }

# # --- Funciones de Visualización ---

# def plot_forecast(results: Dict[str, Any]):
#     """Crea y muestra el gráfico de pronóstico con Plotly."""
#     y_all = results["ground_truth_series"]
#     y_pred = results["predictions"]
#     total_samples = results["params"]["total_samples"]
#     pred_days = results["params"]["pred_days"]
#     window_size = results["params"]["window_size"]
#     dias_graf = pred_days + 40

#     x_real = np.arange(total_samples - dias_graf, total_samples)
#     x_pred = x_real[-pred_days:]
    
#     fig = go.Figure()
#     fig.add_trace(go.Scatter(x=x_real, y=y_all[-dias_graf:], mode='lines+markers', name="Real", line=dict(color='blue')))
#     fig.add_trace(go.Scatter(x=x_pred, y=y_pred, mode='lines+markers', name="Previsto (RF Auto-ajustado)", line=dict(color='orange')))
#     fig.add_shape(type="line", x0=x_pred[0] - 0.5, y0=min(y_all[-dias_graf:]), x1=x_pred[0] - 0.5, y1=max(y_all[-dias_graf:]), line=dict(color="gray", width=1, dash="dot"))
    
#     fig.update_layout(title=f"Previsión Random Forest ({pred_days} días, ventana={window_size})", xaxis_title="Día", yaxis_title="Precio", height=450, legend=dict(x=0.01, y=0.99))
#     st.plotly_chart(fig, use_container_width=True)

# # --- Aplicación Principal de Streamlit ---

# def show():
#     """Función principal que organiza la interfaz y el flujo de la aplicación."""
#     # NO incluir st.set_page_config() aquí
    
#     st.title("🤖 Previsión de Acciones con Auto-ajuste de Modelo")
#     st.write("Pronóstico autorregresivo usando un modelo Random Forest con hiperparámetros optimizados automáticamente.")

#     # --- Panel de Configuración ---
#     with st.sidebar:
#         st.header("⚙️ Configuración General")
#         seed = st.number_input("Semilla para aleatoriedad", value=42, min_value=0)
#         n_stocks = st.slider("Número de acciones a descargar", 10, len(TICKERS_LIST), 20)
#         dias_historico = st.slider("Días de historia a descargar", 200, 2000, 800)

#     # --- Carga y Selección de Datos ---
#     init_session_state(n_stocks, seed)
#     data = get_valid_data(st.session_state['tickers'], dias_historico)

#     if data.empty:
#         st.error("No se pudieron descargar datos válidos. Prueba con otra semilla o aumenta los días de historia.")
#         return

#     st.header("1. Selección de la Acción Objetivo")
#     action_selected = st.selectbox("Elige la acción que quieres predecir:", options=data.columns, index=0)
    
#     with st.expander("Ver datos brutos de las acciones"):
#         st.dataframe(data.head())
        
#     # --- Sección de Previsión ---
#     st.header(f"2. Previsión para {action_selected}")

#     with st.container(border=True):
#         st.subheader("Configuración del Pronóstico")
#         pred_days = st.slider("¿Cuántos días futuros predecir?", 1, 30, 7, key="pred_days")
#         window_size = st.slider("Tamaño de la ventana (historia para predecir)", 10, 90, 30, key="window_size")
    
#     # --- Ejecutar Modelo y Mostrar Resultados ---
#     target_series = data[action_selected].dropna().values
    
#     with st.spinner("Buscando los mejores parámetros y entrenando el modelo..."):
#         results = train_and_evaluate_model(target_series, window_size, pred_days, seed)

#     if results.get("error"):
#         st.warning(results["error"])
#     else:
#         st.subheader("📈 Gráfico de Previsión vs. Real")
#         plot_forecast(results)

#         st.subheader("📊 Métricas de Rendimiento del Modelo (en Test)")
#         m = results["metrics"]
        
#         col1, col2, col3 = st.columns(3)
#         col1.metric(label="MSE (Modelo Auto-ajustado)", value=f"{m['mse_test']:.2f}", help="Error Cuadrático Medio. Más bajo es mejor.")
#         col2.metric(label="MSE (Baseline)", value=f"{m['mse_persistence']:.2f}", help="Error del modelo simple de persistencia.")
#         r2_test_disp = f"{m['r2_test']:.2f}" if m['r2_test'] >= -1 else "—"
#         col3.metric(label="R² Score", value=r2_test_disp, help="Coeficiente de determinación. Más cercano a 1 es mejor.")
        
#         if m['mse_test'] < m['mse_persistence']:
#             st.success("¡Éxito! El modelo auto-ajustado superó al baseline de persistencia.")
#         else:
#             st.warning("El modelo auto-ajustado no superó al baseline. Considera cambiar la configuración.")

#         # ### MEJORA: Mostrar el mejor ajuste de parámetros encontrado ###
#         st.subheader("🏆 Mejor Ajuste de Hiperparámetros Encontrado")
#         st.info("Estos son los parámetros del modelo Random Forest que produjeron los mejores resultados durante la búsqueda automática.")
#         st.json(results['best_params'])


# if __name__ == "__main__":
#     show()















# menu/exploracao_ajustado.py
import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import random
import plotly.graph_objs as go
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit
from sklearn.metrics import mean_squared_error, r2_score
from typing import List, Dict, Any, Tuple

TICKERS_LIST = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA", "BRK-B", "JPM", "V",
    "UNH", "JNJ", "XOM", "PG", "MA", "HD", "CVX", "MRK", "ABBV", "PEP",
    "LLY", "KO", "AVGO", "COST", "WMT", "DIS", "BAC", "MCD", "ADBE", "PFE",
    "TMO", "CSCO", "DHR", "ABT", "ACN", "CRM", "TXN", "NKE", "VZ", "NEE",
    "CMCSA", "WFC", "LIN", "INTC", "QCOM", "MS", "HON", "PM", "UNP", "AMGN",
    "SBUX", "ORCL", "AMD", "CVS", "ISRG", "GILD", "BKNG", "NOW", "ZTS", "MDLZ"
]

def init_session_state(n_stocks: int, seed: int):
    if 'tickers' not in st.session_state or st.session_state.get('tickers_seed') != seed or st.session_state.get('tickers_n') != n_stocks:
        random.seed(seed)
        st.session_state['tickers'] = random.sample(TICKERS_LIST, n_stocks)
        st.session_state['tickers_seed'] = seed
        st.session_state['tickers_n'] = n_stocks

@st.cache_data(show_spinner="Descargando datos de acciones...")
def download_stocks(tickers: List[str], days: int) -> pd.DataFrame:
    period = days + 100
    df = yf.download(tickers, period=f"{period}d", interval="1d", progress=False)
    data = df['Adj Close'] if 'Adj Close' in df else df['Close']
    return data

def get_valid_data(tickers: List[str], days: int) -> pd.DataFrame:
    data = download_stocks(tickers, days)
    min_len = days + 50
    if isinstance(data, pd.Series):
        data = data.to_frame(tickers[0])
    valid_cols = [c for c in data.columns if data[c].dropna().shape[0] >= min_len]
    if not valid_cols:
        return pd.DataFrame()
    data_valid = data[valid_cols].iloc[-min_len:]
    data_valid = data_valid.dropna(axis=1)
    return data_valid

def create_sliding_window_features(target: np.ndarray, window_size: int) -> Tuple[np.ndarray, np.ndarray]:
    num_samples = len(target) - window_size
    X = np.array([target[i:i + window_size] for i in range(num_samples)])
    y = np.array([target[i + window_size] for i in range(num_samples)])
    return X, y

def perform_autoregressive_forecast(model, initial_window: np.ndarray, steps: int) -> np.ndarray:
    forecast = []
    current_window = initial_window.copy()
    for _ in range(steps):
        pred = model.predict(current_window.reshape(1, -1))[0]
        forecast.append(pred)
        current_window = np.roll(current_window, -1)
        current_window[-1] = pred
    return np.array(forecast)

def train_and_evaluate_model(
    target_series: np.ndarray,
    window_size: int,
    pred_days: int,
    seed: int
) -> Dict[str, Any]:
    # 1. Crear dataset
    X_all, y_all = create_sliding_window_features(target_series, window_size)
    if len(y_all) < pred_days * 2 + window_size:
        return {"error": "No hay suficientes datos para la ventana y días de predicción seleccionados."}
    # 2. División Walk-Forward
    X_train, y_train = X_all[:-2 * pred_days], y_all[:-2 * pred_days]
    X_val, y_val = X_all[-2 * pred_days:-pred_days], y_all[-2 * pred_days:-pred_days]
    X_test, y_test = X_all[-pred_days:], y_all[-pred_days:]
    param_dist = {
        "n_estimators": [50, 100, 150, 200, 250, 350, 500, 700, 900],
        "max_depth": [None, 10, 20, 30],
        "min_samples_leaf": [1, 2, 4, 6, 8, 10],
        "min_samples_split": [2, 5, 10]
    }
    tscv = TimeSeriesSplit(n_splits=3)
    random_search = RandomizedSearchCV(
        RandomForestRegressor(random_state=seed),
        param_distributions=param_dist,
        n_iter=10,
        cv=tscv,
        scoring="neg_mean_squared_error",
        n_jobs=-1,
        random_state=seed
    )
    random_search.fit(X_train, y_train)
    rf_best = random_search.best_estimator_
    best_params = random_search.best_params_
    y_test_pred = perform_autoregressive_forecast(rf_best, X_test[0], pred_days)
    mse_test = mean_squared_error(y_test, y_test_pred)
    r2_test = r2_score(y_test, y_test_pred)
    last_val_before_test = y_all[-pred_days - 1]
    persistence_preds = np.insert(y_test[:-1], 0, last_val_before_test)
    mse_persistence = mean_squared_error(y_test, persistence_preds)
    return {
        "error": None,
        "best_params": best_params,
        "predictions": y_test_pred,
        "ground_truth_series": y_all,
        "metrics": {
            "mse_test": mse_test, "r2_test": r2_test,
            "mse_persistence": mse_persistence
        },
        "params": {
            "window_size": window_size,
            "pred_days": pred_days,
            "total_samples": len(y_all)
        }
    }

def plot_forecast(results: Dict[str, Any]):
    y_all = results["ground_truth_series"]
    y_pred = results["predictions"]
    total_samples = results["params"]["total_samples"]
    pred_days = results["params"]["pred_days"]
    window_size = results["params"]["window_size"]
    dias_graf = pred_days + 40
    x_real = np.arange(total_samples - dias_graf, total_samples)
    x_pred = x_real[-pred_days:]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x_real, y=y_all[-dias_graf:], mode='lines+markers', name="Real", line=dict(color='blue')))
    fig.add_trace(go.Scatter(x=x_pred, y=y_pred, mode='lines+markers', name="Previsto (RF Auto-ajustado)", line=dict(color='orange')))
    fig.add_shape(type="line", x0=x_pred[0] - 0.5, y0=min(y_all[-dias_graf:]), x1=x_pred[0] - 0.5, y1=max(y_all[-dias_graf:]), line=dict(color="gray", width=1, dash="dot"))
    fig.update_layout(title=f"Previsión Random Forest ({pred_days} días, ventana={window_size})", xaxis_title="Día", yaxis_title="Precio", height=450, legend=dict(x=0.01, y=0.99))
    st.plotly_chart(fig, use_container_width=True)

def show():
    st.title("🤖 Previsión de Acciones con Auto-ajuste de Modelo")
    st.write("Pronóstico autorregresivo usando un modelo Random Forest con hiperparámetros optimizados automáticamente.")

    with st.sidebar:
        st.header("⚙️ Configuración General")
        seed = st.number_input("Semilla para aleatoriedad", value=42, min_value=0)
        n_stocks = st.slider("Número de acciones a descargar", 10, len(TICKERS_LIST), 20)
        dias_historico = st.slider("Días de historia a descargar", 200, 2000, 800)
        # --- NUEVO: Selección de cuántos datos quieres eliminar del final ---
        n_drop = st.slider("Eliminar últimos N datos del dataset (para simular falta de info o testear escenarios)", 0, 40, 0, help="Elimina datos al final del dataset alineado antes de crear las ventanas.")

    init_session_state(n_stocks, seed)
    data = get_valid_data(st.session_state['tickers'], dias_historico)
    if data.empty:
        st.error("No se pudieron descargar datos válidos. Prueba con otra semilla o aumenta los días de historia.")
        return

    st.header("1. Selección de la Acción Objetivo")
    action_selected = st.selectbox("Elige la acción que quieres predecir:", options=data.columns, index=0)
    with st.expander("Ver datos brutos de las acciones"):
        st.dataframe(data.head())

    st.header(f"2. Previsión para {action_selected}")
    with st.container(border=True):
        st.subheader("Configuración del Pronóstico")
        pred_days = st.slider("¿Cuántos días futuros predecir?", 1, 30, 7, key="pred_days")
        window_size = st.slider("Tamaño de la ventana (historia para predecir)", 10, 90, 30, key="window_size")

    # --- Eliminación de datos finales (por ventana slider) ---
    target_series_full = data[action_selected].dropna().values
    if n_drop > 0:
        if n_drop + window_size + pred_days * 2 >= len(target_series_full):
            st.error("No hay suficientes datos tras eliminar. Reduce el número de datos a eliminar, el tamaño de ventana o los días de predicción.")
            return
        target_series = target_series_full[:-n_drop]
    else:
        target_series = target_series_full

    with st.spinner("Buscando los mejores parámetros y entrenando el modelo..."):
        results = train_and_evaluate_model(target_series, window_size, pred_days, seed)

    if results.get("error"):
        st.warning(results["error"])
    else:
        st.subheader("📈 Gráfico de Previsión vs. Real")
        plot_forecast(results)

        st.subheader("📊 Métricas de Rendimiento del Modelo (en Test)")
        m = results["metrics"]
        col1, col2, col3 = st.columns(3)
        col1.metric(label="MSE (Modelo Auto-ajustado)", value=f"{m['mse_test']:.2f}", help="Error Cuadrático Medio. Más bajo es mejor.")
        col2.metric(label="MSE (Baseline)", value=f"{m['mse_persistence']:.2f}", help="Error del modelo simple de persistencia.")
        r2_test_disp = f"{m['r2_test']:.2f}" if m['r2_test'] >= -1 else "—"
        col3.metric(label="R² Score", value=r2_test_disp, help="Coeficiente de determinación. Más cercano a 1 es mejor.")

        if m['mse_test'] < m['mse_persistence']:
            st.success("¡Éxito! El modelo auto-ajustado superó al baseline de persistencia.")
        else:
            st.warning("El modelo auto-ajustado no superó al baseline. Considera cambiar la configuración.")

        st.subheader("🏆 Mejor Ajuste de Hiperparámetros Encontrado")
        st.info("Estos son los parámetros del modelo Random Forest que produjeron los mejores resultados durante la búsqueda automática.")
        st.json(results['best_params'])

if __name__ == "__main__":
    show()
