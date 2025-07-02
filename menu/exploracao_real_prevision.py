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

#     # RELAJAR UMBRAL PARA NO QUEDARSE SOLO CON 1 o 0
#     CORR_THRESHOLD = 0.7  # <--- Ajusta aquí, antes era 0.9
#     top_corr = [r for r in top7 if abs(r["corr_best"]) >= CORR_THRESHOLD]

#     st.markdown(
#         f"**Solo se usan en el modelo las correlacionadas con |correlación máxima (con lag)| ≥ {CORR_THRESHOLD}. "
#         f"Seleccionadas: {len(top_corr)} de 7 posibles.**"
#     )

#     # Si hay menos de 2 features, advertir
#     if len(top_corr) < 2:
#         st.warning("¡Muy pocas acciones correlacionadas fuertes! Relaja el umbral o aumenta la cantidad de acciones analizadas.")
#         # Si quieres forzar el uso, puedes incluir aunque solo haya 1.

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
#     window_size = st.slider("Tamaño ventana (mayor puede ayudar)", 10, 60, 30)  # Permite jugar con ventana

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
#     last_window = X_all[ult_idx - 1].copy()
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
#         title=f"Previsión Random Forest Multi-step Autoregresivo ({dias_pred} días + 15 días extra, ventana={window_size})",
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
#         - Las correlacionadas se filtran por umbral de correlación (ajusta en `CORR_THRESHOLD`).
#         - Prueba aumentando tamaño de ventana o relajando el umbral si el modelo queda plano.
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
# from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
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
#     st.title("Correlación y Previsión de Series Temporales con Random Forest + Validación Walk-Forward")
#     st.write("Previsión multi-step autoregresiva realista y validación walk-forward.")
#     seed = st.number_input("Semilla para selección de acciones (fijo)", value=42, min_value=0, max_value=9999, step=1)
#     n_stocks = st.slider("Número de acciones a analizar", 10, 50, 20)
#     dias_historico = st.slider("Ventana de días para comparar", 100, 1000, 750)
#     max_lag = st.slider("Máximo desfase (lag, en días)", 5, 30, 25)

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

#     st.markdown("## Previsión de la Acción Objetivo usando Random Forest y validación walk-forward autoregresiva")
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

#     # --- División walk-forward: entrenamiento, validación autoregresiva, test autoregresivo ---
#     valid_steps = dias_pred  # Misma longitud para validar y testear
#     X_train = X_all[:-2*dias_pred]
#     y_train = y_all[:-2*dias_pred]
#     X_val = X_all[-2*dias_pred:-dias_pred]
#     y_val = y_all[-2*dias_pred:-dias_pred]
#     X_test = X_all[-dias_pred:]
#     y_test = y_all[-dias_pred:]

#     # --- Tuning rápido solo n_estimators
#     param_grid = {"n_estimators": [50, 100, 200]}
#     tscv = TimeSeriesSplit(n_splits=3)
#     grid = GridSearchCV(RandomForestRegressor(random_state=seed), param_grid, cv=tscv, n_jobs=-1, scoring="neg_mean_squared_error")
#     grid.fit(X_train, y_train)
#     rf = grid.best_estimator_

#     # --- Validación autoregresiva (rolling forecast) ---
#     y_val_ar = []
#     window = X_val[0].copy()
#     for step in range(valid_steps):
#         pred = rf.predict(window.reshape(1, -1))[0]
#         y_val_ar.append(pred)
#         # Actualiza ventana: target previsto, correlacionadas siguen fijas
#         window = np.roll(window, -1)
#         window[-1] = pred
#     mse_val = mean_squared_error(y_val, y_val_ar)
#     r2_val = r2_score(y_val, y_val_ar)
#     r2_val_disp = f"{r2_val:.2f}" if r2_val >= 0 else "—"

#     # --- Test autoregresivo ---
#     y_pred = []
#     window = X_test[0].copy()
#     for step in range(dias_pred):
#         pred = rf.predict(window.reshape(1, -1))[0]
#         y_pred.append(pred)
#         window = np.roll(window, -1)
#         window[-1] = pred
#     mse_test = mean_squared_error(y_test, y_pred)
#     r2_test = r2_score(y_test, y_pred)
#     r2_test_disp = f"{r2_test:.2f}" if r2_test >= 0 else "—"

#     # --- Gráfico ---
#     x_real = np.arange(total_samples - dias_graf, total_samples)
#     x_pred = x_real[-dias_pred:]

#     fig_pred = go.Figure()
#     fig_pred.add_trace(go.Scatter(
#         x=x_real, y=y_all[-dias_graf:],
#         mode='lines+markers', name="Real", line=dict(color='blue')
#     ))
#     fig_pred.add_trace(go.Scatter(
#         x=x_pred, y=y_pred,
#         mode='lines+markers', name="Previsto RF autoregresivo", line=dict(color='orange')
#     ))
#     fig_pred.add_shape(type="line",
#         x0=x_pred[0] - 0.5, y0=min(y_all[-dias_graf:]), x1=x_pred[0] - 0.5, y1=max(y_all[-dias_graf:]),
#         line=dict(color="gray", width=1, dash="dot"))
#     fig_pred.update_layout(
#         title=f"Previsión Random Forest Multi-step Autoregresivo ({dias_pred} días + 15 extra, ventana={window_size})",
#         xaxis_title="Día",
#         yaxis_title="Precio",
#         height=450,
#         showlegend=True
#     )

#     st.markdown(
#         f"""**MSE validación (autoregresiva):** {mse_val:.2f} | **R² validación:** {r2_val_disp}  
# **MSE test (autoregresivo):** {mse_test:.2f} | **R² test:** {r2_test_disp}"""
#     )
#     st.plotly_chart(fig_pred, use_container_width=True)
#     st.markdown(
#         f"""
#         - **Azul:** valores reales de la acción objetivo.
#         - **Naranja:** previsión Random Forest autoregresiva.
#         - Línea punteada gris: inicio de la previsión.
#         - El error de validación es **walk-forward** (rolling forecast).
#         """
#     )

# if __name__ == "__main__":
#     show()











# # menu/exploracao_mejorado.py
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
# ### MEJORA: Definir constantes para facilitar el mantenimiento.
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
# def download_stocks(tickers: List[str], days: int, max_lag: int) -> pd.DataFrame:
#     """Descarga datos históricos de cierre ajustado para una lista de tickers."""
#     period = days + max_lag + 40  # Búfer para asegurar suficientes datos
#     df = yf.download(tickers, period=f"{period}d", interval="1d", progress=False)
#     data = df['Adj Close'] if 'Adj Close' in df else df['Close']
#     return data

# def get_valid_data(tickers: List[str], days: int, max_lag: int) -> pd.DataFrame:
#     """Filtra y limpia los datos para asegurar que no haya NaNs y tengan la longitud requerida."""
#     data = download_stocks(tickers, days, max_lag)
#     min_len = days + max_lag + 30
    
#     if isinstance(data, pd.Series):
#         data = data.to_frame(tickers[0])
        
#     valid_cols = [c for c in data.columns if data[c].dropna().shape[0] >= min_len]
#     if not valid_cols:
#         return pd.DataFrame()

#     data_valid = data[valid_cols].iloc[-min_len:]
#     data_valid = data_valid.dropna(axis=1)
#     return data_valid

# # --- Funciones de Análisis y Modelado ---

# ### MEJORA: La lógica de pronóstico ahora es metodológicamente sólida.
# ### Se enfoca en un modelo puramente autorregresivo para evitar la fuga de datos.

# def create_sliding_window_features(target: np.ndarray, window_size: int) -> Tuple[np.ndarray, np.ndarray]:
#     """
#     Crea características y etiquetas de una serie temporal para un modelo autorregresivo.
#     Usa solo los valores pasados del propio target.
#     """
#     num_samples = len(target) - window_size
#     X = np.array([target[i:i + window_size] for i in range(num_samples)])
#     y = np.array([target[i + window_size] for i in range(num_samples)])
#     return X, y

# def perform_autoregressive_forecast(model, initial_window: np.ndarray, steps: int) -> np.ndarray:
#     """
#     Realiza un pronóstico autorregresivo multi-step.
#     Usa la predicción de un paso como entrada para el siguiente.
#     """
#     forecast = []
#     current_window = initial_window.copy()
#     for _ in range(steps):
#         pred = model.predict(current_window.reshape(1, -1))[0]
#         forecast.append(pred)
#         # Desplaza la ventana y añade la nueva predicción al final
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
#     Función completa para entrenar, optimizar y evaluar el modelo de pronóstico.
#     """
#     # 1. Crear dataset
#     X_all, y_all = create_sliding_window_features(target_series, window_size)
    
#     if len(y_all) < pred_days * 2 + window_size:
#         return {"error": "No hay suficientes datos para la ventana y días de predicción seleccionados."}

#     # 2. División Walk-Forward (Entrenamiento, Validación, Prueba)
#     X_train, y_train = X_all[:-2 * pred_days], y_all[:-2 * pred_days]
#     X_val, y_val = X_all[-2 * pred_days:-pred_days], y_all[-2 * pred_days:-pred_days]
#     X_test, y_test = X_all[-pred_days:], y_all[-pred_days:]

#     # 3. Optimización de Hiperparámetros con RandomizedSearchCV
#     ### MEJORA: Búsqueda sobre más hiperparámetros para un mejor modelo.
#     param_dist = {
#         "n_estimators": [50, 100, 150, 200],
#         "max_depth": [None, 10, 20, 30],
#         "min_samples_leaf": [1, 2, 4],
#     }
#     tscv = TimeSeriesSplit(n_splits=3)
#     random_search = RandomizedSearchCV(
#         RandomForestRegressor(random_state=seed),
#         param_distributions=param_dist,
#         n_iter=10, cv=tscv, scoring="neg_mean_squared_error", n_jobs=-1, random_state=seed
#     )
#     random_search.fit(X_train, y_train)
#     rf_best = random_search.best_estimator_

#     # 4. Evaluación Walk-Forward en el set de Validación
#     y_val_pred = perform_autoregressive_forecast(rf_best, X_val[0], len(y_val))
#     mse_val = mean_squared_error(y_val, y_val_pred)
#     r2_val = r2_score(y_val, y_val_pred)

#     # 5. Predicción final en el set de Prueba
#     y_test_pred = perform_autoregressive_forecast(rf_best, X_test[0], pred_days)
#     mse_test = mean_squared_error(y_test, y_test_pred)
#     r2_test = r2_score(y_test, y_test_pred)

#     # 6. Evaluación contra un baseline (Modelo de Persistencia)
#     ### MEJORA: Añadir un baseline para contextualizar el rendimiento del modelo.
#     last_val_before_test = y_all[-pred_days - 1]
#     persistence_preds = np.insert(y_test[:-1], 0, last_val_before_test)
#     mse_persistence = mean_squared_error(y_test, persistence_preds)

#     return {
#         "error": None,
#         "model": rf_best,
#         "predictions": y_test_pred,
#         "ground_truth_series": y_all,
#         "metrics": {
#             "mse_val": mse_val, "r2_val": r2_val,
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
#     dias_graf = pred_days + 30

#     x_real = np.arange(total_samples - dias_graf, total_samples)
#     x_pred = x_real[-pred_days:]
    
#     fig = go.Figure()
#     fig.add_trace(go.Scatter(
#         x=x_real, y=y_all[-dias_graf:],
#         mode='lines+markers', name="Real", line=dict(color='blue')
#     ))
#     fig.add_trace(go.Scatter(
#         x=x_pred, y=y_pred,
#         mode='lines+markers', name="Previsto RF Autoregresivo", line=dict(color='orange')
#     ))
#     fig.add_shape(type="line",
#         x0=x_pred[0] - 0.5, y0=min(y_all[-dias_graf:]), x1=x_pred[0] - 0.5, y1=max(y_all[-dias_graf:]),
#         line=dict(color="gray", width=1, dash="dot"))
    
#     fig.update_layout(
#         title=f"Previsión Random Forest Multi-step Autoregresivo ({pred_days} días, ventana={window_size})",
#         xaxis_title="Día (índice temporal)",
#         yaxis_title="Precio",
#         height=450,
#         showlegend=True,
#         legend=dict(x=0.01, y=0.99)
#     )
#     st.plotly_chart(fig, use_container_width=True)

# # --- Aplicación Principal de Streamlit ---

# def show():
#     """Función principal que organiza la interfaz y el flujo de la aplicación."""
#     # st.set_page_config(layout="wide")
#     st.title("🤖 Previsión de Series Temporales con Random Forest")
#     st.write("Análisis de pronóstico autorregresivo con validación walk-forward.")

#     # --- Panel de Configuración ---
#     with st.sidebar:
#         st.header("⚙️ Configuración General")
#         seed = st.number_input("Semilla para selección aleatoria", value=42, min_value=0)
#         n_stocks = st.slider("Número de acciones a analizar", 10, len(TICKERS_LIST), 20)
#         dias_historico = st.slider("Días de historia a descargar", 100, 1000, 750)
#         max_lag = st.slider("Máximo desfase (lag)", 5, 45, 25)

#     # --- Carga y Selección de Datos ---
#     init_session_state(n_stocks, seed)
#     data = get_valid_data(st.session_state['tickers'], dias_historico, max_lag)

#     if data.empty:
#         st.error("No se pudieron descargar datos válidos. Prueba con otra semilla o aumenta los días de historia.")
#         return

#     st.header("1. Selección de la Acción Objetivo")
#     action_selected = st.selectbox(
#         "Elige la acción que quieres predecir:",
#         options=data.columns,
#         index=0
#     )
    
#     with st.expander("Ver datos brutos de las acciones"):
#         st.dataframe(data.head())
        
#     # --- Sección de Previsión ---
#     st.header(f"2. Previsión para {action_selected}")

#     with st.container(border=True):
#         st.subheader("Configuración del Modelo de Previsión")
#         pred_days = st.slider("¿Cuántos días futuros predecir? (horizonte de prueba)", 1, 30, 7, key="pred_days")
#         window_size = st.slider("Tamaño de la ventana (días de historia para predecir el siguiente)", 10, 90, 30, key="window_size")
    
#     # --- Ejecutar Modelo y Mostrar Resultados ---
#     target_series = data[action_selected].dropna().values
    
#     with st.spinner("Entrenando modelo y realizando predicción..."):
#         results = train_and_evaluate_model(target_series, window_size, pred_days, seed)

#     if results.get("error"):
#         st.warning(results["error"])
#     else:
#         st.subheader("📈 Gráfico de Previsión vs. Real")
#         plot_forecast(results)

#         st.subheader("📊 Métricas de Rendimiento del Modelo")
#         m = results["metrics"]
        
#         st.markdown(f"""
#         La evaluación simula un escenario real donde el modelo se prueba en datos que nunca ha visto.
#         - **Baseline (Persistencia):** Es un modelo simple que predice que el precio de mañana será igual al de hoy.
#         - **Nuestro modelo (Random Forest)** debe tener un error (MSE) más bajo que el baseline para ser considerado útil.
#         """)
        
#         col1, col2, col3 = st.columns(3)
#         col1.metric(label="MSE Test (Modelo RF)", value=f"{m['mse_test']:.2f}", help="Error Cuadrático Medio en el conjunto de prueba. Cuanto más bajo, mejor.")
#         col2.metric(label="MSE Baseline (Persistencia)", value=f"{m['mse_persistence']:.2f}", help="Error del modelo simple. Nuestro modelo debe tener un MSE más bajo.")
#         r2_test_disp = f"{m['r2_test']:.2f}" if m['r2_test'] >= -1 else "—"
#         col3.metric(label="R² Test (Modelo RF)", value=r2_test_disp, help="Coeficiente de determinación (de -∞ a 1). Un valor cercano a 1 es bueno.")
        
#         if m['mse_test'] < m['mse_persistence']:
#             st.success("¡Éxito! El modelo de Random Forest superó al baseline de persistencia.")
#         else:
#             st.warning("El modelo de Random Forest no superó al baseline. Considera ajustar la ventana o los días a predecir.")

#         with st.expander("Ver detalles del modelo y métricas de validación"):
#             st.write("**Mejor Modelo Encontrado (tras RandomizedSearch):**")
#             st.code(str(results['model']))
#             st.write("**Métricas en el set de Validación (usado para tuning):**")
#             st.json({
#                 "MSE Validación": f"{m['mse_val']:.2f}",
#                 "R² Validación": f"{m['r2_val']:.2f}"
#             })

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










# # menu/exploracao_xgboost_corregido_final.py
# import streamlit as st
# import pandas as pd
# import numpy as np
# import yfinance as yf
# import xgboost as xgb
# import plotly.graph_objs as go
# from sklearn.metrics import mean_squared_error, r2_score
# from typing import List, Dict, Any, Tuple

# # --- 1. Configuración y Carga de Datos ---

# TICKERS_LIST = [
#     "NVDA", "AMD", "INTC", "QCOM", "AVGO", "TXN", "MU",
#     "MSFT", "GOOGL", "AMZN",
#     "SMH"
# ]

# @st.cache_data(show_spinner="Descargando datos de acciones...")
# def download_stocks(tickers: List[str]) -> pd.DataFrame:
#     """Descarga datos y asegura una frecuencia diaria para evitar errores."""
#     df = yf.download(tickers, period="4y", interval="1d", progress=False)
#     if df.empty:
#         return df
#     df = df.asfreq('D', method='ffill')
#     return df

# def find_best_lagged_correlations(
#     data: pd.DataFrame, target_ticker: str, n_top: int = 4
# ) -> pd.DataFrame:
#     """
#     Encuentra las N acciones más correlacionadas con el target, buscando el desfase (lag) óptimo.
#     """
#     price_col = 'Adj Close' if 'Adj Close' in data.columns else 'Close'

#     target_series = data[price_col][target_ticker].dropna()
#     best_corrs = {}
    
#     for ticker in data[price_col].columns:
#         if ticker == target_ticker:
#             continue
        
#         candidate_series = data[price_col][ticker].dropna()
#         best_corr, best_lag = -1, 0
        
#         for lag in range(1, 21):
#             common_idx = target_series.index.intersection(candidate_series.index.shift(lag))
#             if len(common_idx) < 100: continue

#             target_aligned = target_series.loc[common_idx]
#             candidate_aligned = candidate_series.shift(lag).loc[common_idx]
            
#             corr = target_aligned.corr(candidate_aligned)
#             if abs(corr) > abs(best_corr):
#                 best_corr = corr
#                 best_lag = lag
        
#         if best_lag > 0:
#             best_corrs[ticker] = (best_corr, best_lag)
            
#     top_tickers = sorted(best_corrs.items(), key=lambda item: abs(item[1][0]), reverse=True)[:n_top]
    
#     lagged_features = pd.DataFrame(index=data.index)
#     for ticker, (corr, lag) in top_tickers:
#         lagged_features[f'{ticker}_lag_{lag}'] = data[price_col][ticker].shift(lag)
        
#     return lagged_features


# # --- 2. Ingeniería de Características ---

# def create_features(df: pd.DataFrame, target_ticker: str, correlated_features: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
#     """Crea un conjunto de características robusto para el modelo."""
#     price_col = 'Adj Close' if 'Adj Close' in df.columns else 'Close'

#     data = pd.DataFrame(index=df.index)
#     target_price = df[price_col][target_ticker]

#     data['sma_7'] = target_price.rolling(window=7).mean()
#     data['sma_21'] = target_price.rolling(window=21).mean()
#     data['sma_ratio'] = data['sma_7'] / data['sma_21']
#     data['volatility_7'] = target_price.rolling(window=7).std()
#     data['roc_7'] = (target_price / target_price.shift(7) - 1) * 100
    
#     data['day_of_week'] = data.index.dayofweek
#     data['week_of_year'] = data.index.isocalendar().week.astype(int)
#     data['month'] = data.index.month
    
#     for i in range(1, 4):
#         data[f'target_lag_{i}'] = target_price.shift(i)
        
#     full_features = pd.concat([data, correlated_features], axis=1)
#     target = target_price.shift(-1)
    
#     # MEJORA: Combinar características y objetivo para una limpieza segura.
#     final_df = pd.concat([full_features, target.rename('target')], axis=1)
    
#     # Eliminar cualquier fila que contenga un NaN (ya sea en X o y)
#     final_df = final_df.dropna()
    
#     # Separar de nuevo en X (características) y y (objetivo)
#     X = final_df.drop(columns='target')
#     y = final_df['target']
    
#     return X, y


# # --- 3. Visualización ---

# def plot_results(actual: pd.Series, predicted: pd.Series, title: str):
#     fig = go.Figure()
#     fig.add_trace(go.Scatter(x=actual.index, y=actual, mode='lines', name='Precio Real', line=dict(color='blue')))
#     fig.add_trace(go.Scatter(x=predicted.index, y=predicted, mode='lines', name='Predicción XGBoost', line=dict(color='orange', dash='dash')))
#     fig.update_layout(title=title, xaxis_title="Fecha", yaxis_title="Precio", height=500)
#     st.plotly_chart(fig, use_container_width=True)

# def plot_feature_importance(model, features):
#     importance = pd.DataFrame({'feature': features.columns, 'importance': model.feature_importances_})
#     importance = importance.sort_values('importance', ascending=False).head(15)
    
#     fig = go.Figure(go.Bar(x=importance['importance'], y=importance['feature'], orientation='h'))
#     fig.update_layout(title='Top 15 Características Más Importantes para el Modelo', height=500)
#     st.plotly_chart(fig, use_container_width=True)

# # --- 4. Aplicación Principal de Streamlit ---

# def show():
#     st.title("🚀 Estrategia de Previsión con XGBoost e Ingeniería de Características")
#     st.write("Se predicen los precios a **un día vista (t+1)**, lo que reduce la acumulación de errores.")

#     st.sidebar.header("⚙️ Configuración")
#     target_ticker = st.sidebar.selectbox("Elige la acción a predecir:", TICKERS_LIST, index=0)
#     test_size = st.sidebar.slider("Días para el conjunto de prueba", 30, 120, 60)

#     data_ohlc = download_stocks(TICKERS_LIST)
    
#     if data_ohlc.empty:
#         st.error("No se pudieron descargar datos. Inténtalo de nuevo más tarde.")
#         return

#     with st.spinner("Buscando correlaciones y creando características..."):
#         correlated_features = find_best_lagged_correlations(data_ohlc, target_ticker)
#         X, y = create_features(data_ohlc, target_ticker, correlated_features)

#     st.header(f"Previsión para {target_ticker}")
    
#     X_train, X_test = X.iloc[:-test_size], X.iloc[-test_size:]
#     y_train, y_test = y.iloc[:-test_size], y.iloc[-test_size:]

#     st.info(f"Datos disponibles: {X.shape[0]} días. Entrenando con {X_train.shape[0]} y probando con {X_test.shape[0]}.")

#     with st.spinner("Entrenando el modelo XGBoost..."):
#         model = xgb.XGBRegressor(
#             objective='reg:squarederror',
#             n_estimators=1000,
#             learning_rate=0.05,
#             max_depth=5,
#             subsample=0.8,
#             colsample_bytree=0.8,
#             random_state=42,
#             n_jobs=-1,
#             early_stopping_rounds=50
#         )
        
#         model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)
#         predictions = pd.Series(model.predict(X_test), index=X_test.index)

#     st.subheader("📈 Gráfico de Previsión (1 día vista) vs. Real")
#     plot_results(y_test, predictions, f"Predicción para {target_ticker} en el Período de Prueba")
    
#     st.subheader("📊 Métricas de Rendimiento")
#     mse = mean_squared_error(y_test, predictions)
#     r2 = r2_score(y_test, predictions)
    
#     col1, col2 = st.columns(2)
#     col1.metric("Error Cuadrático Medio (MSE)", f"{mse:.2f}")
#     col2.metric("Coeficiente de Determinación (R²)", f"{r2:.3f}")
    
#     with st.expander("Ver Detalles de la Estrategia y Características"):
#         st.markdown("""
#         #### ¿Cómo funciona?
#         1.  **Alineamiento (Lagged Correlation):** Se buscan las 4 acciones más correlacionadas.
#         2.  **Indicadores Técnicos:** Se calculan medias móviles, volatilidad y momentum.
#         3.  **Características de Calendario:** Se incluye el día de la semana, mes, etc.
#         4.  **Modelo XGBoost:** Un algoritmo potente que aprende de sus errores.
#         5.  **Early Stopping:** El entrenamiento se detiene si el modelo deja de mejorar.
#         """)
#         st.subheader("Importancia de las Características")
#         plot_feature_importance(model, X_train)

# if __name__ == "__main__":
#     show()









# # menu/exploracao_lgbm_semanal.py
# import streamlit as st
# import pandas as pd
# import numpy as np
# import yfinance as yf
# import lightgbm as lgb
# import plotly.graph_objs as go
# from sklearn.metrics import mean_absolute_error, r2_score
# from typing import List, Dict, Tuple

# # --- 1. Configuración y Selección de Universos ---

# # MEJORA: Universos de acciones para elegir
# STOCK_UNIVERSES = {
#     "Tecnología 🦾": ["AAPL", "MSFT", "NVDA", "AMD", "CRM", "ADBE", "ORCL"],
#     "Finanzas 💵": ["JPM", "BAC", "V", "MA", "GS", "MS"],
#     "Consumo 🛒": ["WMT", "COST", "MCD", "NKE", "KO", "PG"],
#     "Salud ⚕️": ["JNJ", "PFE", "UNH", "LLY", "MRK", "ABBV"]
# }
# MARKET_INDEX = 'SPY' # S&P 500 ETF como referencia del mercado

# @st.cache_data(show_spinner="Descargando datos de acciones...")
# def download_stocks(tickers: List[str]) -> pd.DataFrame:
#     """Descarga datos y asegura una frecuencia diaria."""
#     df = yf.download(tickers, period="5y", interval="1d", progress=False)
#     if df.empty:
#         return pd.DataFrame()
#     price_col = 'Adj Close' if 'Adj Close' in df.columns else 'Close'
#     # Usar un solo nivel de columnas para simplificar
#     df_processed = df[price_col]
#     df_processed = df_processed.asfreq('D', method='ffill')
#     return df_processed

# # --- 2. Ingeniería de Características (Estrategia Market-Aware) ---

# def create_features(stock_prices: pd.Series, market_prices: pd.Series) -> Tuple[pd.DataFrame, pd.Series]:
#     """
#     Crea un conjunto de características avanzado y define el objetivo como el retorno a 5 días.
#     """
#     features = pd.DataFrame(index=stock_prices.index)
    
#     # --- Características del Target ---
#     # Lags del precio (autoregresivo)
#     for i in range(1, 6):
#         features[f'price_lag_{i}'] = stock_prices.shift(i)
#     # Indicadores técnicos
#     features['sma_10'] = stock_prices.rolling(10).mean()
#     features['sma_30'] = stock_prices.rolling(30).mean()
#     features['volatility_10'] = stock_prices.rolling(10).std()
#     features['roc_10'] = (stock_prices / stock_prices.shift(10) - 1) * 100
    
#     # --- Características del Mercado (SPY) ---
#     market_returns = market_prices.pct_change()
#     features['market_return_lag_1'] = market_returns.shift(1)
#     features['market_return_lag_5'] = market_returns.shift(5)
#     features['market_volatility_10'] = market_returns.rolling(10).std()
    
#     # --- Características de Calendario ---
#     features['day_of_week'] = features.index.dayofweek
#     features['month'] = features.index.month

#     # --- OBJETIVO (y): Retorno a 5 días vista ---
#     target = (stock_prices.shift(-5) / stock_prices - 1) * 100
    
#     # --- Limpieza Final ---
#     full_df = pd.concat([features, target.rename('target')], axis=1)
#     full_df = full_df.dropna()
    
#     X = full_df.drop(columns='target')
#     y = full_df['target']
    
#     return X, y

# # --- 3. Entrenamiento y Predicción ---

# def train_and_predict(X_train: pd.DataFrame, y_train: pd.Series, X_test: pd.DataFrame) -> pd.Series:
#     """Entrena un modelo LightGBM y devuelve las predicciones."""
#     model = lgb.LGBMRegressor(random_state=42, n_estimators=200, learning_rate=0.05, num_leaves=31)
#     model.fit(X_train, y_train)
#     predictions = pd.Series(model.predict(X_test), index=X_test.index)
#     return predictions

# # --- 4. Visualización Avanzada ---

# def plot_cumulative_return(actual_returns: pd.Series, predicted_returns: pd.Series, ticker: str):
#     """
#     Grafica el retorno acumulado de la estrategia del modelo vs. Buy & Hold.
#     """
#     # Estrategia del Modelo: Invierte solo si el retorno predicho es positivo
#     model_strategy_returns = actual_returns[predicted_returns > 0]
    
#     # Calcular retornos acumulados
#     buy_hold_cumulative = (1 + actual_returns / 100).cumprod()
#     model_cumulative = (1 + model_strategy_returns / 100).cumprod()
    
#     fig = go.Figure()
#     fig.add_trace(go.Scatter(x=buy_hold_cumulative.index, y=buy_hold_cumulative, mode='lines', name='Estrategia: Comprar y Mantener (Buy & Hold)'))
#     fig.add_trace(go.Scatter(x=model_cumulative.index, y=model_cumulative, mode='lines', name='Estrategia: Guiada por el Modelo'))
    
#     fig.update_layout(
#         title=f"Backtest de Estrategia para {ticker} (Retorno Acumulado)",
#         yaxis_title="Retorno Acumulado (1 = Punto de Partida)",
#         xaxis_title="Fecha",
#         legend=dict(x=0.01, y=0.99)
#     )
#     st.plotly_chart(fig, use_container_width=True)

# def plot_feature_importance(model, features):
#     """Grafica la importancia de las características."""
#     importance = pd.DataFrame({'feature': features.columns, 'importance': model.feature_importances_})
#     importance = importance.sort_values('importance', ascending=False).head(15)
#     fig = go.Figure(go.Bar(x=importance['importance'], y=importance['feature'], orientation='h'))
#     fig.update_layout(title='Top 15 Características Más Importantes', height=500)
#     st.plotly_chart(fig, use_container_width=True)


# # --- 5. Aplicación Principal ---

# def show():
#     st.title("🧠 Previsión de Retorno Semanal con LightGBM")
#     st.write("Una estrategia robusta que predice el rendimiento porcentual a 5 días y evalúa su valor mediante un backtest de retorno acumulado.")

#     # --- Selección de Universo y Acción ---
#     st.sidebar.header("⚙️ Configuración")
#     selected_universe_name = st.sidebar.selectbox("1. Elige un universo de acciones:", list(STOCK_UNIVERSES.keys()))
#     tickers_in_universe = STOCK_UNIVERSES[selected_universe_name]
#     target_ticker = st.sidebar.selectbox("2. Elige la acción a predecir:", tickers_in_universe)
#     test_size = st.sidebar.slider("Días para el conjunto de prueba", 60, 250, 120)

#     # --- Proceso Principal ---
#     all_tickers_to_load = tickers_in_universe + [MARKET_INDEX]
#     all_data = download_stocks(all_tickers_to_load)
    
#     if all_data.empty:
#         st.error("No se pudieron descargar los datos. Inténtalo de nuevo.")
#         return

#     stock_prices = all_data[target_ticker]
#     market_prices = all_data[MARKET_INDEX]
    
#     with st.spinner("Creando características y entrenando el modelo..."):
#         X, y = create_features(stock_prices, market_prices)
        
#         X_train, X_test = X.iloc[:-test_size], X.iloc[-test_size:]
#         y_train, y_test = y.iloc[:-test_size], y.iloc[-test_size:]
        
#         predictions = train_and_predict(X_train, y_train, X_test)

#     # --- Mostrar Resultados ---
#     st.header(f"Resultados para {target_ticker}")

#     st.subheader("Simulación de Estrategia vs. Comprar y Mantener")
#     plot_cumulative_return(y_test, predictions, target_ticker)
    
#     st.subheader("📊 Métricas de Rendimiento del Modelo")
#     mae = mean_absolute_error(y_test, predictions)
#     r2 = r2_score(y_test, predictions)
    
#     col1, col2 = st.columns(2)
#     col1.metric("Error Absoluto Medio (MAE)", f"{mae:.2f}%", help="En promedio, el modelo se equivoca en ± este % de retorno.")
#     col2.metric("Coeficiente de Determinación (R²)", f"{r2:.3f}", help="¿Qué % de la variación del retorno explica el modelo?")

#     with st.expander("Ver Detalles de la Estrategia y Modelo"):
#         st.markdown(f"""
#         #### ¿Cómo funciona esta estrategia?
#         1.  **Objetivo Claro:** El modelo aprende a predecir el rendimiento porcentual de `{target_ticker}` en los próximos 5 días hábiles.
#         2.  **Contexto de Mercado:** En lugar de correlaciones complejas, se usan características clave del S&P 500 (`{MARKET_INDEX}`), como su rendimiento y volatilidad. Esto le da al modelo una visión del panorama general.
#         3.  **Indicadores Técnicos:** Se usan medias móviles y métricas de volatilidad de la propia acción para capturar su comportamiento y tendencia recientes.
#         4.  **Modelo Rápido y Eficaz:** `LightGBM` se entrena para encontrar patrones complejos que conectan las características con el rendimiento futuro.
#         5.  **Evaluación Práctica:** El gráfico de retorno acumulado no solo mide el error, sino que responde a la pregunta: **"¿Me habría sido útil este modelo para tomar decisiones?"**
#         """)
#         model_for_importance = lgb.LGBMRegressor(random_state=42).fit(X_train, y_train)
#         plot_feature_importance(model_for_importance, X_train)

# if __name__ == "__main__":
#     show()










# menu/estrategia_optuna.py
import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import lightgbm as lgb
import plotly.graph_objs as go
import optuna
from sklearn.metrics import mean_absolute_error, r2_score
from typing import List, Dict, Tuple

# --- 1. Configuración y Selección de Universos ---
STOCK_UNIVERSES = {
    "Tecnología 🦾": ["AAPL", "MSFT", "NVDA", "AMD", "CRM", "ADBE", "ORCL"],
    "Finanzas 💵": ["JPM", "BAC", "V", "MA", "GS", "MS"],
    "Consumo 🛒": ["WMT", "COST", "MCD", "NKE", "KO", "PG"],
    "Salud ⚕️": ["JNJ", "PFE", "UNH", "LLY", "MRK", "ABBV"]
}
MARKET_INDEX = 'SPY' # S&P 500 ETF como referencia del mercado
N_TOP_CORRELATED = 2 # Usar las 2 acciones más correlacionadas

# --- 2. Carga y Procesamiento de Datos ---
@st.cache_data(show_spinner="Descargando datos de mercado...")
def download_stocks(tickers: List[str]) -> pd.DataFrame:
    """Descarga datos y los prepara con una frecuencia diaria."""
    df = yf.download(tickers, period="5y", interval="1d", progress=False)
    if df.empty: return pd.DataFrame()
    
    price_col = 'Adj Close' if 'Adj Close' in df.columns else 'Close'
    # Usar un solo nivel de columnas para simplificar
    if isinstance(df.columns, pd.MultiIndex):
        df_processed = df[price_col]
    else:
        df_processed = df[price_col] if price_col in df.columns else df['Close']
        
    df_processed = df_processed.asfreq('D', method='ffill')
    return df_processed

def find_top_correlated_peers(all_data: pd.DataFrame, target_ticker: str) -> List[str]:
    """Identifica los N pares más correlacionados basados en retornos diarios."""
    returns = all_data.pct_change().dropna()
    if target_ticker not in returns.columns: return []
    
    # Asegurarse de que el índice de mercado y el target no se incluyan en la búsqueda de pares
    cols_to_drop = [target_ticker]
    if MARKET_INDEX in returns.columns:
        cols_to_drop.append(MARKET_INDEX)
        
    correlations = returns.corr()[target_ticker].drop(cols_to_drop, errors='ignore')
    top_peers = correlations.abs().nlargest(N_TOP_CORRELATED).index.tolist()
    return top_peers

# --- 3. Ingeniería de Características Híbrida ---
def create_features(
    stock_prices: pd.Series, market_prices: pd.Series, peer_data: pd.DataFrame
) -> Tuple[pd.DataFrame, pd.Series]:
    """Crea características del target, del mercado y de sus pares correlacionados."""
    features = pd.DataFrame(index=stock_prices.index)
    stock_returns = stock_prices.pct_change()

    # Características del Target
    features['return_lag_1'] = stock_returns.shift(1)
    features['return_lag_5'] = stock_returns.shift(5)
    features['sma_10_30_ratio'] = stock_prices.rolling(10).mean() / stock_prices.rolling(30).mean()
    features['volatility_10'] = stock_returns.rolling(10).std()

    # Características del Mercado (SPY)
    market_returns = market_prices.pct_change()
    features['market_return_lag_1'] = market_returns.shift(1)
    
    # Características de Pares Correlacionados (Alineamiento)
    for peer_ticker in peer_data.columns:
        peer_returns = peer_data[peer_ticker].pct_change()
        features[f'peer_{peer_ticker}_return_lag_1'] = peer_returns.shift(1)

    # Características de Calendario
    features['month'] = features.index.month
    
    # Objetivo: Retorno porcentual a 5 días
    target = (stock_prices.shift(-5) / stock_prices - 1) * 100
    
    # Limpieza final
    full_df = pd.concat([features, target.rename('target')], axis=1).dropna()
    return full_df.drop(columns='target'), full_df['target']

# --- 4. Optimización y Entrenamiento con Optuna ---
def tune_and_train_model(X_train, y_train, X_val, y_val, n_trials=50):
    """Usa Optuna para encontrar los mejores hiperparámetros y entrena el modelo final."""
    def objective(trial):
        params = {
            'objective': 'regression_l1', 'metric': 'mae', 'verbosity': -1,
            'n_estimators': 1000, 'random_state': 42,
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1),
            'num_leaves': trial.suggest_int('num_leaves', 20, 50),
            'max_depth': trial.suggest_int('max_depth', 5, 12),
            'subsample': trial.suggest_float('subsample', 0.7, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.7, 1.0),
        }
        model = lgb.LGBMRegressor(**params)
        model.fit(X_train, y_train, eval_set=[(X_val, y_val)],
                  callbacks=[lgb.early_stopping(15, verbose=False)])
        preds = model.predict(X_val)
        return mean_absolute_error(y_val, preds)

    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=n_trials)
    
    best_params = study.best_params
    final_model = lgb.LGBMRegressor(objective='regression_l1', random_state=42, n_estimators=2000, **best_params)
    final_model.fit(X_train, y_train, eval_set=[(X_val, y_val)], callbacks=[lgb.early_stopping(30, verbose=False)])
    
    return final_model, best_params

# --- 5. Visualización ---
def plot_cumulative_return(actual_returns: pd.Series, predicted_returns: pd.Series, ticker: str):
    # Estrategia del Modelo: Invierte solo si el retorno predicho es positivo
    model_strategy_returns = actual_returns[predicted_returns > 0]
    
    # Calcular retornos acumulados
    buy_hold_cumulative = (1 + actual_returns / 100).cumprod()
    model_cumulative = (1 + model_strategy_returns / 100).cumprod()
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=buy_hold_cumulative.index, y=buy_hold_cumulative, mode='lines', name='Estrategia: Comprar y Mantener'))
    fig.add_trace(go.Scatter(x=model_cumulative.index, y=model_cumulative, mode='lines', name='Estrategia: Guiada por el Modelo'))
    fig.update_layout(title=f"Backtest de Estrategia para {ticker}", yaxis_title="Retorno Acumulado (1 = Punto de Partida)", legend=dict(x=0.01, y=0.99))
    st.plotly_chart(fig, use_container_width=True)

# --- 6. Aplicación Principal ---
def show():
    # st.set_page_config(layout="wide")
    st.title("🏆 Previsión Semanal con Estrategia Híbrida y Optuna")
    st.info("Esta estrategia predice el rendimiento semanal de una acción usando su propia data, la del mercado (SPY) y la de sus pares más correlacionados. El modelo se optimiza automáticamente con Optuna.")

    st.sidebar.header("⚙️ Configuración")
    universe_name = st.sidebar.selectbox("1. Elige un universo de acciones:", list(STOCK_UNIVERSES.keys()))
    tickers = STOCK_UNIVERSES[universe_name]
    target_ticker = st.sidebar.selectbox("2. Elige la acción a predecir:", tickers)
    test_size = st.sidebar.slider("Días para el conjunto de prueba", 90, 365, 180)
    
    all_tickers_to_load = list(set(tickers + [MARKET_INDEX]))
    all_data = download_stocks(all_tickers_to_load)
    
    if all_data.empty or target_ticker not in all_data.columns or MARKET_INDEX not in all_data.columns:
        st.error("No se pudieron descargar los datos para la acción objetivo o el índice de mercado. Inténtalo de nuevo."); return

    top_peers = find_top_correlated_peers(all_data, target_ticker)
    st.sidebar.success(f"Pares para alineamiento: {', '.join(top_peers)}")
    
    X, y = create_features(all_data[target_ticker], all_data[MARKET_INDEX], all_data[top_peers])
    
    train_end_idx = len(X) - test_size
    val_start_idx = int(train_end_idx * 0.8) # Usar 20% del set de entrenamiento para validación
    
    X_train, y_train = X[:val_start_idx], y[:val_start_idx]
    X_val, y_val = X[val_start_idx:train_end_idx], y[val_start_idx:train_end_idx]
    X_test, y_test = X[train_end_idx:], y[train_end_idx:]
    
    st.info(f"Datos: {len(X_train)} para entrenar, {len(X_val)} para validar/optimizar, y {len(X_test)} para probar.")

    with st.spinner("Optimizando hiperparámetros con Optuna y entrenando... (puede tardar hasta un minuto)"):
        model, best_params = tune_and_train_model(X_train, y_train, X_val, y_val)
        predictions = pd.Series(model.predict(X_test), index=X_test.index)
    
    st.header(f"Resultados para {target_ticker}")
    plot_cumulative_return(y_test, predictions, target_ticker)
    
    mae = mean_absolute_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)
    
    col1, col2 = st.columns(2)
    col1.metric("Error Absoluto Medio (MAE)", f"{mae:.2f}%", help="En promedio, la predicción de retorno semanal se equivoca en ± este %.")
    col2.metric("Coeficiente de Determinación (R²)", f"{r2:.3f}", help="Porcentaje de la variación del retorno que el modelo logra explicar.")

    with st.expander("Ver mejores parámetros y detalles del modelo"):
        st.subheader("Mejores Hiperparámetros Encontrados por Optuna")
        st.json(best_params)
        
        st.subheader("Importancia de las Características")
        importance_df = pd.DataFrame({
            'feature': model.feature_name_,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)
        st.dataframe(importance_df)

if __name__ == "__main__":
    show()