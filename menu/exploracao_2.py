# # menu/exploracao.py
import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go

def find_best_lag(series_ref, series_comp, max_lag=15):
    """
    Calcula la correlación cruzada entre dos series y encuentra el retardo
    que maximiza la correlación absoluta.
    """
    correlations = {}
    for lag in range(-max_lag, max_lag + 1):
        shifted = series_comp.shift(lag)
        valid = pd.concat([series_ref, shifted], axis=1).dropna()
        if valid.shape[0] < 2:
            continue
        corr = valid.iloc[:,0].corr(valid.iloc[:,1])
        if not np.isnan(corr):
            correlations[lag] = corr

    if not correlations:
        return 0, 0

    best_lag = max(correlations, key=lambda k: abs(correlations[k]))
    return best_lag, correlations[best_lag]

def load_data(tickers, period):
    """
    Descarga los datos de las acciones una por una y los une en un solo DataFrame
    de formato ancho.
    """
    final_df = pd.DataFrame()
    progress_bar = st.progress(0, text="Iniciando descarga...")
    status_text = st.empty()

    for i, ticker in enumerate(tickers):
        progress_value = (i + 1) / len(tickers)
        status_text.text(f"Descargando {ticker}... ({i+1}/{len(tickers)})")

        try:
            data = yf.download(ticker, period=period, interval="1d")
        except Exception as e:
            st.error(f"Error al descargar '{ticker}': {type(e).__name__} - {e}")
            progress_bar.progress(progress_value)
            continue

        if data.empty:
            st.warning(f"No se encontraron datos para '{ticker}'. Se omitirá.")
            progress_bar.progress(progress_value)
            continue

        if 'Adj Close' in data.columns:
            price_col = 'Adj Close'
        elif 'Close' in data.columns:
            price_col = 'Close'
        else:
            st.warning(f"No se encontró la columna 'Adj Close' o 'Close' para '{ticker}'. Se omitirá.")
            progress_bar.progress(progress_value)
            continue

        price_series = data[price_col]
        if isinstance(price_series, pd.DataFrame):
            price_series = price_series.iloc[:, 0]
        price_series.name = ticker

        if final_df.empty:
            final_df = price_series.to_frame()
        else:
            final_df = final_df.join(price_series, how='outer')

        progress_bar.progress(progress_value)

    status_text.text("¡Descarga completa!")
    progress_bar.empty()
    return final_df

def render():
    st.title("🔎 Exploración de Correlación con Retardo")
    st.markdown("""
    Esta herramienta analiza cómo los movimientos de una acción de referencia se correlacionan
    con los de otras a lo largo del tiempo. Buscamos encontrar **relaciones de liderazgo y retraso**.
    """)

    st.header("1. Configuración del Análisis")

    col1, col2 = st.columns([1, 2])
    with col1:
        ref_ticker = st.text_input("Acción de Referencia:", "AMZN")
        period = st.selectbox("Período de Análisis:", ["2y", "3y", "5y", "10y"], index=1)
        max_lag = st.slider("Máximo de días de desfase:", min_value=5, max_value=30, value=15)
        modo = st.selectbox("Tipo de análisis:", ["Rendimientos diarios (%)", "Precio ajustado"])

    with col2:
        default_tickers = "MSFT, WMT, FDX, UPS, NVDA, GOOGL, SHOP"
        comp_tickers_str = st.text_area("Acciones para Comparar (separadas por coma):", default_tickers, height=100)

    comp_tickers = [ticker.strip().upper() for ticker in comp_tickers_str.split(",") if ticker.strip()]

    if st.button("🚀 Ejecutar Análisis de Correlación", type="primary"):
        with st.spinner("Realizando análisis completo..."):
            if not ref_ticker or not comp_tickers:
                st.warning("Por favor, ingrese una acción de referencia y al menos una acción para comparar.")
                st.stop()

            all_tickers = list(set([ref_ticker] + comp_tickers))

            data_close = load_data(all_tickers, period)

            if data_close.empty or ref_ticker not in data_close.columns:
                st.error(f"No se pudieron obtener datos válidos para la acción de referencia {ref_ticker}. El análisis no puede continuar.")
                st.stop()

            data_close.dropna(how='all', inplace=True)

            if modo == "Rendimientos diarios (%)":
                df_corr = data_close.pct_change().dropna()
            else:
                df_corr = data_close.dropna()

            if df_corr.empty or ref_ticker not in df_corr.columns:
                st.error("No hay suficientes datos para el análisis. Intente con un período más largo o revise los datos descargados.")
                st.stop()

            ref_series = df_corr[ref_ticker]

            results = []
            for ticker in comp_tickers:
                if ticker in df_corr.columns and ticker != ref_ticker:
                    comp_series = df_corr[ticker]
                    best_lag_val, best_corr_val = find_best_lag(ref_series, comp_series, max_lag)
                    results.append({
                        "Acción Comparada": ticker,
                        "Mejor Desfase (días)": best_lag_val,
                        "Máxima Correlación": best_corr_val
                    })

            if not results:
                st.warning("No se pudo calcular la correlación para ninguna de las acciones.")
                st.stop()

            results_df = pd.DataFrame(results).sort_values(by="Máxima Correlación", key=abs, ascending=False).reset_index(drop=True)

        st.header("2. Resultados del Análisis")
        st.dataframe(results_df.style.format({"Máxima Correlación": "{:.4f}"}), use_container_width=True)
        st.markdown(f"La tabla muestra el desfase en días que maximiza la correlación de **{ref_ticker}** con las demás acciones. <br>Tipo de análisis: <b>{modo}</b>", unsafe_allow_html=True)

        st.header("3. Visualización Gráfica")
        fig_corr = go.Figure(go.Bar(
            x=results_df["Acción Comparada"],
            y=results_df["Máxima Correlación"],
            text=results_df["Máxima Correlación"].apply(lambda x: f"{x:.3f}"),
            textposition='auto',
            marker_color='indianred'
        ))
        fig_corr.update_layout(title_text=f"Máxima Correlación de cada Acción con {ref_ticker}")
        st.plotly_chart(fig_corr, use_container_width=True)

        fig_lag = go.Figure(go.Bar(
            x=results_df["Acción Comparada"],
            y=results_df["Mejor Desfase (días)"],
            text=results_df["Mejor Desfase (días)"],
            textposition='auto',
            marker_color='lightblue'
        ))
        fig_lag.update_layout(title_text=f"Mejor Desfase (días) de cada Acción con {ref_ticker}")
        st.plotly_chart(fig_lag, use_container_width=True)
        

        st.header("4. Análisis Detallado de la Mejor Relación con Desfase")
        # Filtrar para quedarnos solo con las correlaciones cuyo mejor desfase sea distinto de cero
        nonzero_lag = results_df[results_df["Mejor Desfase (días)"] != 0]
        if not nonzero_lag.empty:
            # Elegir la de mayor correlación absoluta entre las de lag distinto de cero
            idx_best = nonzero_lag["Máxima Correlación"].abs().idxmax()
            best_row = nonzero_lag.loc[idx_best]
            best_stock = best_row["Acción Comparada"]
            best_lag_found = best_row["Mejor Desfase (días)"]
            best_corr = best_row["Máxima Correlación"]

            st.markdown(
                f"La relación **más fuerte con desfase** se encontró con <b>{best_stock}</b>, "
                f"en <b>{best_lag_found:+d} días</b>, "
                f"con una correlación de <b>{best_corr:.3f}</b>.",
                unsafe_allow_html=True
            )

            shifted_series = df_corr[best_stock].shift(best_lag_found)

            fig_detail = go.Figure()
            fig_detail.add_trace(go.Scatter(x=df_corr.index, y=ref_series, name=f"{ref_ticker}"))
            fig_detail.add_trace(go.Scatter(
                x=df_corr.index,
                y=shifted_series,
                name=f"{best_stock} (desfasado {best_lag_found} días)",
                line=dict(dash='dash')))
            fig_detail.update_layout(title=f"Comparación: {ref_ticker} vs. {best_stock} (alineado, desfase {best_lag_found:+d} días)")
            st.plotly_chart(fig_detail, use_container_width=True)
        else:
            st.info(
                "Ninguna acción presentó una correlación máxima relevante con desfase distinto de cero. "
                "Intenta aumentar el rango de días de desfase o prueba con otros tickers."
            )


        
