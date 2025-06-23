# menu/exploracao.py
import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go

# --- INICIO DE LA CORRECCIÓN ---
def find_best_lag(series_ref, series_comp, max_lag=15):
    """
    Calcula la correlación cruzada entre dos series y encuentra el retardo
    que maximiza la correlación.
    """
    correlations = {}
    for lag in range(-max_lag, max_lag + 1):
        corr = series_ref.corr(series_comp.shift(lag))
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

        # Asegurar que siempre es Serie (no DataFrame)
        price_series = data[price_col]
        if isinstance(price_series, pd.DataFrame):
            # Si por alguna razón es DataFrame, tomar la primera columna como Serie
            price_series = price_series.iloc[:, 0]
        price_series.name = ticker

        if final_df.empty:
            final_df = price_series.to_frame()
        else:
            # Unir por índice (fecha)
            final_df = final_df.join(price_series, how='outer')

        progress_bar.progress(progress_value)

    status_text.text("¡Descarga completa!")
    progress_bar.empty()
    return final_df

# --- FIN DE LA CORRECCIÓN ---

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
            returns = data_close.pct_change().dropna()

            if returns.empty or ref_ticker not in returns.columns:
                st.error("No hay suficientes datos para calcular los rendimientos. Intente con un período más largo.")
                st.stop()

            ref_series = returns[ref_ticker]

            results = []
            for ticker in comp_tickers:
                if ticker in returns.columns and ticker != ref_ticker:
                    comp_series = returns[ticker]
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
        st.markdown(f"La tabla muestra el desfase en días que maximiza la correlación entre los rendimientos de **{ref_ticker}** y las demás acciones.")

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

        st.header("4. Análisis Detallado de la Mejor Relación")
        if not results_df.empty:
            best_stock = results_df.iloc[0]["Acción Comparada"]
            best_lag_found = results_df.iloc[0]["Mejor Desfase (días)"]

            st.markdown(f"La relación más fuerte se encontró con **{best_stock}**, con un desfase de **{best_lag_found} días**.")

            shifted_series = returns[best_stock].shift(best_lag_found)

            fig_detail = go.Figure()
            fig_detail.add_trace(go.Scatter(x=returns.index, y=ref_series, name=f"Rendimientos de {ref_ticker}"))
            fig_detail.add_trace(go.Scatter(x=returns.index, y=shifted_series, name=f"Rendimientos de {best_stock} (desfasado {best_lag_found} días)", line=dict(dash='dash')))
            fig_detail.update_layout(title=f"Comparación de Rendimientos: {ref_ticker} vs. {best_stock} (alineado)")
            st.plotly_chart(fig_detail, use_container_width=True)
