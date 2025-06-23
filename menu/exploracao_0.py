import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
from sklearn.preprocessing import MinMaxScaler

def render():
    st.title("Exploración de Acciones")
    
    # Sección de configuración
    st.header("Configuración")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Selección de acciones
        default_tickers = ["AAPL", "MSFT", "GOOGL", "AMZN", "META"]
        user_input = st.text_input("Ingrese símbolos de acciones separados por coma:", 
                                   value="AAPL, MSFT, GOOGL")
        tickers = [ticker.strip() for ticker in user_input.split(",")]
        
        # Selección de período
        period_options = ["1mo", "3mo", "6mo", "1y", "2y", "5y", "max"]
        selected_period = st.selectbox("Seleccione período:", period_options, index=3)
    
    with col2:
        # Selección de intervalo
        interval_options = ["1d", "5d", "1wk", "1mo", "3mo"]
        selected_interval = st.selectbox("Seleccione intervalo:", interval_options, index=0)
        
        # Normalización
        normalize = st.checkbox("Normalizar datos", value=True)
    
    # Botón para cargar datos
    if st.button("Cargar datos"):
        with st.spinner("Cargando datos de acciones..."):
            try:
                # Obtener datos
                data = {}
                valid_tickers = []
                
                for ticker in tickers:
                    try:
                        stock_data = yf.download(ticker, period=selected_period, 
                                               interval=selected_interval)
                        if not stock_data.empty:
                            data[ticker] = stock_data
                            valid_tickers.append(ticker)
                    except Exception as e:
                        st.warning(f"Error al cargar datos para {ticker}: {e}")
                
                if not data:
                    st.error("No se pudieron cargar datos para ninguna de las acciones seleccionadas.")
                    return
                
                # Mostrar datos
                st.header("Visualización de Precios")
                
                # Crear DataFrame combinado para precios de cierre
                close_prices = pd.DataFrame()
                
                for ticker in valid_tickers:
                    close_prices[ticker] = data[ticker]['Close']
                
                # Normalizar si se seleccionó
                if normalize:
                    scaler = MinMaxScaler()
                    normalized_data = pd.DataFrame(
                        scaler.fit_transform(close_prices),
                        columns=close_prices.columns,
                        index=close_prices.index
                    )
                    plot_data = normalized_data
                    y_title = "Precio normalizado"
                else:
                    plot_data = close_prices
                    y_title = "Precio de cierre (USD)"
                
                # Gráfico de precios
                fig = px.line(plot_data, x=plot_data.index, y=plot_data.columns,
                             title="Evolución de precios de acciones")
                fig.update_layout(
                    xaxis_title="Fecha",
                    yaxis_title=y_title,
                    legend_title="Acciones",
                    height=600
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # Matriz de correlación
                st.header("Análisis de Correlación")
                
                corr_matrix = close_prices.corr()
                fig_corr = px.imshow(
                    corr_matrix,
                    text_auto=True,
                    color_continuous_scale='RdBu_r',
                    title="Matriz de Correlación entre Acciones"
                )
                fig_corr.update_layout(height=500)
                st.plotly_chart(fig_corr, use_container_width=True)
                
                # Estadísticas básicas
                st.header("Estadísticas Básicas")
                
                # Calcular rendimientos diarios
                returns = close_prices.pct_change().dropna()
                
                # Métricas en columnas
                metric_cols = st.columns(len(valid_tickers))
                
                for i, ticker in enumerate(valid_tickers):
                    with metric_cols[i]:
                        current_price = close_prices[ticker].iloc[-1]
                        price_change = close_prices[ticker].iloc[-1] - close_prices[ticker].iloc[-2]
                        pct_change = (price_change / close_prices[ticker].iloc[-2]) * 100
                        
                        st.metric(
                            label=ticker,
                            value=f"${current_price:.2f}",
                            delta=f"{pct_change:.2f}%"
                        )
                
                # Tabla de estadísticas
                st.subheader("Resumen estadístico")
                stats_df = pd.DataFrame({
                    'Volatilidad (%)': returns.std() * 100,
                    'Rendimiento Promedio (%)': returns.mean() * 100,
                    'Precio Mínimo': close_prices.min(),
                    'Precio Máximo': close_prices.max(),
                    'Precio Actual': close_prices.iloc[-1]
                }).T
                
                st.dataframe(stats_df)
                
                # Datos crudos
                with st.expander("Ver datos crudos"):
                    st.dataframe(close_prices)
                
            except Exception as e:
                st.error(f"Error al procesar los datos: {e}")
    
    # Sugerencias de algoritmos para series temporales
    st.header("Algoritmos recomendados para previsión de series temporales")
    
    algorithms = [
        {"name": "ARIMA/SARIMA", "description": "Modelos clásicos para series temporales con componentes autorregresivos y de media móvil"},
        {"name": "Prophet", "description": "Desarrollado por Facebook, maneja bien tendencias, estacionalidad y eventos especiales"},
        {"name": "LSTM/GRU", "description": "Redes neuronales recurrentes para capturar dependencias temporales complejas"},
        {"name": "XGBoost/LightGBM", "description": "Algoritmos de gradient boosting que funcionan bien con características temporales"},
        {"name": "VAR (Vector Autoregression)", "description": "Para modelar relaciones entre múltiples series temporales"},
        {"name": "Dynamic Time Warping (DTW)", "description": "Para alineamiento de series temporales con desfases variables"},
        {"name": "Transformers", "description": "Arquitecturas de atención para capturar relaciones a largo plazo en series temporales"}
    ]
    
    for algo in algorithms:
        st.markdown(f"**{algo['name']}**: {algo['description']}")