# # exploracao.py

# import streamlit as st
# import pandas as pd
# import plotly.graph_objects as go
# import numpy as np

# def carregar_dados():
#     df_r = pd.read_csv("data/R_alinhados.csv")
#     df_vitals = pd.read_csv("data/vitals_alinhado.csv")

#     sensores_r = ['1', '2', '3']
#     dict_vars = {
#         'SpO2': 'SpO2_%_Value_',
#         'PR': 'PR_bpm_Value_',
#         'Pi': 'Pi_Value_',
#         'Events': 'Events_'
#     }

#     dfs_r = {}
#     for s in sensores_r:
#         cols = ['Epoch Time'] + [v + s for v in dict_vars.values()]
#         df_tmp = df_r[cols].copy()
#         df_tmp.rename(columns={
#             f'SpO2_%_Value_{s}': f'SpO2_R{s}',
#             f'PR_bpm_Value_{s}': f'PR_R{s}',
#             f'Pi_Value_{s}': f'Pi_R{s}',
#             f'Events_{s}': f'Events_R{s}'
#         }, inplace=True)
#         df_tmp['datetime'] = pd.to_datetime(df_tmp['Epoch Time'], unit='ms')
#         # Ajustar a la zona horaria local (ajusta el número según tu zona horaria)
#         df_tmp['datetime'] = df_tmp['datetime'].dt.tz_localize('UTC').dt.tz_convert('America/Los_Angeles')  # UTC-7/UTC-8
#         # Eliminar información de zona horaria para evitar problemas de visualización
#         df_tmp['datetime'] = df_tmp['datetime'].dt.tz_localize(None)

        
#         dfs_r[f'R{s}'] = df_tmp

#     df_vitals.rename(columns={
#         'HeartRate (bpm)': 'HeartRate',
#         'Systolic (mmHg)': 'Systolic',
#         'Diastolic (mmHg)': 'Diastolic',
#         'MAP (mmHg)': 'MAP',
#         'Respiration (Bpm)': 'Respiration',
#     }, inplace=True)
#     if 'TimeStamp (mS)' in df_vitals.columns:
#         df_vitals['datetime'] = pd.to_datetime(df_vitals['TimeStamp (mS)'], unit='ms')
#     elif 'Epoch Time' in df_vitals.columns:
#         df_vitals['datetime'] = pd.to_datetime(df_vitals['Epoch Time'], unit='ms')

#     # Ajustar a la zona horaria
#     df_vitals['datetime'] = df_vitals['datetime'].dt.tz_localize('UTC').dt.tz_convert('America/Los_Angeles')  # UTC-7/UTC-8
#     df_vitals['datetime'] = df_vitals['datetime'].dt.tz_localize(None)
    
#     # Convertir columnas a numéricas para poder aplicar filtros
#     for s in sensores_r:
#         for col in [f'SpO2_R{s}', f'PR_R{s}', f'Pi_R{s}']:
#             dfs_r[f'R{s}'][col] = pd.to_numeric(dfs_r[f'R{s}'][col], errors='coerce')
    
#     for col in ['MAP', 'HeartRate']:
#         df_vitals[col] = pd.to_numeric(df_vitals[col], errors='coerce')
    
#     # Filtrar datos de vitals
#     df_vitals = df_vitals[
#         (df_vitals['MAP'] <= 130) & 
#         ((df_vitals['HeartRate'] >= 40) & (df_vitals['HeartRate'] <= 250))
#     ]
    
#     return dfs_r, df_vitals

# def matriz_corr_quadrada(df, variaveis_corr):
#     corr = df[variaveis_corr].corr()
#     corr = corr.reindex(index=variaveis_corr, columns=variaveis_corr)
#     return corr

# def render():
#     st.markdown("<h2 style='text-align: center; font-size: 32px;'>Correlações entre sensores R e Vitals</h2>", unsafe_allow_html=True)

#     dfs_r, df_vitals = carregar_dados()
#     sensores = list(dfs_r.keys())

#     col1, col2 = st.columns(2)

#     with col1:
#         st.markdown("<h3 style='font-size: 24px;'>Matriz de correlação</h3>", unsafe_allow_html=True)
        
#         # Selector con texto más grande
#         st.markdown("<div style='font-size: 18px;'>Selecione o sensor R que deseja explorar junto ao sensor Vitals:</div>", unsafe_allow_html=True)
#         sensor_sel = st.selectbox("", sensores, index=2, label_visibility="collapsed")  # Padrão R3

#         variaveis_r = [f'SpO2_{sensor_sel}', f'PR_{sensor_sel}', f'Pi_{sensor_sel}']
#         variaveis_vitals_corr = ['MAP', 'Respiration']
#         variaveis_corr = variaveis_r + variaveis_vitals_corr

#         # Aplicar filtros al sensor R seleccionado
#         df_r_filtered = dfs_r[sensor_sel].copy()
#         df_r_filtered = df_r_filtered[
#             (df_r_filtered[f'SpO2_{sensor_sel}'] >= 90) & 
#             (df_r_filtered[f'Pi_{sensor_sel}'] >= 0.3) & 
#             (df_r_filtered[f'PR_{sensor_sel}'] >= 40) & 
#             (df_r_filtered[f'PR_{sensor_sel}'] <= 250)
#         ]

#         # Merge temporal con datos ya filtrados
#         df_merge = pd.merge_asof(
#             df_r_filtered.sort_values('datetime'),
#             df_vitals.sort_values('datetime'),
#             on='datetime',
#             direction='nearest',
#             tolerance=pd.Timedelta(milliseconds=1000)
#         )

#         # Limpeza: converte para numérico e remove valores inválidos ('-', '', None)
#         for col in variaveis_corr:
#             df_merge[col] = pd.to_numeric(df_merge[col], errors='coerce')

#         corr_matrix = matriz_corr_quadrada(df_merge, variaveis_corr)

#         # Plotly Heatmap
#         z_text = np.round(corr_matrix.values, 2).astype(str)
        
#         fig_heat = go.Figure(data=go.Heatmap(
#             z=corr_matrix.values,
#             x=variaveis_corr,
#             y=variaveis_corr,
#             colorscale='viridis',            
#             colorbar=dict(
#                 title="", 
#                 tickfont=dict(size=16, color="black")
#             ),
#             zmin=-1, zmax=1,
#             hoverongaps=False,
#             text=z_text,
#             texttemplate="%{text}",
#             textfont=dict(size=16, color="black")
#         ))

#         fig_heat.update_layout(
#             title=dict(
#                 text=f"Matriz de correlação para os sensores {sensor_sel} e Vitals",
#                 font=dict(size=22, color="black")
#             ),
#             xaxis=dict(
#                 tickangle=-90, 
#                 tickfont=dict(size=18, color="black"),
#                 title_font=dict(size=20, color="black")
#             ),
#             yaxis=dict(
#                 tickfont=dict(size=18, color="black"), 
#                 autorange="reversed",
#                 title_font=dict(size=20, color="black")
#             ),
#             width=500,
#             height=500,
#             font=dict(size=20, color="black"),
#             margin=dict(l=40, r=40, t=70, b=40),
#             paper_bgcolor='white',
#             plot_bgcolor='white'
#         )

#         st.plotly_chart(fig_heat, use_container_width=False)

#     with col2:
#         st.markdown("<h3 style='font-size: 24px;'>Visualização interativa das variáveis</h3>", unsafe_allow_html=True)
        
#         variaveis_vitals = ['MAP', 'Respiration', 'HeartRate']
#         # Selectors alinhados lado a lado
#         csel1, csel2 = st.columns(2)
#         with csel1:
#             st.markdown(f"<div style='font-size: 18px;'>Variáveis do sensor {sensor_sel}:</div>", unsafe_allow_html=True)
#             selected_r = st.multiselect(
#                 "",
#                 variaveis_r,
#                 default=[f'PR_{sensor_sel}', f'Pi_{sensor_sel}'],
#                 label_visibility="collapsed"
#             )
#         with csel2:
#             st.markdown("<div style='font-size: 18px;'>Variáveis do sensor Vitals:</div>", unsafe_allow_html=True)
#             selected_vitals = st.multiselect(
#                 "",
#                 variaveis_vitals,
#                 default=['MAP'],
#                 label_visibility="collapsed"
#             )

#         color_dict = {
#             f'SpO2_{sensor_sel}': 'blue',
#             f'PR_{sensor_sel}': 'red',
#             f'Pi_{sensor_sel}': 'green',
#             'MAP': 'black',
#             'Respiration': 'magenta',
#             'HeartRate': 'purple'
#         }

#         fig = go.Figure()

#         # Añadir datos de Rx
#         for col in selected_r:
#             # Si la variable seleccionada es Pi (del sensor actual), graficar en y2
#             if col == f'Pi_{sensor_sel}':
#                 fig.add_trace(go.Scatter(
#                     x=df_merge['datetime'], y=df_merge[col],
#                     mode='markers',
#                     marker=dict(color=color_dict.get(col, 'gray'), size=4),
#                     name=col,
#                     yaxis='y2'
#                 ))
#             else:
#                 fig.add_trace(go.Scatter(
#                     x=df_merge['datetime'], y=df_merge[col],
#                     mode='markers',
#                     marker=dict(color=color_dict.get(col, 'gray'), size=4),
#                     name=col
#                 ))

#         # Añadir datos de Vitals (siempre en el eje primario)
#         for col in selected_vitals:
#             fig.add_trace(go.Scatter(
#                 x=df_merge['datetime'], y=df_merge[col],
#                 mode='markers',
#                 marker=dict(color=color_dict.get(col, 'gray'), size=4),
#                 name=col
#             ))

#         # Solo mostrar el eje secundario si se seleccionó Pi
#         if f'Pi_{sensor_sel}' in selected_r:
#             fig.update_layout(
#                 yaxis2=dict(
#                     title=dict(
#                         text=f'Pi_{sensor_sel}',
#                         font=dict(size=20, color="black")
#                     ),
#                     overlaying='y',
#                     side='right',
#                     range=[0, 3],
#                     showgrid=False,
#                     tickfont=dict(size=16, color="black")
#                 )
#             )

#         fig.update_layout(
#             xaxis=dict(
#                 title=dict(
#                     text="Tempo",
#                     font=dict(size=20, color="black")
#                 ),
#                 tickfont=dict(size=16, color="black")
#             ),
#             yaxis=dict(
#                 title=dict(
#                     text="Valor",
#                     font=dict(size=20, color="black")
#                 ),
#                 tickfont=dict(size=16, color="black")
#             ),
#             legend=dict(
#                 title=dict(
#                     text="Variáveis",
#                     font=dict(size=18, color="black")
#                 ),
#                 font=dict(size=16, color="black")
#             ),
#             hovermode="x unified",
#             height=470,
#             margin=dict(t=10),
#             font=dict(size=16, color="black"),
#             paper_bgcolor='white',
#             plot_bgcolor='white'
#         )

#         st.plotly_chart(fig, use_container_width=True)

# if __name__ == "__main__":
#     render()








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