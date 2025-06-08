# previsao.py

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import plotly.graph_objects as go

def carregar_dados():
    df_r = pd.read_csv("data/R_alinhados.csv")
    df_vitals = pd.read_csv("data/vitals_alinhado.csv")

    sensores_r = ['1', '2', '3']
    dict_vars = {
        'SpO2': 'SpO2_%_Value_',
        'PR': 'PR_bpm_Value_',
        'Pi': 'Pi_Value_',
        'Events': 'Events_'
    }

    dfs_r = {}
    for s in sensores_r:
        cols = ['Epoch Time'] + [v + s for v in dict_vars.values()]
        df_tmp = df_r[cols].copy()
        df_tmp.rename(columns={
            f'SpO2_%_Value_{s}': f'SpO2_R{s}',
            f'PR_bpm_Value_{s}': f'PR_R{s}',
            f'Pi_Value_{s}': f'Pi_R{s}',
            f'Events_{s}': f'Events_R{s}'
        }, inplace=True)
        df_tmp['datetime'] = pd.to_datetime(df_tmp['Epoch Time'], unit='ms')
        # Ajustar a la zona horaria local (ajusta el número según tu zona horaria)
        df_tmp['datetime'] = df_tmp['datetime'].dt.tz_localize('UTC').dt.tz_convert('America/Los_Angeles')  # UTC-7/UTC-8
        # Eliminar información de zona horaria para evitar problemas de visualización
        df_tmp['datetime'] = df_tmp['datetime'].dt.tz_localize(None)
        
        # Convertir columnas a numéricas
        for col in [f'SpO2_R{s}', f'PR_R{s}', f'Pi_R{s}']:
            df_tmp[col] = pd.to_numeric(df_tmp[col], errors='coerce')
        
        # Aplicar filtros al sensor R - con manejo más flexible para evitar filtrar todos los datos
        valid_data = df_tmp.dropna(subset=[f'SpO2_R{s}', f'PR_R{s}', f'Pi_R{s}'])
        
        # Aplicar filtros solo si hay suficientes datos después de filtrar
        filtered_tmp = valid_data[
            (valid_data[f'SpO2_R{s}'] >= 90) & 
            (valid_data[f'Pi_R{s}'] >= 0.3) & 
            (valid_data[f'PR_R{s}'] >= 40) & 
            (valid_data[f'PR_R{s}'] <= 250)
        ]
        
        # Si después de filtrar quedan muy pocos datos, usar los datos originales con una advertencia
        if len(filtered_tmp) < 100:  # umbral arbitrario, ajustar según sea necesario
            dfs_r[f'R{s}'] = valid_data
            dfs_r[f'R{s}_filtered'] = False
        else:
            dfs_r[f'R{s}'] = filtered_tmp
            dfs_r[f'R{s}_filtered'] = True
    
    # Preparar datos de vitals
    df_vitals.rename(columns={
        'HeartRate (bpm)': 'HeartRate',
        'Systolic (mmHg)': 'Systolic',
        'Diastolic (mmHg)': 'Diastolic',
        'MAP (mmHg)': 'MAP',
        'Respiration (Bpm)': 'Respiration',
    }, inplace=True)
    
    if 'TimeStamp (mS)' in df_vitals.columns:
        df_vitals['datetime'] = pd.to_datetime(df_vitals['TimeStamp (mS)'], unit='ms')
    elif 'Epoch Time' in df_vitals.columns:
        df_vitals['datetime'] = pd.to_datetime(df_vitals['Epoch Time'], unit='ms')

    # Ajustar a la zona horaria
    df_vitals['datetime'] = df_vitals['datetime'].dt.tz_localize('UTC').dt.tz_convert('America/Los_Angeles')  # UTC-7/UTC-8
    df_vitals['datetime'] = df_vitals['datetime'].dt.tz_localize(None)
    
    # Convertir MAP y HeartRate a numéricos
    for col in ['MAP', 'HeartRate']:
        if col in df_vitals.columns:
            df_vitals[col] = pd.to_numeric(df_vitals[col], errors='coerce')
    
    # Asegurarse de que MAP existe
    if 'MAP' not in df_vitals.columns and 'MAP (mmHg)' in df_vitals.columns:
        df_vitals['MAP'] = pd.to_numeric(df_vitals['MAP (mmHg)'], errors='coerce')
    
    # Filtrar datos de vitals - con manejo más flexible
    valid_vitals = df_vitals.dropna(subset=['MAP'])
    
    if 'HeartRate' in valid_vitals.columns:
        filtered_vitals = valid_vitals[
            (valid_vitals['MAP'] <= 130) & 
            ((valid_vitals['HeartRate'] >= 40) & (valid_vitals['HeartRate'] <= 250))
        ]
    else:
        filtered_vitals = valid_vitals[valid_vitals['MAP'] <= 130]
    
    # Si después de filtrar quedan muy pocos datos, usar los datos originales
    if len(filtered_vitals) < 100:
        return dfs_r, valid_vitals, False
    else:
        return dfs_r, filtered_vitals, True

def render():
    st.markdown("<h2 style='text-align: center; font-size: 32px;'>Estimativa da MAP a partir das variáveis PR e Pi de sensores R</h2>", unsafe_allow_html=True)
    
    dfs_r, df_vitals, vitals_filtered = carregar_dados()
    sensores = [k for k in dfs_r.keys() if not k.endswith('_filtered')]
    
    if not sensores:
        st.error("Não foram encontrados dados de sensores R. Verifique os arquivos de dados.")
        return
    
    # Crear dos columnas para el selector y las métricas de test
    col_selector, col_test_metrics = st.columns([1, 1])
    
    with col_selector:
        # Selector de sensor R con texto más grande
        default_index = min(2, len(sensores)-1)
        st.markdown("<div style='font-size: 18px;'>Selecione o sensor R para estimativa da MAP:</div>", unsafe_allow_html=True)
        sensor_sel = st.selectbox("", sensores, index=default_index, label_visibility="collapsed")
    
    # Verificar si se aplicaron filtros y mostrar advertencia si es necesario
    if not dfs_r.get(f'{sensor_sel}_filtered', True) or not vitals_filtered:
        st.warning("""
        ⚠️ Aviso: Alguns filtros de qualidade de dados foram relaxados porque havia poucos dados disponíveis após a filtragem.
        Os resultados podem não ser tão precisos quanto o esperado.
        """)
    
    # Alinhar datasets por tempo
    df_merge = pd.merge_asof(
        dfs_r[sensor_sel].sort_values('datetime'),
        df_vitals[['datetime', 'MAP']].sort_values('datetime'),
        on='datetime',
        direction='nearest',
        tolerance=pd.Timedelta(milliseconds=1000)
    )
    
    # Eliminar filas con NaN
    df_merge = df_merge.dropna(subset=[f'PR_{sensor_sel}', f'Pi_{sensor_sel}', 'MAP'])
    
    # Verificar si hay suficientes datos para el modelo
    if len(df_merge) < 10:
        st.error(f"""
        ❌ Erro: Não há dados suficientes para criar um modelo com o sensor {sensor_sel}.
        Por favor, selecione outro sensor ou ajuste os critérios de filtragem.
        """)
        return
    
    # Preparar dados para o modelo
    X = df_merge[[f'PR_{sensor_sel}', f'Pi_{sensor_sel}']]
    y = df_merge['MAP']
    
    # Ajustar el tamaño de test_size según la cantidad de datos disponibles
    test_size = min(0.4, 0.8 * (1 - 10/len(df_merge)))
    
    try:
        # Divisão treino/validação/teste
        X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=test_size, random_state=42)
        X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)
        
        # Modelo Random Forest
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        
        # Previsões
        y_train_pred = model.predict(X_train)
        y_val_pred = model.predict(X_val)
        y_test_pred = model.predict(X_test)
        
        # Mostrar métricas de test junto al selector en una línea
        with col_test_metrics:
            # Calcular métricas de test
            mse_test = mean_squared_error(y_test, y_test_pred)
            rmse_test = np.sqrt(mse_test)
            r2_test = r2_score(y_test, y_test_pred)
            
            # Mostrar métricas en una línea con texto más grande
            st.markdown(
                f"<div style='font-size: 18px; padding-top: 25px; text-align: center;'><b>Métricas de teste:</b> MSE: {mse_test:.2f} | RMSE: {rmse_test:.2f} | R²: {r2_test:.2f}</div>", 
                unsafe_allow_html=True
            )
        
        # Criar duas colunas para os gráficos
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("<h3 style='text-align: center; font-size: 24px;'>Comparação: MAP real vs estimada</h3>", unsafe_allow_html=True)
            
            # Gráfico de dispersão com Plotly
            fig_scatter = go.Figure()
            
            fig_scatter.add_trace(go.Scatter(
                x=y_test, 
                y=y_test_pred,
                mode='markers',
                marker=dict(color='black', size=10, opacity=0.5),
                name='Pontos de teste'
            ))
            
            # Linha de referência (diagonal)
            min_val = min(y_test.min(), y_test_pred.min())
            max_val = max(y_test.max(), y_test_pred.max())
            
            fig_scatter.add_trace(go.Scatter(
                x=[min_val, max_val],
                y=[min_val, max_val],
                mode='lines',
                line=dict(color='orange', dash='dash', width=3),
                name='Linha ideal'
            ))
            
            # Configuración con texto más grande
            fig_scatter.update_layout(
                xaxis_title="MAP Real",
                yaxis_title="MAP Estimada",
                height=400,
                margin=dict(l=40, r=40, t=10, b=40),
                font=dict(
                    size=18,  # Tamaño de fuente general
                ),
                legend=dict(
                    font=dict(size=16)  # Tamaño de fuente de la leyenda
                )
            )
            
            # Aumentar tamaño de los títulos de los ejes
            fig_scatter.update_xaxes(title_font=dict(size=20, color="black"), tickfont=dict(size=16, color="black"))
            fig_scatter.update_yaxes(title_font=dict(size=20, color="black"), tickfont=dict(size=16, color="black"))
            
            st.plotly_chart(fig_scatter, use_container_width=True)
        
        with col2:
            st.markdown("<h3 style='text-align: center; font-size: 24px;'>MAP real vs estimada no tempo</h3>", unsafe_allow_html=True)
            
            # Obter índices originais dos dados de teste
            test_indices = X_test.index
            
            # Criar dataframe com resultados de teste
            df_test_results = pd.DataFrame({
                'datetime': df_merge.loc[test_indices, 'datetime'],
                'MAP_real': y_test,
                'MAP_estimado': y_test_pred
            }).sort_values('datetime')
            
            # Gráfico de série temporal com Plotly
            fig_time = go.Figure()
            
            fig_time.add_trace(go.Scatter(
                x=df_test_results['datetime'],
                y=df_test_results['MAP_real'],
                mode='lines+markers',
                name='MAP Real',
                line=dict(color='blue', width=3),
                marker=dict(size=5)
            ))
            
            fig_time.add_trace(go.Scatter(
                x=df_test_results['datetime'],
                y=df_test_results['MAP_estimado'],
                mode='lines+markers',
                name='MAP Estimada',
                line=dict(color='red', width=3),
                marker=dict(size=5)
            ))
            
            # Configuración con texto más grande
            fig_time.update_layout(
                xaxis_title="Tempo",
                yaxis_title="MAP",
                height=400,
                margin=dict(l=40, r=40, t=10, b=40),
                font=dict(
                    size=18,  # Tamaño de fuente general
                    color="black"  # Color de fuente general
                ),
                legend=dict(
                    orientation="h", 
                    yanchor="bottom", 
                    y=1.02, 
                    xanchor="right", 
                    x=1,
                    font=dict(size=16)  # Tamaño de fuente de la leyenda
                )
            )
            
            # Aumentar tamaño de los títulos de los ejes
            fig_time.update_xaxes(title_font=dict(size=20, color="black"), tickfont=dict(size=16, color="black"))
            fig_time.update_yaxes(title_font=dict(size=20, color="black"), tickfont=dict(size=16, color="black"))
            
            st.plotly_chart(fig_time, use_container_width=True)
        
        # Métricas detalladas después de los gráficos
        st.markdown("<h3 style='text-align: center; font-size: 24px;'>Métricas do modelo</h3>", unsafe_allow_html=True)
        
        # Crear dos columnas para las métricas
        col_metrics1, col_metrics2 = st.columns(2)
        
        with col_metrics1:
            # Crear un DataFrame para todas las métricas
            metrics_data = []
            
            # Función para calcular métricas y añadirlas al DataFrame
            def calculate_metrics(y_true, y_pred, split):
                mse = mean_squared_error(y_true, y_pred)
                rmse = np.sqrt(mse)
                r2 = r2_score(y_true, y_pred)
                metrics_data.append({
                    'Conjunto': split,
                    'MSE': mse,
                    'RMSE': rmse,
                    'R²': r2
                })
            
            # Calcular métricas para cada conjunto
            calculate_metrics(y_train, y_train_pred, "Treino")
            calculate_metrics(y_val, y_val_pred, "Validação")
            calculate_metrics(y_test, y_test_pred, "Teste")
            
            # Crear DataFrame y mostrar como tabla
            metrics_df = pd.DataFrame(metrics_data)
            
            # Título con tamaño grande
            st.markdown("<h4 style='font-size: 22px;'>Métricas de avaliação</h4>", unsafe_allow_html=True)
                                  
            st.dataframe(
                metrics_df.style.format({
                    'MSE': '{:.2f}',
                    'RMSE': '{:.2f}',
                    'R²': '{:.2f}'
                }),
                hide_index=True,
                height=150  # Ajustar altura para mejor visualización
            )
        
        with col_metrics2:
            # Coeficientes de Determinação das Variáveis
            st.markdown("<h4 style='font-size: 22px;'>Coeficientes de determinação das variáveis</h4>", unsafe_allow_html=True)
            
            # Crear una tabla en lugar de un gráfico de barras
            feature_importance = pd.DataFrame({
                'Variável': [f'PR_{sensor_sel}', f'Pi_{sensor_sel}'],
                'Coeficiente': model.feature_importances_
            }).sort_values('Coeficiente', ascending=False)
            
            # Formatear los valores como porcentajes
            feature_importance['Contribuição (%)'] = feature_importance['Coeficiente'] * 100
            
            # Mostrar como tabla formateada con texto más grande
            st.dataframe(
                feature_importance[['Variável', 'Contribuição (%)']].style.format({
                    'Contribuição (%)': '{:.2f}%'
                }),
                hide_index=True,
                height=150  # Ajustar altura para mejor visualización
            )
    
    except Exception as e:
        st.error(f"""
        ❌ Erro ao criar o modelo: {str(e)}
        
        Isso pode ocorrer devido a:
        - Dados insuficientes após a filtragem
        - Problemas de qualidade nos dados
        - Incompatibilidade entre os sensores
        
        Tente selecionar outro sensor ou ajustar os critérios de filtragem.
        """)
        st.write(f"Número de registros disponíveis após filtragem e alinhamento: {len(df_merge)}")

if __name__ == "__main__":
    render()