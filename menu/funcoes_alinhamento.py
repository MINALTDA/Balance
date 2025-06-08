import pandas as pd
import numpy as np
import random
import plotly.graph_objects as go

def carregar_e_limpar_r3(path):
    df_r3 = pd.read_csv(path)
    df_r3['SpO2'] = pd.to_numeric(df_r3['SpO2 % Value'], errors='coerce')
    df_r3['PR'] = pd.to_numeric(df_r3['PR bpm Value'], errors='coerce')
    df_r3['Pi'] = pd.to_numeric(df_r3['Pi Value'], errors='coerce')
    df_r3['Epoch'] = pd.to_numeric(df_r3[' Epoch Time'], errors='coerce')
    df_r3_limpo = df_r3[(df_r3['SpO2'] > 90) & (df_r3['Pi'] > 0.3)].copy()
    df_r3_limpo.reset_index(drop=True, inplace=True)
    return df_r3_limpo

def carregar_e_limpar_r3_consolidado(path):
    df = pd.read_csv(path)
    df_r3 = df[[
        'SpO2_%_Value_3', 
        'PR_bpm_Value_3', 
        'Pi_Value_3',
        'Epoch Time'
    ]].copy()
    df_r3.rename(columns={
        'SpO2_%_Value_3': 'SpO2',
        'PR_bpm_Value_3': 'PR',
        'Pi_Value_3': 'Pi',
        'Epoch Time': 'Epoch'
    }, inplace=True)
    df_r3['SpO2'] = pd.to_numeric(df_r3['SpO2'], errors='coerce')
    df_r3['PR'] = pd.to_numeric(df_r3['PR'], errors='coerce')
    df_r3['Pi'] = pd.to_numeric(df_r3['Pi'], errors='coerce')
    df_r3['Epoch'] = pd.to_numeric(df_r3['Epoch'], errors='coerce')
    df_r3_limpo = df_r3[(df_r3['SpO2'] > 90) & (df_r3['Pi'] > 0.3)].copy()
    df_r3_limpo.reset_index(drop=True, inplace=True)
    return df_r3_limpo

def carregar_vitals(path, min_hr=40, max_hr=250):
    df_vitals = pd.read_csv(path)
    df_vitals['TimeStamp'] = pd.to_numeric(df_vitals['TimeStamp (mS)'], errors='coerce')
    df_vitals['HeartRate'] = pd.to_numeric(df_vitals['HeartRate (bpm)'], errors='coerce')
    total = len(df_vitals)
    mask = (df_vitals['HeartRate'] >= min_hr) & (df_vitals['HeartRate'] <= max_hr)
    df_vitals_limpo = df_vitals[mask].copy()
    df_vitals_limpo.reset_index(drop=True, inplace=True)
    filtrados = total - len(df_vitals_limpo)
    print(f"[carregar_vitals] Registros originais: {total}")
    print(f"[carregar_vitals] Registros removidos por HeartRate fora de [{min_hr},{max_hr}]: {filtrados}")
    print(f"[carregar_vitals] Registros finais: {len(df_vitals_limpo)}")
    return df_vitals_limpo

def ajuste_tempo_zero(df_r3, df_vitals):
    tempo_inicial_r3 = df_r3['Epoch'].min()
    tempo_inicial_vitals = df_vitals['TimeStamp'].min()
    df_r3['tempo_relativo'] = df_r3['Epoch'] - tempo_inicial_r3
    df_vitals['tempo_relativo'] = df_vitals['TimeStamp'] - tempo_inicial_vitals
    return df_r3, df_vitals

def encontrar_registros_similares(df_r3, valor_vitals, max_desvio=2):
    return df_r3[(df_r3['PR'] >= valor_vitals - max_desvio) &
                 (df_r3['PR'] <= valor_vitals + max_desvio)]

def calcular_metricas_alinhamento(df_r3, df_vitals, desfase, max_diff_tempo=500, limiar_penalizacao=3, penalizacoes_minimas=50):
    df_r3_temp = df_r3.copy()
    df_r3_temp['tempo_ajustado'] = df_r3_temp['tempo_relativo'] + desfase
    df_vitals_temp = df_vitals
    erros_absolutos = []
    num_penalizacoes = 0
    num_comparacoes = 0
    idx_r3 = 0
    for idx_vitals, row_vitals in df_vitals_temp.iterrows():
        tempo_vitals = row_vitals['tempo_relativo']
        valor_vitals = row_vitals['HeartRate']
        while idx_r3 < len(df_r3_temp):
            tempo_r3 = df_r3_temp.iloc[idx_r3]['tempo_ajustado']
            if tempo_r3 > tempo_vitals + max_diff_tempo:
                break
            if abs(tempo_r3 - tempo_vitals) <= max_diff_tempo:
                valor_r3 = df_r3_temp.iloc[idx_r3]['PR']
                erro = abs(valor_r3 - valor_vitals)
                erros_absolutos.append(erro)
                if erro > limiar_penalizacao:
                    num_penalizacoes += 1
                    if num_penalizacoes >= penalizacoes_minimas:
                        return {
                            'desfase': desfase,
                            'media_erros': float('inf'),
                            'num_penalizacoes': num_penalizacoes,
                            'num_comparacoes': num_comparacoes,
                            'metrica': float('inf')
                        }
                num_comparacoes += 1
                break
            idx_r3 += 1
    media_erros = np.mean(erros_absolutos) if erros_absolutos else float('inf')
    metrica = media_erros / num_comparacoes if num_comparacoes > 0 else float('inf')
    return {
        'desfase': desfase,
        'media_erros': media_erros,
        'num_penalizacoes': num_penalizacoes,
        'num_comparacoes': num_comparacoes,
        'metrica': metrica
    }

def alinhar_sensores_heuristico(df_r3, df_vitals, num_pontos_aleatorios=20, max_desvio=2, 
                               max_diff_tempo=500, limiar_penalizacao=3, min_comparacoes_percentual=0.1):
    indices_aleatorios = random.sample(range(len(df_vitals)), min(num_pontos_aleatorios, len(df_vitals)))
    pontos_aleatorios = df_vitals.iloc[indices_aleatorios]
    min_comparacoes = min(int(len(df_r3) * min_comparacoes_percentual), int(len(df_vitals) * min_comparacoes_percentual))
    resultados = []
    num_penalizacoes = len(df_vitals)
    for _, ponto_vitals in pontos_aleatorios.iterrows():
        valor_vitals = ponto_vitals['HeartRate']
        tempo_vitals = ponto_vitals['tempo_relativo']
        registros_similares = encontrar_registros_similares(df_r3, valor_vitals, max_desvio)
        for _, registro_r3 in registros_similares.iterrows():
            tempo_r3 = registro_r3['tempo_relativo']
            desfase = tempo_vitals - tempo_r3
            metricas = calcular_metricas_alinhamento(
                df_r3, df_vitals, desfase, max_diff_tempo, limiar_penalizacao, num_penalizacoes
            )
            num_penalizacoes = metricas['num_penalizacoes']
            if metricas['num_comparacoes'] >= min_comparacoes:
                resultados.append(metricas)
    if not resultados:
        return None
    resultados_ordenados = sorted(resultados, key=lambda x: (x['num_penalizacoes'], x['metrica']))
    return resultados_ordenados[0]

def aplicar_desfase(df, desfase, coluna_tempo='tempo_relativo'):
    df_ajustado = df.copy()
    df_ajustado[f'{coluna_tempo}_ajustado'] = df_ajustado[coluna_tempo] + desfase
    return df_ajustado

def visualizar_alinhamento(df_r3, df_vitals, desfase=0):
    df_r3_ajustado = aplicar_desfase(df_r3, desfase)
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df_r3_ajustado['tempo_relativo_ajustado']/1000,
        y=df_r3_ajustado['PR'],
        mode='markers',
        marker=dict(size=2, color='blue', opacity=0.7),
        name='PR bpm - R3'
    ))
    fig.add_trace(go.Scatter(
        x=df_vitals['tempo_relativo']/1000,
        y=df_vitals['HeartRate'],
        mode='markers',
        marker=dict(size=2, color='red', opacity=0.7),
        name='HeartRate - vitals'
    ))
    fig.update_layout(
        title={
            'text': f'Alinhamento entre PR (R3) e HeartRate (vitals) - Desfase: {desfase/1000:.2f} segundos',
            'font': {'size': 24, 'color': 'black'}
        },
        xaxis=dict(
            title=dict(text='Tempo (segundos desde o início)', font=dict(size=20, color='black')),
            tickfont=dict(size=16, color='black')
        ),
        yaxis=dict(
            title=dict(text='Batimentos por minuto (bpm)', font=dict(size=20, color='black')),
            tickfont=dict(size=16, color='black')
        ),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            font=dict(size=16, color='black')
        ),
        font=dict(size=16, color='black'),
        height=600,
        hovermode='closest',
        dragmode='zoom',
        showlegend=True
    )
    return fig

def visualizar_ajuste_tempo_zero_interativo(df_r3, df_vitals):
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df_r3['tempo_relativo']/1000,
        y=df_r3['PR'],
        mode='markers',
        marker=dict(size=2, color='blue', opacity=0.7),
        name='PR bpm - R3'
    ))
    fig.add_trace(go.Scatter(
        x=df_vitals['tempo_relativo']/1000,
        y=df_vitals['HeartRate'],
        mode='markers',
        marker=dict(size=2, color='red', opacity=0.7),
        name='HeartRate - vitals'
    ))
    fig.update_layout(
        title={
            'text': 'Comparação entre PR (R3) e HeartRate (vitals) - Ajuste de tempo zero',
            'font': {'size': 24, 'color': 'black'}
        },
        xaxis=dict(
            title=dict(text='Tempo (segundos desde o início)', font=dict(size=20, color='black')),
            tickfont=dict(size=16, color='black')
        ),
        yaxis=dict(
            title=dict(text='Batimentos por minuto (bpm)', font=dict(size=20, color='black')),
            tickfont=dict(size=16, color='black')
        ),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            font=dict(size=16, color='black')
        ),
        font=dict(size=16, color='black'),
        height=600,
        hovermode='closest',
        dragmode='zoom',
        showlegend=True
    )
    return fig