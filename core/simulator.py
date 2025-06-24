import numpy as np
import pandas as pd

def backtest_impulso_resposta(
    df, 
    ativo_impulso, 
    ativo_resposta, 
    direcao='short',  # ou 'long'
    threshold=1.5, 
    janela=52
):
    """
    Simula uma estratégia baseada em choques no ativo_impulso e posição no ativo_resposta.
    - df: DataFrame de retornos logarítmicos semanais
    - ativo_impulso: ETF que gera o choque
    - ativo_resposta: ETF para operar
    - direcao: 'short' (venda) ou 'long' (compra)
    - threshold: múltiplos de desvio padrão para definir choque
    - janela: quantas semanas olhar para média/desvio
    """
    sinais = []
    retornos_estrategia = []
    datas = []

    for i in range(janela, len(df)):
        janela_serie = df[ativo_impulso].iloc[i-janela:i]
        media = janela_serie.mean()
        std = janela_serie.std()
        ret_impulso = df[ativo_impulso].iloc[i]
        ret_resposta = df[ativo_resposta].iloc[i]
        # Checa choque positivo
        if ret_impulso > media + threshold * std:
            sinais.append(1)
            if direcao == 'short':
                retorno = -ret_resposta  # venda: ganha se cair
            else:
                retorno = ret_resposta   # compra: ganha se subir
        else:
            sinais.append(0)
            retorno = 0
        retornos_estrategia.append(retorno)
        datas.append(df.index[i])

    resultado = pd.DataFrame({
        'Data': datas,
        'Sinal': sinais,
        'Retorno_estrategia': retornos_estrategia,
        'Retorno_buy_hold': df[ativo_resposta].iloc[janela:].values
    })
    resultado['Retorno_acumulado_estrategia'] = resultado['Retorno_estrategia'].cumsum()
    resultado['Retorno_acumulado_buy_hold'] = resultado['Retorno_buy_hold'].cumsum()
    return resultado
