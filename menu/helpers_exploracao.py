import yfinance as yf
import numpy as np
import pandas as pd
from itertools import combinations
import streamlit as st

@st.cache_data(show_spinner=False)
def load_prices_explor(tickers):
    df = yf.download(tickers, start="2018-01-01", progress=False)["Close"]
    if isinstance(df, pd.Series):
        df = df.to_frame()
    return df

def buscar_lags(prices, tickers, lookback, max_lag, criterio):
    resultados = []
    for t1, t2 in combinations(tickers, 2):
        melhor_valor = None
        melhor_lag = 0
        for lag in range(-max_lag, max_lag + 1):
            if lag == 0:
                continue
            serie1 = prices[t1]
            serie2 = prices[t2].shift(lag)
            df = pd.DataFrame({t1: serie1, t2: serie2}).dropna()
            if len(df) < lookback + 10:
                continue
            spread = np.log(df[t1]) - np.log(df[t2])
            if criterio == "Sharpe Ratio":
                ret = spread.diff().dropna()
                metric = ret.mean() / ret.std() * np.sqrt(252) if ret.std() > 0 else -np.inf
                comp = lambda x, y: x > y
            elif criterio == "Autocorrelação (absoluta)":
                ret = spread.diff().dropna()
                metric = abs(pd.Series(ret).autocorr(lag=1))
                comp = lambda x, y: x < y
            elif criterio == "Drawdown mínimo":
                cumul = ret = spread.diff().cumsum().apply(np.exp)
                metric = ((cumul.cummax() - cumul) / cumul.cummax()).max()
                comp = lambda x, y: x < y
            else:
                continue
            if (melhor_valor is None) or comp(metric, melhor_valor):
                melhor_valor = metric
                melhor_lag = lag
        if melhor_valor is not None:
            resultados.append({
                "Par": f"{t1}-{t2}",
                "Lag ótimo": melhor_lag,
                "Valor do critério": melhor_valor
            })
    # Ordena conforme critério (maior Sharpe, menor autocorr/drawdown)
    if criterio == "Sharpe Ratio":
        resultados = sorted(resultados, key=lambda x: -x["Valor do critério"])
    else:
        resultados = sorted(resultados, key=lambda x: x["Valor do critério"])
    return resultados

def montar_dataframe_resultados(resultados):
    if not resultados:
        return pd.DataFrame(columns=["Par", "Lag ótimo", "Valor do critério"])
    return pd.DataFrame(resultados)
