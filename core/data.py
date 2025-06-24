import yfinance as yf
import pandas as pd
import numpy as np

def get_weekly_log_returns(tickers, start='2022-01-01'):
    """
    Baixa preços de fechamento semanal dos ETFs e calcula log-retornos.
    """
    df = yf.download(tickers, start=start, auto_adjust=True, progress=False)['Close']
    df = df.resample('W-FRI').last()
    log_returns = pd.DataFrame()
    for ticker in tickers:
        # log_returns[ticker] = (df[ticker] / df[ticker].shift(1)).apply(lambda x: None if pd.isna(x) else pd.np.log(x))
        log_returns[ticker] = (df[ticker] / df[ticker].shift(1)).apply(lambda x: None if pd.isna(x) else np.log(x))
    log_returns = log_returns.dropna()
    return log_returns

def test_stationarity_adf(df):
    """
    Executa o teste de Dickey-Fuller aumentado para cada série.
    """
    from statsmodels.tsa.stattools import adfuller
    results = []
    for col in df.columns:
        serie = df[col].dropna()
        adf = adfuller(serie)
        results.append({
            'Ativo': col,
            'ADF': round(adf[0], 3),
            'Valor-p': round(adf[1], 4),
            'Estacionária': 'Sim' if adf[1] < 0.05 else 'Não'
        })
    return pd.DataFrame(results)
