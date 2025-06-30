import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import random
import plotly.graph_objs as go
import plotly.colors

TICKERS_LIST = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA", "BRK-B", "JPM", "V",
    "UNH", "JNJ", "XOM", "PG", "MA", "HD", "CVX", "MRK", "ABBV", "PEP",
    "LLY", "KO", "AVGO", "COST", "WMT", "DIS", "BAC", "MCD", "ADBE", "PFE",
    "TMO", "CSCO", "DHR", "ABT", "ACN", "CRM", "TXN", "NKE", "VZ", "NEE",
    "CMCSA", "WFC", "LIN", "INTC", "QCOM", "MS", "HON", "PM", "UNP", "AMGN",
    "SBUX", "ORCL", "AMD", "CVS", "ISRG", "GILD", "BKNG", "NOW", "ZTS", "MDLZ"
]

def get_valid_random_stocks(n=50, days=30, max_lag=30, max_attempts=5):
    tried = set()
    valid_tickers = []
    attempt = 0
    while len(valid_tickers) < n and attempt < max_attempts:
        tickers_to_try = list(set(TICKERS_LIST) - tried)
        if not tickers_to_try:
            break
        sample_size = min(n - len(valid_tickers) + 10, len(tickers_to_try))
        tickers_sample = random.sample(tickers_to_try, sample_size)
        tried.update(tickers_sample)
        data = yf.download(tickers_sample, period=f'{days+max_lag+10}d', interval='1d', progress=False)
        if 'Adj Close' in data:
            prices = data['Adj Close']
        elif 'Close' in data:
            prices = data['Close']
        else:
            prices = pd.DataFrame()
        filtered = [col for col in prices.columns if prices[col].dropna().shape[0] >= days + max_lag]
        found = [col for col in filtered if col not in valid_tickers]
        valid_tickers.extend(found)
        attempt += 1
    if not valid_tickers:
        st.error("Não foi possível encontrar ações com dados válidos. Verifique sua conexão, tente novamente ou diminua o número de ações.")
        st.stop()
    if len(valid_tickers) < n:
        st.warning(f"Apenas {len(valid_tickers)} ações com dados válidos foram encontradas. Usando todas.")
        n = len(valid_tickers)
    chosen = valid_tickers[:n]
    final_data = yf.download(chosen, period=f'{days+max_lag+10}d', interval='1d', progress=False)
    if 'Adj Close' in final_data:
        available = [c for c in chosen if c in final_data['Adj Close'].columns]
        all_prices = final_data['Adj Close'][available]
    elif 'Close' in final_data:
        available = [c for c in chosen if c in final_data['Close'].columns]
        all_prices = final_data['Close'][available]
    else:
        st.error("Não foi possível baixar preços finais das ações selecionadas.")
        st.stop()
    return chosen, all_prices

def find_best_corr(target_series, candidate_series, max_lag=30):
    best_corr = -np.inf
    best_lag = None
    best_aligned = None
    idx = target_series.index
    for lag in range(-max_lag, 0):  # solo lags negativos
        shifted = candidate_series.shift(-lag)
        shifted = shifted.reindex(idx)
        valid = target_series.notnull() & shifted.notnull()
        if valid.sum() < 5:
            continue
        corr = target_series[valid].corr(shifted[valid])
        if np.isnan(corr):
            continue
        if abs(corr) > abs(best_corr):
            best_corr = corr
            best_lag = lag
            best_aligned = shifted
    if best_lag is None or best_aligned is None:
        return np.nan, np.nan, pd.Series(index=target_series.index, data=np.nan)
    return best_corr, best_lag, best_aligned

def show():
    st.title("Diagnóstico de Dados e Shift em Séries para Alinhamento")
    st.write(
        "Diagnóstico completo dos dados baixados e passo-a-passo de como ocorre o shift em uma das séries candidatas para detectar problemas de perda de dados ou alinhamento."
    )
    num_stocks = st.slider("Número de ações para explorar", 10, 50, 20)
    dias_historico = st.slider("Período de dados históricos (dias)", 30, 1000, 750)
    max_lag = 30
    tickers, all_prices = get_valid_random_stocks(num_stocks, days=dias_historico, max_lag=max_lag)

    st.subheader("Diagnóstico dos dados baixados")
    st.write("Shape do DataFrame original (linhas, colunas):", all_prices.shape)
    st.write("Primeiras datas:", all_prices.index[:3].tolist())
    st.write("Últimas datas:", all_prices.index[-3:].tolist())
    st.write("Ações:", list(all_prices.columns))
    st.write("Total de valores faltantes por ação:")
    st.dataframe(all_prices.isnull().sum())
    st.write("Total de valores faltantes (todas as ações):", int(all_prices.isnull().sum().sum()))
    st.write("Quantidade de valores válidos por ação:")
    st.dataframe(all_prices.notnull().sum())

    final_index = all_prices.index[-dias_historico:]
    st.write(f"Quantidade de datas finais para análise: {len(final_index)}")
    data = all_prices.loc[final_index]
    st.write("Valores faltantes por ação na janela final:")
    st.dataframe(data.isnull().sum())
    st.write("Total de valores faltantes (janela final):", int(data.isnull().sum().sum()))
    st.write("Quantidade de valores válidos por ação (janela final):")
    st.dataframe(data.notnull().sum())
    st.write("Shape do DataFrame final para análise (janela):", data.shape)

    # ----- Diagnóstico do shift para a primeira ação candidata -----
    action_selected = st.selectbox("Selecione a ação de interesse (target)", options=tickers)
    target_full = all_prices[action_selected]  # Serie target com todos os dados
    data_no_target = data.drop(columns=[action_selected])

    if len(data_no_target.columns) > 0:
        candidato = data_no_target.columns[0]
        orig_candidate = all_prices[candidato]  # Toda a serie candidata
        st.subheader(f"Diagnóstico detalhado do shift para: {candidato}")

        st.write("Target (full) - primeiros 5 valores:")
        st.write(target_full.head())
        st.write("Candidato (full) - primeiros 5 valores:")
        st.write(orig_candidate.head())

        lag_test = -30
        shifted = orig_candidate.shift(-lag_test)
        st.write(f"Candidato após shift({-lag_test}): primeiros 5 valores:")
        st.write(shifted.head())

        target_cut = target_full.loc[final_index]
        shifted_cut = shifted.reindex(final_index)

        st.write("Target (janela final) - primeiros 5 valores:")
        st.write(target_cut.head())
        st.write("Candidato (shifted, janela final) - primeiros 5 valores:")
        st.write(shifted_cut.head())

        valid = target_cut.notnull() & shifted_cut.notnull()
        st.write(f"Valores válidos após shift e reindex: {valid.sum()}")
        st.write("Correlación calculada (apenas valores válidos):")
        if valid.sum() > 0:
            corr = target_cut[valid].corr(shifted_cut[valid])
            st.write(corr)
        else:
            st.write("Sem valores válidos para correlacionar.")

    else:
        st.write("Não há ações candidatas para diagnosticar shift.")

if __name__ == "__main__":
    show()
