#menu/exploracao.py
import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import random
import matplotlib.pyplot as plt

TICKERS_LIST = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA", "BRK-B", "JPM", "V",
    "UNH", "JNJ", "XOM", "PG", "MA", "HD", "CVX", "MRK", "ABBV", "PEP",
    "LLY", "KO", "AVGO", "COST", "WMT", "DIS", "BAC", "MCD", "ADBE", "PFE",
    "TMO", "CSCO", "DHR", "ABT", "ACN", "CRM", "TXN", "NKE", "VZ", "NEE",
    "CMCSA", "WFC", "LIN", "INTC", "QCOM", "MS", "HON", "PM", "UNP", "AMGN",
    "SBUX", "ORCL", "AMD", "CVS", "ISRG", "GILD", "BKNG", "NOW", "ZTS", "MDLZ"
]

def get_valid_random_stocks(n=50, days=30, max_attempts=5):
    tried = set()
    valid_tickers = []
    attempt = 0
    while len(valid_tickers) < n and attempt < max_attempts:
        tickers_to_try = list(set(TICKERS_LIST) - tried)
        if len(tickers_to_try) == 0:
            break
        sample_size = min(n - len(valid_tickers) + 10, len(tickers_to_try))
        tickers_sample = random.sample(tickers_to_try, sample_size)
        tried.update(tickers_sample)
        data = yf.download(tickers_sample, period=f'{days+10}d', interval='1d', progress=False)
        if 'Adj Close' in data:
            prices = data['Adj Close'].dropna(axis=1, how='any')
        elif 'Close' in data:
            prices = data['Close'].dropna(axis=1, how='any')
        else:
            prices = pd.DataFrame()
        found = [col for col in prices.columns if col not in valid_tickers]
        valid_tickers.extend(found)
        attempt += 1
    if len(valid_tickers) == 0:
        st.error("Não foi possível encontrar ações com dados válidos. Verifique sua conexão, tente novamente ou diminua o número de ações.")
        st.stop()
    if len(valid_tickers) < n:
        st.warning(f"Apenas {len(valid_tickers)} ações com dados válidos foram encontradas. Usando todas.")
        n = len(valid_tickers)
    chosen = valid_tickers[:n]
    final_data = yf.download(chosen, period=f'{days+10}d', interval='1d', progress=False)

    if 'Adj Close' in final_data:
        available = [c for c in chosen if c in final_data['Adj Close'].columns]
        if not available:
            st.error("Nenhuma das ações selecionadas possui dados válidos em 'Adj Close'.")
            st.stop()
        final_prices = final_data['Adj Close'][available].dropna(axis=0, how='any')[-days:]
    elif 'Close' in final_data:
        available = [c for c in chosen if c in final_data['Close'].columns]
        if not available:
            st.error("Nenhuma das ações selecionadas possui dados válidos em 'Close'.")
            st.stop()
        final_prices = final_data['Close'][available].dropna(axis=0, how='any')[-days:]
    else:
        st.error("Não foi possível baixar preços finais das ações selecionadas.")
        st.stop()

    return chosen, final_prices

def best_alignment_greedy(data):
    tickers = list(data.columns)
    used = [tickers[0]]
    unused = set(tickers[1:])
    order = [tickers[0]]
    while unused:
        last = order[-1]
        next_ticker = max(unused, key=lambda x: abs(data[last].corr(data[x])))
        order.append(next_ticker)
        unused.remove(next_ticker)
    return order

def calc_corr_mean(data, order=None):
    cols = order if order is not None else list(data.columns)
    pair_corrs = [abs(data[t].corr(data[s])) for t, s in zip(cols[:-1], cols[1:])]
    corr_mean = np.mean(pair_corrs)
    return corr_mean

def plot_series(data, order=None, title=""):
    fig, ax = plt.subplots(figsize=(12, 6))
    if order:
        for ticker in order:
            ax.plot(data.index, data[ticker], label=ticker, alpha=0.6)
    else:
        for ticker in data.columns:
            ax.plot(data.index, data[ticker], label=ticker, alpha=0.6)
    ax.set_title(title)
    ax.legend(fontsize=6, ncol=5)
    st.pyplot(fig)

def show():
    st.title("Exploração: Alineamento Ótimo de Séries de Ações")
    st.write(
        "Seleciona ações aleatórias do mercado americano, verifica quais possuem dados válidos e explora como alinhá-las para maximizar a média das correlações entre pares consecutivos. Observe a diferença visual e estatística do alinhamento."
    )
    num_stocks = st.slider("Número de ações para explorar", 10, 50, 30)
    dias_historico = st.slider("Período de dados históricos (dias)", 30, 1000, 750)
    tickers, data = get_valid_random_stocks(num_stocks, days=dias_historico)
    st.write(f"Ações válidas selecionadas ({len(tickers)}):", ", ".join(tickers))
    st.dataframe(data)

    st.subheader("Séries não alinhadas (ordem aleatória)")
    plot_series(data, title="Séries originais")
    orig_corr_mean = calc_corr_mean(data)

    st.subheader("Séries alinhadas (máxima correlação entre pares consecutivos)")
    aligned_order = best_alignment_greedy(data)
    plot_series(data, order=aligned_order, title="Séries alinhadas pela heurística")
    aligned_corr_mean = calc_corr_mean(data, order=aligned_order)

    st.markdown("### Média das correlações entre pares consecutivos")
    col1, col2 = st.columns(2)
    with col1:
        st.info(f"**Média das correlações - original:** {orig_corr_mean:.3f}")
    with col2:
        st.success(f"**Média das correlações - alinhado:** {aligned_corr_mean:.3f}")

    # ---- SALVA NO SESSION_STATE (inclui dias_historico!) ----
    st.session_state['tickers_alinhados'] = tickers
    st.session_state['dados_alinhados'] = data
    st.session_state['ordem_alinhada'] = aligned_order
    st.session_state['dias_historico'] = dias_historico

if __name__ == "__main__":
    show()
