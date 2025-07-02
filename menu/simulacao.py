import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from itertools import combinations
from datetime import timedelta
import plotly.graph_objs as go
import matplotlib.pyplot as plt

TICKERS = [
    'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'META', 'NVDA',
    'TLT', 'USO', 'QQQ', 'XLF', 'XLY', 'SPY', 'DIA', 'IWM', 'EEM',
    'GLD', 'SLV', 'EWZ', 'VNQ', 'XLV', 'XLI', 'XLB', 'XLE', 'XLK', 'XLP'
]

def show():
    st.markdown(
        "<h1 style='color:#0070F3; font-weight:800;'>Simulação Avançada: Pair Trading Lead-Lag vs Tradicional</h1>",
        unsafe_allow_html=True
    )
    st.info(
        "Simule estratégias quantitativas de **pair trading** com tecnologia de alinhamento temporal (lead-lag) e compare com a abordagem tradicional. Descubra, com dados reais e métricas detalhadas, como uma análise científica pode potencializar seus retornos – ideal para investidores institucionais e inovadores!"
    )

    st.subheader("Configuração do Backtest")
    col1, col2 = st.columns(2)
    with col1:
        commission = st.number_input("Comissão ida+volta (ex: 0.001 = 0.1%)", value=0.001, step=0.0001, format="%.4f")
        lookback = st.slider("Janela rolling (dias)", 20, 120, 60)
        z_entry = st.slider("Z-score de entrada", 1.0, 3.0, 2.0)
        z_exit = st.slider("Z-score de saída", 0.0, 2.0, 0.5)
    with col2:
        max_lag = st.slider("Lag máximo a buscar (dias)", 0, 15, 5)
        top_tickers = st.multiselect("Selecione ativos para análise", TICKERS, default=['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'NVDA'])
        if len(top_tickers) < 2:
            st.warning("Selecione pelo menos dois ativos para formar pares.")
            return

    @st.cache_data(show_spinner=False)
    def load_prices(tickers):
        df = yf.download(tickers, start="2018-01-01", progress=False)["Close"]
        if isinstance(df, pd.Series):
            df = df.to_frame()
        return df

    prices = load_prices(top_tickers).dropna()

    def pair_trading(prices, t1, t2, lookback, z_entry, z_exit, commission, lag=0):
        if lag > 0:
            t2_aligned = prices[t2].shift(lag)
            df = pd.DataFrame({t1: prices[t1], t2: t2_aligned}).dropna()
        elif lag < 0:
            t1_aligned = prices[t1].shift(-lag)
            df = pd.DataFrame({t1: t1_aligned, t2: prices[t2]}).dropna()
        else:
            df = pd.DataFrame({t1: prices[t1], t2: prices[t2]}).dropna()
        if len(df) < lookback + 10:
            return None
        log_spread = np.log(df[t1] / df[t2])
        spread_mean = log_spread.rolling(window=lookback).mean()
        spread_std = log_spread.rolling(window=lookback).std()
        zscore = (log_spread - spread_mean) / spread_std
        df["Zscore"] = zscore
        df["Sinal"] = 0
        df.loc[df["Zscore"] < -z_entry, "Sinal"] = 1
        df.loc[df["Zscore"] > z_entry, "Sinal"] = -1
        df.loc[(df["Zscore"].abs() < z_exit), "Sinal"] = 0
        df["Posicao"] = df["Sinal"].replace(to_replace=0, method="ffill").fillna(0)
        df["Retorno_spread"] = np.log(df[t1] / df[t1].shift(1)) - np.log(df[t2] / df[t2].shift(1))
        df["Retorno_estrategia"] = df["Posicao"].shift(1) * df["Retorno_spread"]
        df["Trade"] = df["Posicao"].diff().abs()
        df["Retorno_estrategia"] -= commission * df["Trade"]
        df["Cumulativo"] = df["Retorno_estrategia"].cumsum().apply(np.exp)
        sharpe = np.nan
        if df["Retorno_estrategia"].std() > 0:
            sharpe = np.mean(df["Retorno_estrategia"]) / np.std(df["Retorno_estrategia"]) * np.sqrt(252)
        n_trades = int(df["Trade"].sum())
        growth = df["Cumulativo"].iloc[-1]
        max_dd = ((df["Cumulativo"].cummax() - df["Cumulativo"]) / df["Cumulativo"].cummax()).max()
        return {
            "sharpe": sharpe,
            "growth": growth,
            "drawdown": max_dd,
            "n_trades": n_trades,
            "df": df,
            "lag": lag
        }

    def lag_min_autocorr(prices, t1, t2, max_lag, lookback):
        lags = list(range(-max_lag, max_lag + 1))
        if 0 in lags:
            lags.remove(0)
        best_metric = np.inf
        best_lag = 0
        for lag in lags:
            if lag > 0:
                b_aligned = prices[t2].shift(lag)
                spread = np.log(prices[t1]) - np.log(b_aligned)
            elif lag < 0:
                a_aligned = prices[t1].shift(-lag)
                spread = np.log(a_aligned) - np.log(prices[t2])
            else:
                spread = np.log(prices[t1]) - np.log(prices[t2])
            ret = spread.diff().dropna()
            if len(ret) < lookback + 5:
                continue
            metric = abs(pd.Series(ret).autocorr(lag=1))
            if metric < best_metric:
                best_metric = metric
                best_lag = lag
        return best_lag

    def get_signals(df, z_entry, z_exit):
        sinais = []
        position = 0
        for idx, row in df.iterrows():
            if position == 0:
                if row["Zscore"] < -z_entry:
                    sinais.append(("long_open", idx, row["Cumulativo"]))
                    position = 1
                elif row["Zscore"] > z_entry:
                    sinais.append(("short_open", idx, row["Cumulativo"]))
                    position = -1
            elif position == 1:
                if abs(row["Zscore"]) < z_exit:
                    sinais.append(("long_close", idx, row["Cumulativo"]))
                    position = 0
            elif position == -1:
                if abs(row["Zscore"]) < z_exit:
                    sinais.append(("short_close", idx, row["Cumulativo"]))
                    position = 0
        return sinais

    # ---- Busca o melhor lag para cada par ----
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("### 🔝 Top 5 pares e defasagens por Sharpe Ratio (com comissões)")
    results = []
    for t1, t2 in combinations(top_tickers, 2):
        best_sharpe = -np.inf
        best_lag = 0
        best_result = None
        for lag in range(-max_lag, max_lag+1):
            if lag == 0:
                continue
            res = pair_trading(prices, t1, t2, lookback, z_entry, z_exit, commission, lag)
            if res and res["sharpe"] > best_sharpe:
                best_sharpe = res["sharpe"]
                best_lag = lag
                best_result = res
        res_trad = pair_trading(prices, t1, t2, lookback, z_entry, z_exit, commission, lag=0)
        lag_exp = lag_min_autocorr(prices, t1, t2, max_lag, lookback)
        res_exp = pair_trading(prices, t1, t2, lookback, z_entry, z_exit, commission, lag_exp)
        if best_result and res_trad and res_exp:
            results.append({
                "par": f"{t1}-{t2}",
                "lag": best_lag,
                "sharpe_lag": best_result["sharpe"],
                "crescimento_lag": best_result["growth"],
                "drawdown_lag": best_result["drawdown"],
                "n_trades_lag": best_result["n_trades"],
                "df_lag": best_result["df"],
                "sharpe_trad": res_trad["sharpe"],
                "crescimento_trad": res_trad["growth"],
                "drawdown_trad": res_trad["drawdown"],
                "n_trades_trad": res_trad["n_trades"],
                "df_trad": res_trad["df"],
                "lag_exp": lag_exp,
                "sharpe_exp": res_exp["sharpe"],
                "crescimento_exp": res_exp["growth"],
                "drawdown_exp": res_exp["drawdown"],
                "n_trades_exp": res_exp["n_trades"],
                "df_exp": res_exp["df"]
            })

    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values(by="sharpe_lag", ascending=False)
    results_df["selector"] = results_df["par"] + " (lag=" + results_df["lag"].astype(str) + ")"
    st.dataframe(
        results_df[["par", "lag", "sharpe_lag", "crescimento_lag", "lag_exp", "sharpe_exp", "crescimento_exp", "sharpe_trad", "crescimento_trad"]].head(5),
        use_container_width=True
    )

    selected_row = st.selectbox(
        "Selecione um par para comparar estratégias",
        results_df["selector"]
    )
    row = results_df.loc[results_df["selector"] == selected_row].iloc[0]

    df_lag = row["df_lag"]
    df_trad = row["df_trad"]
    df_exp = row["df_exp"]

    # --- Gráfico comparativo: todo o período ---
    st.subheader(f"Comparação de estratégias para {row['par']} (período completo)")
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(df_lag.index, df_lag["Cumulativo"], label=f"Lead-Lag ótimo (lag={row['lag']})")
    ax.plot(df_exp.index, df_exp["Cumulativo"], label=f"Lead-Lag exploratório (lag={row['lag_exp']})", linestyle='-.')
    ax.plot(df_trad.index, df_trad["Cumulativo"], label="Tradicional (lag=0)", linestyle='--')
    ax.set_ylabel("Crescimento acumulado (R$ inicial = 1)")
    ax.set_title(f"Crescimento acumulado: {row['par']}")
    ax.legend()
    st.pyplot(fig, use_container_width=True)

    st.markdown("#### Métricas detalhadas")
    col1, col2, col3 = st.columns(3)
    col1.metric("Sharpe (Lead-Lag ótimo)", f"{row['sharpe_lag']:.2f}")
    col1.metric("Crescimento (Lead-Lag ótimo)", f"{row['crescimento_lag']:.2f}x")
    col2.metric("Sharpe (Exploratório)", f"{row['sharpe_exp']:.2f}")
    col2.metric("Crescimento (Exploratório)", f"{row['crescimento_exp']:.2f}x")
    col3.metric("Sharpe (Tradicional)", f"{row['sharpe_trad']:.2f}")
    col3.metric("Crescimento (Tradicional)", f"{row['crescimento_trad']:.2f}x")
    st.caption(
        "Compare como a estratégia muda ao utilizar o lag ótimo exploratório (científico), o ótimo da estratégia e o enfoque tradicional. "
        "Resultados 100% baseados em dados históricos reais."
    )

    # --- Teste fora da amostra: último ano ---
    st.subheader("Comparação fora da amostra (último ano)")
    all_dates = df_lag.index
    if len(all_dates) < 252:
        st.warning("Não há histórico suficiente para realizar teste fora da amostra (último ano).")
    else:
        cutoff_date = all_dates[-252]
        train_prices = prices.loc[:cutoff_date]
        test_prices = prices.loc[cutoff_date:]

        lag_trading_train = None
        lag_exp_train = None
        best_sharpe_train = -np.inf
        for lag in range(-max_lag, max_lag+1):
            if lag == 0:
                continue
            res = pair_trading(train_prices, row["par"].split("-")[0], row["par"].split("-")[1], lookback, z_entry, z_exit, commission, lag)
            if res and res["sharpe"] > best_sharpe_train:
                best_sharpe_train = res["sharpe"]
                lag_trading_train = lag
        lag_exp_train = lag_min_autocorr(train_prices, row["par"].split("-")[0], row["par"].split("-")[1], max_lag, lookback)

        # Simula no conjunto de teste
        res_lag_test = pair_trading(test_prices, row["par"].split("-")[0], row["par"].split("-")[1], lookback, z_entry, z_exit, commission, lag_trading_train)
        res_exp_test = pair_trading(test_prices, row["par"].split("-")[0], row["par"].split("-")[1], lookback, z_entry, z_exit, commission, lag_exp_train)
        res_trad_test = pair_trading(test_prices, row["par"].split("-")[0], row["par"].split("-")[1], lookback, z_entry, z_exit, commission, lag=0)

        fig2 = go.Figure()
        def plot_signals(df, sinais, nome, cor, cor_sinal):
            fig2.add_trace(go.Scatter(x=df.index, y=df["Cumulativo"], mode='lines', name=nome, line=dict(color=cor)))
            for typ, idx, val in sinais:
                if typ == "long_open":
                    fig2.add_trace(go.Scatter(x=[idx], y=[val], mode='markers', marker=dict(symbol="triangle-up", color=cor_sinal, size=12), name="Abrir Long"))
                elif typ == "short_open":
                    fig2.add_trace(go.Scatter(x=[idx], y=[val], mode='markers', marker=dict(symbol="triangle-down", color=cor_sinal, size=12), name="Abrir Short"))
                elif typ == "long_close":
                    fig2.add_trace(go.Scatter(x=[idx], y=[val], mode='markers', marker=dict(symbol="circle", color=cor_sinal, size=10), name="Fechar Long"))
                elif typ == "short_close":
                    fig2.add_trace(go.Scatter(x=[idx], y=[val], mode='markers', marker=dict(symbol="circle", color=cor_sinal, size=10), name="Fechar Short"))

        # --- Lead-Lag ótimo da estratégia
        if res_lag_test:
            df = res_lag_test["df"]
            sinais = get_signals(df, z_entry, z_exit)
            plot_signals(df, sinais, f"Lead-Lag ótimo (lag={lag_trading_train})", "blue", "green")

        # --- Lead-Lag exploratório
        if res_exp_test:
            df = res_exp_test["df"]
            sinais = get_signals(df, z_entry, z_exit)
            plot_signals(df, sinais, f"Lead-Lag exploratório (lag={lag_exp_train})", "orange", "magenta")

        # --- Tradicional
        if res_trad_test:
            df = res_trad_test["df"]
            sinais = get_signals(df, z_entry, z_exit)
            plot_signals(df, sinais, "Tradicional (lag=0)", "black", "blue")

        fig2.update_layout(
            title=f"Teste fora da amostra - Último ano: {row['par']}",
            yaxis_title="Crescimento acumulado (R$ inicial = 1)",
            legend=dict(itemsizing='constant'),
            height=400
        )
        st.plotly_chart(fig2, use_container_width=True)

        st.markdown("#### Resultados fora da amostra (último ano):")
        col1, col2, col3 = st.columns(3)
        if res_lag_test:
            col1.metric("Sharpe Lead-Lag ótimo", f"{res_lag_test['sharpe']:.2f}")
            col1.metric(f"Crescimento Lead-Lag ótimo (lag={lag_trading_train})", f"{res_lag_test['growth']:.2f}x")
        if res_exp_test:
            col2.metric("Sharpe Exploratório", f"{res_exp_test['sharpe']:.2f}")
            col2.metric(f"Crescimento Exploratório (lag={lag_exp_train})", f"{res_exp_test['growth']:.2f}x")
        if res_trad_test:
            col3.metric("Sharpe Tradicional", f"{res_trad_test['sharpe']:.2f}")
            col3.metric("Crescimento Tradicional (lag=0)", f"{res_trad_test['growth']:.2f}x")
