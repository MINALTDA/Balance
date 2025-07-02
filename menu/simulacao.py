import streamlit as st
import plotly.graph_objs as go
import numpy as np
from menu.helpers_simulacao import (
    load_prices, pair_trading, lag_min_autocorr, get_signals, calcular_top5
)

TICKERS = [
    'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'META', 'NVDA',
    'TLT', 'USO', 'QQQ', 'XLF', 'XLY', 'SPY', 'DIA', 'IWM', 'EEM',
    'GLD', 'SLV', 'EWZ', 'VNQ', 'XLV', 'XLI', 'XLB', 'XLE', 'XLK', 'XLP'
]

COMMISSION = 0.001  # 0.1%

SIGNAL_STYLE = {
    "long_open":    {"symbol": "triangle-up",   "color": "#15803d", "label": "Abrir Long"},
    "long_close":   {"symbol": "circle",        "color": "#65e38c", "label": "Fechar Long"},
    "short_open":   {"symbol": "triangle-down", "color": "#c81e1e", "label": "Abrir Short"},
    "short_close":  {"symbol": "circle",        "color": "#f87171", "label": "Fechar Short"},
}

def add_signals(fig, df, sinais, strategy_name, legendgroup, showlegend=False):
    legends_drawn = set()
    for typ, idx, val in sinais:
        stl = SIGNAL_STYLE[typ]
        legend_label = stl["label"]
        show = showlegend and (legend_label not in legends_drawn)
        fig.add_trace(go.Scatter(
            x=[idx], y=[val], mode='markers',
            marker=dict(symbol=stl["symbol"], color=stl["color"], size=13 if "open" in typ else 10),
            name=legend_label if show else strategy_name,
            legendgroup=legendgroup,
            showlegend=show
        ))
        if show:
            legends_drawn.add(legend_label)

def show():
    st.markdown(
        "<h1 style='color:#0070F3; font-weight:800;'>Pair Trading com otimização de Lead-Lag</h1>",
        unsafe_allow_html=True
    )
    st.info(
        "Simule estratégias quantitativas de **pair trading** com tecnologia de alinhamento temporal (lead-lag) e compare com a abordagem tradicional. Descubra, com dados reais e métricas detalhadas, como uma análise científica pode potencializar seus retornos – ideal para investidores institucionais e inovadores."
    )

    with st.sidebar:
        st.markdown("## Ativos para análise")
        top_tickers = st.multiselect(
            "Selecione ativos para análise",
            TICKERS,
            default=['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'NVDA']
        )
        if len(top_tickers) < 2:
            st.warning("Selecione pelo menos dois ativos para formar pares.")
            return

    with st.sidebar.expander("⚙️ Editar configuração sugerida", expanded=False):
        lookback = st.slider("Janela rolling (dias)", 20, 120, 60)
        z_entry = st.slider("Z-score de entrada", 1.0, 3.0, 2.0)
        z_exit = st.slider("Z-score de saída", 0.0, 2.0, 0.5)
        max_lag = st.slider("Lag máximo a buscar (dias)", 0, 15, 5)

    prices = load_prices(top_tickers).dropna()
    results_df = calcular_top5(prices, top_tickers, lookback, z_entry, z_exit, COMMISSION, max_lag)
    if results_df.empty:
        st.warning("Nenhum par válido encontrado com os parâmetros escolhidos.")
        return

    selected_row = st.selectbox(
        "Selecione um par para comparar estratégias",
        results_df["selector"]
    )
    row = results_df.loc[results_df["selector"] == selected_row].iloc[0]
    df_lag = row["df_lag"]
    df_trad = row["df_trad"]
    df_exp = row["df_exp"]

    # --- Nomes técnicos das estratégias, consistentes com as legendas ---
    nome_sharpe = f"Maximizar sharpe (lag={row['lag']})"
    nome_autocorr = f"Minimizar autocorrelação (lag={row['lag_exp']})"
    nome_trad = "Tradicional (lag=0)"

    # --- Corta o último ano para treino
    all_dates = df_lag.index
    if len(all_dates) < 252:
        st.warning("Não há histórico suficiente para separar treino e teste.")
        return
    cutoff_date = all_dates[-252]
    df_lag_treino = df_lag.loc[:cutoff_date]
    df_trad_treino = df_trad.loc[:cutoff_date]
    df_exp_treino = df_exp.loc[:cutoff_date]

    # --- Gráfico 1: só TREINO ---
    st.subheader(f'Comparação de estratégias para {row["par"]} — "Treino" (sem o último ano)')
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df_lag_treino.index, y=df_lag_treino["Cumulativo"], mode='lines', name=nome_sharpe, legendgroup="LeadLag", line=dict(color="blue", width=2)))
    fig.add_trace(go.Scatter(x=df_exp_treino.index, y=df_exp_treino["Cumulativo"], mode='lines', name=nome_autocorr, legendgroup="Exploratorio", line=dict(color="orange", width=2, dash="dash")))
    fig.add_trace(go.Scatter(x=df_trad_treino.index, y=df_trad_treino["Cumulativo"], mode='lines', name=nome_trad, legendgroup="Tradicional", line=dict(color="black", width=2, dash="dot")))
    fig.update_layout(
        yaxis_title="Crescimento acumulado (R$ inicial = 1)",
        title=f'Crescimento acumulado — período de "Treino": {row["par"]}',
        legend=dict(itemsizing='constant'),
        height=400
    )
    st.plotly_chart(fig, use_container_width=True)

    # --- Top 5 Pares: seção colapsável ---
    with st.expander("🔝 Exibir Top 5 pares e defasagens por Sharpe Ratio (com comissões)"):
        st.dataframe(
            results_df[["par", "lag", "sharpe_lag", "crescimento_lag", "lag_exp", "sharpe_exp", "crescimento_exp", "sharpe_trad", "crescimento_trad"]].head(5),
            use_container_width=True
        )

    st.markdown('#### Métricas detalhadas do período de "Treino"')
    col1, col2, col3 = st.columns(3)
    col1.metric(f"Sharpe - {nome_sharpe}", f"{df_lag_treino['Cumulativo'].pct_change().mean() / df_lag_treino['Cumulativo'].pct_change().std() * np.sqrt(252):.2f}" if df_lag_treino['Cumulativo'].pct_change().std() > 0 else "-")
    col1.metric(f"Crescimento - {nome_sharpe}", f"{df_lag_treino['Cumulativo'].iloc[-1]:.2f}x")
    col2.metric(f"Sharpe - {nome_autocorr}", f"{df_exp_treino['Cumulativo'].pct_change().mean() / df_exp_treino['Cumulativo'].pct_change().std() * np.sqrt(252):.2f}" if df_exp_treino['Cumulativo'].pct_change().std() > 0 else "-")
    col2.metric(f"Crescimento - {nome_autocorr}", f"{df_exp_treino['Cumulativo'].iloc[-1]:.2f}x")
    col3.metric(f"Sharpe - {nome_trad}", f"{df_trad_treino['Cumulativo'].pct_change().mean() / df_trad_treino['Cumulativo'].pct_change().std() * np.sqrt(252):.2f}" if df_trad_treino['Cumulativo'].pct_change().std() > 0 else "-")
    col3.metric(f"Crescimento - {nome_trad}", f"{df_trad_treino['Cumulativo'].iloc[-1]:.2f}x")
    st.caption(
        "Compare o desempenho das estratégias apenas no período de treino (exceto o último ano, reservado para teste fora da amostra)."
    )

    # --- Gráfico 2: Teste fora da amostra com marcas para as 3 curvas ---
    st.subheader('Simulação do último ano — "Teste"')
    train_prices = prices.loc[:cutoff_date]
    test_prices = prices.loc[cutoff_date:]

    lag_trading_train = None
    lag_exp_train = None
    best_sharpe_train = -np.inf
    for lag in range(-max_lag, max_lag+1):
        if lag == 0:
            continue
        res = pair_trading(train_prices, row["par"].split("-")[0], row["par"].split("-")[1], lookback, z_entry, z_exit, COMMISSION, lag)
        if res and res["sharpe"] > best_sharpe_train:
            best_sharpe_train = res["sharpe"]
            lag_trading_train = lag
    lag_exp_train = lag_min_autocorr(train_prices, row["par"].split("-")[0], row["par"].split("-")[1], max_lag, lookback)

    res_lag_test = pair_trading(test_prices, row["par"].split("-")[0], row["par"].split("-")[1], lookback, z_entry, z_exit, COMMISSION, lag_trading_train)
    res_exp_test = pair_trading(test_prices, row["par"].split("-")[0], row["par"].split("-")[1], lookback, z_entry, z_exit, COMMISSION, lag_exp_train)
    res_trad_test = pair_trading(test_prices, row["par"].split("-")[0], row["par"].split("-")[1], lookback, z_entry, z_exit, COMMISSION, lag=0)

    fig2 = go.Figure()
    # Lead-Lag ótimo
    if res_lag_test:
        df = res_lag_test["df"]
        strategy_name = f"Maximizar sharpe (lag={lag_trading_train})"
        fig2.add_trace(go.Scatter(x=df.index, y=df["Cumulativo"], mode='lines', name=strategy_name, legendgroup="LeadLag", line=dict(color="blue", width=2)))
        sinais = get_signals(df, z_entry, z_exit)
        add_signals(fig2, df, sinais, strategy_name, "LeadLag", showlegend=False)
    # Lead-Lag exploratório
    if res_exp_test:
        df = res_exp_test["df"]
        strategy_name = f"Minimizar autocorrelação (lag={lag_exp_train})"
        fig2.add_trace(go.Scatter(x=df.index, y=df["Cumulativo"], mode='lines', name=strategy_name, legendgroup="Exploratorio", line=dict(color="orange", width=2, dash="dash")))
        sinais = get_signals(df, z_entry, z_exit)
        add_signals(fig2, df, sinais, strategy_name, "Exploratorio", showlegend=False)
    # Tradicional
    if res_trad_test:
        df = res_trad_test["df"]
        strategy_name = "Tradicional (lag=0)"
        fig2.add_trace(go.Scatter(x=df.index, y=df["Cumulativo"], mode='lines', name=strategy_name, legendgroup="Tradicional", line=dict(color="black", width=2, dash="dot")))
        sinais = get_signals(df, z_entry, z_exit)
        add_signals(fig2, df, sinais, strategy_name, "Tradicional", showlegend=True)

    fig2.update_layout(
        title=f'Teste com dados do último ano: {row["par"]}',
        yaxis_title="Crescimento acumulado (R$ inicial = 1)",
        legend=dict(itemsizing='constant'),
        height=400
    )
    st.plotly_chart(fig2, use_container_width=True)

    st.markdown('#### Resultados da simulação do último ano')
    col1, col2, col3 = st.columns(3)
    if res_lag_test:
        col1.metric(f"Sharpe - {nome_sharpe}", f"{res_lag_test['sharpe']:.2f}")
        col1.metric(f"Crescimento - {nome_sharpe}", f"{res_lag_test['growth']:.2f}x")
    if res_exp_test:
        col2.metric(f"Sharpe - {nome_autocorr}", f"{res_exp_test['sharpe']:.2f}")
        col2.metric(f"Crescimento - {nome_autocorr}", f"{res_exp_test['growth']:.2f}x")
    if res_trad_test:
        col3.metric(f"Sharpe - {nome_trad}", f"{res_trad_test['sharpe']:.2f}")
        col3.metric(f"Crescimento - {nome_trad}", f"{res_trad_test['growth']:.2f}x")
