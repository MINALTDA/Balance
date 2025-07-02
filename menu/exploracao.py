# import streamlit as st
# import yfinance as yf
# import pandas as pd
# import numpy as np
# import plotly.graph_objs as go

# def show():
#     st.title("🔬 Exploração do alinhamento temporal (Lead-Lag)")

#     st.info(
#         "Esta seção é dedicada **exclusivamente para exploração acadêmica** dos critérios de alinhamento temporal (lead-lag) entre pares de ativos. "
#         "**Não estará disponível para uso em produção.** O objetivo é apoiar estudos e comparações de diferentes métodos para busca do lag ótimo usando dados reais."
#     )

#     # Sidebar: apenas ativos visíveis por padrão
#     tickers = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "META", "NVDA"]
#     st.sidebar.header("Selecione os ativos")
#     t1 = st.sidebar.selectbox("Ativo líder (A)", tickers, 0)
#     t2 = st.sidebar.selectbox("Ativo seguidor (B)", tickers, 1)

#     # Parâmetros avançados em painel colapsável
#     with st.sidebar.expander("⚙️ Parâmetros avançados", expanded=False):
#         max_lag = st.slider("Lag máximo a explorar (dias)", 1, 15, 7)
#         lookback = st.slider("Janela rolling (dias)", 10, 120, 60)
#         z_entry = st.slider("Z-score de entrada", 1.0, 3.0, 2.0)
#         z_exit = st.slider("Z-score de saída", 0.0, 2.0, 0.5)
#         commission = st.number_input("Comissão ida+volta (ex: 0.001 = 0.1%)", value=0.001, step=0.0001, format="%.4f")

#     st.markdown(
#         """
#         <span style='font-size:1.1em'>
#         Este painel compara **dois critérios científicos de alinhamento temporal (lead-lag)**:
#         <ul>
#           <li><b>Lag por reversão à média:</b> Seleciona o lag com menor autocorrelação dos retornos do spread.</li>
#           <li><b>Lag por Sharpe da estratégia:</b> Seleciona o lag que maximiza o Sharpe anualizado da estratégia de trading, com os parâmetros do simulador.</li>
#         </ul>
#         O gráfico central mostra as duas métricas para cada lag, com destaque visual para os lags otimizados.
#         </span>
#         """,
#         unsafe_allow_html=True
#     )

#     @st.cache_data(show_spinner=False)
#     def carregar_precos(tickers):
#         df = yf.download(tickers, start="2018-01-01", progress=False)["Close"]
#         if isinstance(df, pd.Series):
#             df = df.to_frame()
#         return df

#     precos = carregar_precos([t1, t2]).dropna()

#     def calcular_spread(precos, lag):
#         if lag > 0:
#             b_alinhado = precos[t2].shift(lag)
#             spread = np.log(precos[t1]) - np.log(b_alinhado)
#         elif lag < 0:
#             a_alinhado = precos[t1].shift(-lag)
#             spread = np.log(a_alinhado) - np.log(precos[t2])
#         else:
#             spread = np.log(precos[t1]) - np.log(precos[t2])
#         return spread.dropna()

#     def autocorr_spread(precos, lag, lookback):
#         spread = calcular_spread(precos, lag)
#         ret = spread.diff().dropna()
#         if len(ret) < lookback + 5:
#             return np.nan
#         return abs(pd.Series(ret).autocorr(lag=1))

#     def sharpe_estrategia(precos, lag, lookback, z_entry, z_exit, commission):
#         if lag > 0:
#             t2_alinhado = precos[t2].shift(lag)
#             df = pd.DataFrame({t1: precos[t1], t2: t2_alinhado}).dropna()
#         elif lag < 0:
#             t1_alinhado = precos[t1].shift(-lag)
#             df = pd.DataFrame({t1: t1_alinhado, t2: precos[t2]}).dropna()
#         else:
#             df = pd.DataFrame({t1: precos[t1], t2: precos[t2]}).dropna()
#         if len(df) < lookback + 10:
#             return np.nan
#         log_spread = np.log(df[t1] / df[t2])
#         spread_mean = log_spread.rolling(window=lookback).mean()
#         spread_std = log_spread.rolling(window=lookback).std()
#         zscore = (log_spread - spread_mean) / spread_std
#         df["Zscore"] = zscore
#         df["Sinal"] = 0
#         df.loc[df["Zscore"] < -z_entry, "Sinal"] = 1
#         df.loc[df["Zscore"] > z_entry, "Sinal"] = -1
#         df.loc[(df["Zscore"].abs() < z_exit), "Sinal"] = 0
#         df["Posicao"] = df["Sinal"].replace(to_replace=0, method="ffill").fillna(0)
#         df["Retorno_spread"] = np.log(df[t1] / df[t1].shift(1)) - np.log(df[t2] / df[t2].shift(1))
#         df["Retorno_estrategia"] = df["Posicao"].shift(1) * df["Retorno_spread"]
#         df["Trade"] = df["Posicao"].diff().abs()
#         df["Retorno_estrategia"] -= commission * df["Trade"]
#         if df["Retorno_estrategia"].std() > 0:
#             return np.mean(df["Retorno_estrategia"]) / np.std(df["Retorno_estrategia"]) * np.sqrt(252)
#         else:
#             return np.nan

#     # --- Calcula métricas para cada lag ---
#     lags = list(range(-max_lag, max_lag + 1))
#     if 0 in lags:
#         lags.remove(0)
#     autocorrs = []
#     sharpes = []
#     for lag in lags:
#         autocorrs.append(autocorr_spread(precos, lag, lookback))
#         sharpes.append(sharpe_estrategia(precos, lag, lookback, z_entry, z_exit, commission))

#     # Seleciona os lags otimizados
#     lag_auto = lags[np.nanargmin(autocorrs)]  # menor autocorrelação
#     lag_sharpe = lags[np.nanargmax(sharpes)]  # maior Sharpe

#     # --- Gráfico duplo eixo y: autocorrelação e Sharpe vs. lag ---
#     st.subheader("Evolução das métricas em função do lag (alinhamento temporal)")
#     fig = go.Figure()

#     fig.add_trace(go.Scatter(x=lags, y=autocorrs, name="Autocorrelação absoluta", yaxis="y1", line=dict(color="green")))
#     fig.add_trace(go.Scatter(x=lags, y=sharpes, name="Sharpe Ratio da estratégia", yaxis="y2", line=dict(color="purple")))

#     fig.add_vline(x=lag_auto, line=dict(color="green", dash="dash"), annotation_text=f"Menor autocorr.: {lag_auto}", annotation_position="top left")
#     fig.add_vline(x=lag_sharpe, line=dict(color="purple", dash="dash"), annotation_text=f"Máx. Sharpe: {lag_sharpe}", annotation_position="top right")

#     fig.update_layout(
#         xaxis=dict(title="Lag aplicado (dias)"),
#         yaxis=dict(title="Autocorrelação abs.", tickfont=dict(color="green")),
#         yaxis2=dict(title="Sharpe Ratio estratégia", tickfont=dict(color="purple"), anchor="x", overlaying="y", side="right"),
#         legend=dict(x=0.01, y=0.99),
#         height=370
#     )
#     st.plotly_chart(fig, use_container_width=True)

#     st.markdown(
#         f"""
#         - <span style="color:green"><b>Lag com menor autocorrelação:</b> {lag_auto}</span>
#         - <span style="color:purple"><b>Lag com maior Sharpe Ratio:</b> {lag_sharpe}</span>
#         """, unsafe_allow_html=True
#     )

#     # --- Visualização dos spreads e z-scores para os lags otimizados ---
#     def plot_spread_zscore(lag, label):
#         spread = calcular_spread(precos, lag)
#         spread_mean = spread.rolling(window=lookback).mean()
#         spread_std = spread.rolling(window=lookback).std()
#         zscore = (spread - spread_mean) / spread_std

#         st.subheader(f"Spread e Z-score para {label} (lag={lag})")
#         fig2 = go.Figure()
#         fig2.add_trace(go.Scatter(x=spread.index, y=spread, name="Spread"))
#         fig2.add_trace(go.Scatter(x=spread_mean.index, y=spread_mean, name="Média móvel", line=dict(dash='dash')))
#         fig2.update_layout(height=240, yaxis_title="Spread")
#         st.plotly_chart(fig2, use_container_width=True)

#         fig3 = go.Figure()
#         fig3.add_trace(go.Scatter(x=zscore.index, y=zscore, name="Z-score"))
#         fig3.add_trace(go.Scatter(x=zscore.index, y=np.zeros_like(zscore), name="Média", line=dict(dash='dash')))
#         fig3.update_layout(height=200, yaxis_title="Z-score")
#         st.plotly_chart(fig3, use_container_width=True)

#     st.markdown("## Visualização dos spreads e z-scores para diferentes lags otimizados")
#     st.markdown("### (A) Lag = 0 (sem alinhamento)")
#     plot_spread_zscore(0, "Sem alinhamento")

#     st.markdown(f"### (B) Lag com menor autocorrelação dos retornos do spread (lag={lag_auto})")
#     plot_spread_zscore(lag_auto, "Lag otimizado por reversão à média")

#     st.markdown(f"### (C) Lag com maior Sharpe Ratio da estratégia (lag={lag_sharpe})")
#     plot_spread_zscore(lag_sharpe, "Lag otimizado pelo Sharpe da estratégia")

#     st.info(
#         "O gráfico duplo acima destaca visualmente os pontos de inflexão para cada critério, mostrando o lag selecionado por reversão à média e o lag selecionado pelo Sharpe da estratégia. "
#         "Os gráficos abaixo permitem comparar o comportamento do spread e do z-score para cada lag de interesse, apoiando decisões científicas justificadas para aplicação em trading."
#     )








import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objs as go

def show():
    st.title("🔬 Exploração do alinhamento temporal (Lead-Lag)")
    
    # Sidebar: apenas ativos visíveis por padrão
    tickers = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "META", "NVDA"]
    st.sidebar.header("Selecione os ativos")
    t1 = st.sidebar.selectbox("Ativo líder (A)", tickers, 0)
    t2 = st.sidebar.selectbox("Ativo seguidor (B)", tickers, 1)

    # Parâmetros avançados em painel colapsável
    with st.sidebar.expander("⚙️ Parâmetros avançados", expanded=False):
        max_lag = st.slider("Lag máximo a explorar (dias)", 1, 15, 7)
        lookback = st.slider("Janela rolling (dias)", 10, 120, 60)
        z_entry = st.slider("Z-score de entrada", 1.0, 3.0, 2.0)
        z_exit = st.slider("Z-score de saída", 0.0, 2.0, 0.5)
        commission = st.number_input("Comissão ida+volta (ex: 0.001 = 0.1%)", value=0.001, step=0.0001, format="%.4f")

    st.markdown(
        """
        <span style='font-size:1.15em'>
        Este painel compara dois critérios de alinhamento temporal (lead-lag):
        <ul>
          <li><b>Lag por reversão à média:</b> Seleciona o lag com menor autocorrelação dos retornos do spread.</li>
          <li><b>Lag por Sharpe da estratégia:</b> Seleciona o lag que maximiza o Sharpe anualizado da estratégia de trading, com os parâmetros do simulador.</li>
        </ul>
        O gráfico central mostra as duas métricas para cada lag, com destaque visual para os lags otimizados.
        </span>
        """,
        unsafe_allow_html=True
    )

    @st.cache_data(show_spinner=False)
    def carregar_precos(tickers):
        df = yf.download(tickers, start="2018-01-01", progress=False)["Close"]
        if isinstance(df, pd.Series):
            df = df.to_frame()
        return df

    precos = carregar_precos([t1, t2]).dropna()

    def calcular_spread(precos, lag):
        if lag > 0:
            b_alinhado = precos[t2].shift(lag)
            spread = np.log(precos[t1]) - np.log(b_alinhado)
        elif lag < 0:
            a_alinhado = precos[t1].shift(-lag)
            spread = np.log(a_alinhado) - np.log(precos[t2])
        else:
            spread = np.log(precos[t1]) - np.log(precos[t2])
        return spread.dropna()

    def autocorr_spread(precos, lag, lookback):
        spread = calcular_spread(precos, lag)
        ret = spread.diff().dropna()
        if len(ret) < lookback + 5:
            return np.nan
        return abs(pd.Series(ret).autocorr(lag=1))

    def sharpe_estrategia(precos, lag, lookback, z_entry, z_exit, commission):
        if lag > 0:
            t2_alinhado = precos[t2].shift(lag)
            df = pd.DataFrame({t1: precos[t1], t2: t2_alinhado}).dropna()
        elif lag < 0:
            t1_alinhado = precos[t1].shift(-lag)
            df = pd.DataFrame({t1: t1_alinhado, t2: precos[t2]}).dropna()
        else:
            df = pd.DataFrame({t1: precos[t1], t2: precos[t2]}).dropna()
        if len(df) < lookback + 10:
            return np.nan
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
        if df["Retorno_estrategia"].std() > 0:
            return np.mean(df["Retorno_estrategia"]) / np.std(df["Retorno_estrategia"]) * np.sqrt(252)
        else:
            return np.nan

    # --- Calcula métricas para cada lag ---
    lags = list(range(-max_lag, max_lag + 1))
    if 0 in lags:
        lags.remove(0)
    autocorrs = []
    sharpes = []
    for lag in lags:
        autocorrs.append(autocorr_spread(precos, lag, lookback))
        sharpes.append(sharpe_estrategia(precos, lag, lookback, z_entry, z_exit, commission))

    # Seleciona os lags otimizados
    lag_auto = lags[np.nanargmin(autocorrs)]  # menor autocorrelação
    lag_sharpe = lags[np.nanargmax(sharpes)]  # maior Sharpe

    # --- Gráfico duplo eixo y: autocorrelação e Sharpe vs. lag ---
    st.subheader("Evolução das métricas em função do lag (alinhamento temporal)")
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=lags, y=autocorrs, name="Autocorrelação absoluta", yaxis="y1",
        line=dict(color="green", width=3)
    ))
    fig.add_trace(go.Scatter(
        x=lags, y=sharpes, name="Sharpe Ratio da estratégia", yaxis="y2",
        line=dict(color="purple", width=3)
    ))

    fig.add_vline(x=lag_auto, line=dict(color="green", dash="dash"), annotation_text=f"Menor autocorr.: {lag_auto}", annotation_position="top left")
    fig.add_vline(x=lag_sharpe, line=dict(color="purple", dash="dash"), annotation_text=f"Máx. Sharpe: {lag_sharpe}", annotation_position="top left")

    fig.update_layout(
        xaxis=dict(
            title=dict(text="Lag aplicado (dias)", font=dict(size=18)),
            tickfont=dict(size=15)
        ),
        yaxis=dict(
            title=dict(text="Autocorrelação abs.", font=dict(size=17)),
            tickfont=dict(color="green", size=15)
        ),
        yaxis2=dict(
            title=dict(text="Sharpe Ratio estratégia", font=dict(size=17)),
            tickfont=dict(color="purple", size=15),
            anchor="x", overlaying="y", side="right"
        ),
        legend=dict(x=0.99, y=0.99, xanchor='right', yanchor='top', font=dict(size=15)),
        height=430,
        font=dict(size=16)
    )
    st.plotly_chart(fig, use_container_width=True)

    st.markdown(
        f"""
        - <span style="color:green"><b>Lag com menor autocorrelação:</b> {lag_auto}</span>
        - <span style="color:purple"><b>Lag com maior Sharpe Ratio:</b> {lag_sharpe}</span>
        """, unsafe_allow_html=True
    )

    # --- Selector para qual estratégia visualizar (A, B, C) ---
    st.markdown("## Visualização dos spreads e z-scores para diferentes lags otimizados")
    estrategia_opcoes = {
        f"(A) Sem alinhamento (lag=0)": 0,
        f"(B) Lag com menor autocorrelação (lag={lag_auto})": lag_auto,
        f"(C) Lag com maior Sharpe Ratio (lag={lag_sharpe})": lag_sharpe
    }
    escolha = st.radio(
        "Selecione a estratégia/lag para visualização detalhada:",
        options=list(estrategia_opcoes.keys()),
        index=1  # autocorrelação como padrão
    )
    lag_selecionado = estrategia_opcoes[escolha]

    def plot_spread_zscore(lag, label):
        spread = calcular_spread(precos, lag)
        spread_mean = spread.rolling(window=lookback).mean()
        spread_std = spread.rolling(window=lookback).std()
        zscore = (spread - spread_mean) / spread_std

        st.subheader(f"Spread e Z-score para {label} (lag={lag})")
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(x=spread.index, y=spread, name="Spread", line=dict(width=3)))
        fig2.add_trace(go.Scatter(x=spread_mean.index, y=spread_mean, name="Média móvel", line=dict(dash='dash', width=2)))
        fig2.update_layout(
            height=340,
            yaxis_title="Spread",
            font=dict(size=16),
            legend=dict(font=dict(size=15))
        )
        st.plotly_chart(fig2, use_container_width=True)

        fig3 = go.Figure()
        fig3.add_trace(go.Scatter(x=zscore.index, y=zscore, name="Z-score", line=dict(width=3)))
        fig3.add_trace(go.Scatter(x=zscore.index, y=np.zeros_like(zscore), name="Média", line=dict(dash='dash', width=2)))
        fig3.update_layout(
            height=260,
            yaxis_title="Z-score",
            font=dict(size=16),
            legend=dict(font=dict(size=15))
        )
        st.plotly_chart(fig3, use_container_width=True)

    # Visualização única (de acordo com seleção)
    if lag_selecionado == 0:
        plot_spread_zscore(0, "Sem alinhamento")
    elif lag_selecionado == lag_auto:
        plot_spread_zscore(lag_auto, "Lag otimizado por reversão à média")
    elif lag_selecionado == lag_sharpe:
        plot_spread_zscore(lag_sharpe, "Lag otimizado pelo Sharpe da estratégia")

    st.info(
        "O gráfico duplo acima destaca visualmente os pontos de inflexão para cada critério, mostrando o lag selecionado por reversão à média e o lag selecionado pelo Sharpe da estratégia. "
        "Os gráficos detalhados permitem comparar o comportamento do spread e do z-score para cada lag de interesse, apoiando decisões científicas justificadas para aplicação em trading."
    )

    st.info(
        "Esta seção é dedicada **exclusivamente para exploração acadêmica** dos critérios de alinhamento temporal (lead-lag) entre pares de ativos. "
        "**Não estará disponível para uso em produção.** O objetivo é apoiar estudos e comparações de diferentes métodos para busca do lag ótimo usando dados reais."
    )
