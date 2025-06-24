# import streamlit as st
# import pandas as pd
# from core.data import get_weekly_log_returns, test_stationarity_adf
# from core.modelo import select_var_order, fit_var, granger_matrix, get_irf

# # TICKERS = ['TLT', 'VIX', 'USO', 'QQQ', 'XLF', 'XLY']
# TICKERS = ['TLT', 'USO', 'QQQ', 'XLF', 'XLY']

# def show():
#     st.markdown("<h1 style='color:#0070F3; font-weight:800;'>Modelo Econométrico (VAR)</h1>", unsafe_allow_html=True)

#     st.info(
#         "Esta seção mostra o núcleo científico da estratégia: desde o tratamento dos dados até a interpretação das relações entre os setores."
#     )
    

#     # 1. Baixar/preparar dados
#     st.subheader("1. Dados semanais dos ETFs")
#     df = get_weekly_log_returns(TICKERS)
#     st.dataframe(df.tail(), use_container_width=True)

#     st.write("Shape dos dados:", df.shape)
    

#     # 2. Estacionariedade
#     st.subheader("2. Teste de Estacionariedade (ADF)")
#     stationarity = test_stationarity_adf(df)
#     st.dataframe(stationarity, use_container_width=True)
#     st.caption("Critério: valor-p < 0,05 indica série estacionária.")

#     # 3. Seleção de ordem VAR
#     st.subheader("3. Seleção do número de defasagens (lags)")
#     p_opt, ic_df = select_var_order(df)
#     st.line_chart(ic_df['AIC'])
#     st.success(f"Lag ótimo (AIC): {p_opt}")

#     # 4. Ajuste do modelo
#     st.subheader("4. Ajuste do Modelo VAR")
#     var_model = fit_var(df, p_opt)
#     st.code(var_model.summary())

#     # 5. Causalidade de Granger (matriz)
#     st.subheader("5. Relações de causalidade (Granger)")
#     st.caption("Células verdes: causalidade significativa (valor-p < 0,05).")
#     # granger_df = granger_matrix(df, maxlag=p_opt)
#     maxlag_granger = max(1, p_opt)
#     granger_df = granger_matrix(df, maxlag=maxlag_granger)
#     styled = granger_df.style.applymap(
#         lambda v: 'background-color:#33D69F;color:white;' if v != '-' and float(v) < 0.05 else ''
#     )
#     st.dataframe(styled, use_container_width=True)

#     # 6. Função Impulso-Resposta (IRF)
#     st.subheader("6. Função Impulso-Resposta")
#     impulse = st.selectbox("Selecione a variável de choque (impulso):", TICKERS)
#     response = st.selectbox("Selecione a variável de resposta:", TICKERS)
#     periods = st.slider("Semanas para simular", 2, 12, 6)

#     # irf_df = get_irf(var_model, impulse, response, periods)
#     if var_model.k_ar < 1:
#         st.warning("Não é possível calcular a Função Impulso-Resposta porque o modelo foi ajustado com lag zero. Tente aumentar o período dos dados ou incluir mais ativos.")
#     else:
#         impulse = st.selectbox("Selecione a variável de choque (impulso):", TICKERS)
#         response = st.selectbox("Selecione a variável de resposta:", TICKERS)
#         periods = st.slider("Semanas para simular", 2, 12, 6)
#         irf_df = get_irf(var_model, impulse, response, periods)
#         st.line_chart(irf_df['Resposta'])
#         st.caption("Simulação: choque de 1 desvio padrão no impulso escolhido.")

#     st.line_chart(irf_df['Resposta'])
#     st.caption("Simulação: choque de 1 desvio padrão no impulso escolhido.")

#     st.markdown("---")
#     st.info("Agora, utilize os resultados para alimentar as regras de sinais automáticos!")









import streamlit as st
import pandas as pd
from core.data import get_weekly_log_returns, test_stationarity_adf
from core.modelo import select_var_order, fit_var, granger_matrix, get_irf

# TICKERS = ['TLT', 'USO', 'QQQ', 'XLF', 'XLY']
TICKERS = [
    'TLT', 'USO', 'QQQ', 'XLF', 'XLY', 'SPY', 'DIA', 'IWM', 'EEM',
    'GLD', 'SLV', 'EWZ', 'VNQ', 'XLV', 'XLI', 'XLB', 'XLE', 'XLK', 'XLP'
]

def show():
    st.markdown("<h1 style='color:#0070F3; font-weight:800;'>Modelo Econométrico (VAR)</h1>", unsafe_allow_html=True)
    st.info(
        "Explore como os choques macroeconômicos se propagam entre setores e veja, de forma interativa, as relações de causalidade, impulso-resposta e previsibilidade do sistema."
    )

    # 1. Baixar/preparar dados
    st.subheader("1. Dados semanais dos ETFs")
    df = get_weekly_log_returns(TICKERS)
    st.write("Shape dos dados:", df.shape)
    st.write("Período de datas:", df.index.min(), "a", df.index.max())
    st.write("Valores ausentes por coluna:")
    st.write(df.isna().sum())
    st.dataframe(df.tail(), use_container_width=True)

    # 2. Teste de estacionariedade
    st.subheader("2. Teste de Estacionariedade (ADF)")
    stationarity = test_stationarity_adf(df)
    st.dataframe(stationarity, use_container_width=True)
    st.caption("Critério: valor-p < 0,05 indica série estacionária.")

    # 3. Seleção de ordem VAR (automática e manual)
    st.subheader("3. Seleção do número de defasagens (lags)")

    
    # p_opt, ic_df = select_var_order(df, maxlags=5)
    p_opt, ic_df = select_var_order(df, maxlags=5, min_lag=1)


    st.line_chart(ic_df['AIC'])
    st.success(f"Lag ótimo sugerido (AIC): {p_opt}")

    st.info("Você pode escolher manualmente o número de lags para o modelo VAR abaixo. Recomendado: 1 a 3 para dados semanais.")
    p_manual = st.number_input(
        "Lag a ser usado no VAR", 
        min_value=1, max_value=5, value=max(1, p_opt), step=1
    )

    # 4. Ajuste do modelo
    st.subheader("4. Ajuste do Modelo VAR")
    var_model = fit_var(df, int(p_manual))
    st.code(var_model.summary())

    # 5. Causalidade de Granger (matriz)
    st.subheader("5. Relações de causalidade (Granger)")
    st.caption("Células verdes: causalidade significativa (valor-p < 0,05).")
    maxlag_granger = int(p_manual)
    granger_df = granger_matrix(df, maxlag=maxlag_granger)
    styled = granger_df.style.applymap(
        lambda v: 'background-color:#33D69F;color:white;' if v != '-' and float(v) < 0.05 else ''
    )
    st.dataframe(styled, use_container_width=True)

    # 6. Função Impulso-Resposta (IRF)
    st.subheader("6. Função Impulso-Resposta")
    if var_model.k_ar < 1:
        st.warning("Não é possível calcular a Função Impulso-Resposta porque o modelo foi ajustado com lag zero. Aumente o lag e tente novamente.")
    else:
        impulse = st.selectbox("Selecione a variável de choque (impulso):", df.columns)
        response = st.selectbox("Selecione a variável de resposta:", df.columns)
        periods = st.slider("Semanas para simular", 2, 12, 6)
        irf_df = get_irf(var_model, impulse, response, periods)
        st.line_chart(irf_df['Resposta'])
        st.caption("Simulação: choque de 1 desvio padrão no impulso escolhido.")

    st.markdown("---")
    st.info("Utilize os resultados para alimentar regras automáticas de sinais na próxima etapa do dashboard!")
