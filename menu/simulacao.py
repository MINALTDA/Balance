import streamlit as st
from core.data import get_weekly_log_returns
from core.simulator import backtest_impulso_resposta

# TICKERS = ['TLT', 'USO', 'QQQ', 'XLF', 'XLY']

TICKERS = [
    'TLT', 'USO', 'QQQ', 'XLF', 'XLY', 'SPY', 'DIA', 'IWM', 'EEM',
    'GLD', 'SLV', 'EWZ', 'VNQ', 'XLV', 'XLI', 'XLB', 'XLE', 'XLK', 'XLP'
]



def show():
    st.markdown("<h1 style='color:#0070F3; font-weight:800;'>Simulação de Estratégia Impulso-Resposta</h1>", unsafe_allow_html=True)
    st.info(
        "Simule o resultado de uma estratégia baseada em choques macroeconômicos: gere sinais automáticos e compare com o buy & hold do ETF."
    )

    # 1. Baixar/preparar dados
    df = get_weekly_log_returns(TICKERS)

    st.subheader("Configuração da Estratégia")
    col1, col2 = st.columns(2)
    with col1:
        ativo_impulso = st.selectbox("Ativo de Impulso (gera o choque)", TICKERS, index=0)
        threshold = st.slider("Threshold (desvios padrão)", 0.5, 3.0, 1.5, 0.1)
    with col2:
        ativo_resposta = st.selectbox("Ativo de Resposta (operado)", TICKERS, index=1)
        direcao = st.selectbox("Direção da Operação", ["short", "long"], index=0)
    janela = st.slider("Janela para média/desvio (semanas)", 12, 104, 52, 4)

    if ativo_impulso == ativo_resposta:
        st.warning("Escolha ativos diferentes para impulso e resposta.")
        return

    resultado = backtest_impulso_resposta(
        df, ativo_impulso, ativo_resposta, direcao, threshold, janela
    )

    st.subheader("Resultados da Simulação")
    st.write(f"Total de sinais gerados: {resultado['Sinal'].sum()}")
    st.line_chart(
        resultado.set_index('Data')[['Retorno_acumulado_estrategia', 'Retorno_acumulado_buy_hold']],
        use_container_width=True
    )
    st.caption("Curva acumulada: azul = estratégia, laranja = buy & hold.")
    col1, col2, col3 = st.columns(3)
    col1.metric("Retorno total estratégia", f"{100 * resultado['Retorno_acumulado_estrategia'].iloc[-1]:.2f}%")
    col2.metric("Retorno buy & hold", f"{100 * resultado['Retorno_acumulado_buy_hold'].iloc[-1]:.2f}%")
    col3.metric("Taxa de acerto (se positivo)", 
        f"{100 * (resultado['Retorno_estrategia']>0).sum() / max(1,resultado['Sinal'].sum()):.1f}%" if resultado['Sinal'].sum()>0 else "-")
    
    with st.expander("Ver detalhes das operações"):
        st.dataframe(resultado, use_container_width=True)
