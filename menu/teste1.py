# menu/teste1.py
import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objs as go
from alpaca_trade_api.rest import REST
import time
import os
from dotenv import load_dotenv

# --- INICIO DEL BLOQUE DE CARGA DE CLAVES ---
# Esta lógica permite usar .env.local para desarrollo y st.secrets para deploy
load_dotenv(".env.local")

# Verifica si la app está en Streamlit Cloud (donde st.secrets está disponible)
if hasattr(st, 'secrets'):
    try:
        # Intenta leer las claves desde los secretos de Streamlit
        API_KEY = os.getenv("ALPACA_API_KEY") or st.secrets["ALPACA_API_KEY"]
        API_SECRET = os.getenv("ALPACA_API_SECRET") or st.secrets["ALPACA_API_SECRET"]
    except KeyError:
        st.error("Error: Las claves 'ALPACA_API_KEY' o 'ALPACA_API_SECRET' no están configuradas en los secretos de Streamlit.")
        API_KEY, API_SECRET = None, None
else:
    # Si no, usa las variables de entorno (cargadas desde .env.local)
    API_KEY = os.getenv("ALPACA_API_KEY")
    API_SECRET = os.getenv("ALPACA_API_SECRET")

BASE_URL = "https://paper-api.alpaca.markets"
# --- FIN DEL BLOQUE DE CARGA DE CLAVES ---


def render():
    st.title("💸 Teste1 - Estratégia de trading conservadora (Simulação Alpaca Paper)")

    st.markdown("""
    <div style='font-size:18px;'>
    <b>Descrição:</b> <br>
    Simule operações de compra e venda automatizada de ações usando a API da Alpaca em <b>modo paper</b>.<br>
    Esta estratégia é conservadora, baseada em cruzamento de médias móveis.<br>
    <b>Observação:</b> Para operar em conta real na Alpaca, é necessário ser residente nos EUA.<br>
    <b>Importante:</b> Esta aba executa simulações históricas. Para execução contínua automática, utilize um script Python fora do Streamlit.<br>
    </div>
    """, unsafe_allow_html=True)

    st.header("Configuração da simulação")

    symbols = st.text_input("Ações para simular (separadas por vírgula)", value="AAPL,MSFT,GOOGL,AMZN")
    capital = st.number_input("Capital inicial de simulação (USD)", value=10000, min_value=1000, step=100)
    janela_ma_curta = st.number_input("Período da Média Móvel Curta (dias)", min_value=3, max_value=20, value=5)
    janela_ma_longa = st.number_input("Período da Média Móvel Longa (dias)", min_value=10, max_value=60, value=21)
    periodo_analise = st.selectbox("Período histórico para análise", ["1y", "5y", "6mo", "3mo", "1mo"], index=0)

    executar = st.button("Executar simulação", type="primary")

    if not (API_KEY and API_SECRET):
        st.error("Por favor, configure suas credenciais da Alpaca no arquivo .env.local (localmente) ou nos segredos de Streamlit (em deploy).")
        st.stop()

    if not executar:
        st.stop()

    try:
        api = REST(API_KEY, API_SECRET, BASE_URL)
        st.success("Conexão bem-sucedida com a API Alpaca em modo paper.")
    except Exception as e:
        st.error(f"Erro ao conectar na API Alpaca: {str(e)}")
        st.stop()
    
    st.subheader("Resultados da estratégia")

    resultados = []
    for symbol in [s.strip().upper() for s in symbols.split(",")]:
        data = yf.download(symbol, period=periodo_analise)
        
        if isinstance(data.columns, pd.MultiIndex):
            if symbol in data.columns.get_level_values(1):
                data = data.xs(symbol, axis=1, level=1, drop_level=False)
                data.columns = data.columns.get_level_values(0)
            else:
                st.warning(f"Não encontrei colunas para {symbol}. Pulei este ativo.")
                continue

        if data is None or data.empty or "Close" not in data.columns:
            st.warning(f"Sem dados históricos válidos para {symbol}. Pulei este ativo.")
            continue

        data = data.dropna(subset=["Close"])
        if data.empty:
            st.warning(f"Todos os registros de {symbol} estão vazios após limpeza. Pulei este ativo.")
            continue

        if len(data) < max(janela_ma_curta, janela_ma_longa):
            st.warning(f"Dados insuficientes para {symbol}. Pulando.")
            continue
        data["MA_curta"] = data["Close"].rolling(janela_ma_curta).mean()
        data["MA_longa"] = data["Close"].rolling(janela_ma_longa).mean()

        data["Signal"] = 0
        data.loc[data.index[janela_ma_longa:], "Signal"] = np.where(
            data["MA_curta"][janela_ma_longa:] > data["MA_longa"][janela_ma_longa:], 1, 0
        )
        data["Trade"] = data["Signal"].diff()

        posicao = 0
        saldo = capital / len(symbols.split(","))
        entradas = []
        preco_entrada = None
        
        for i, row in data.iterrows():
            trade_val = row["Trade"]
            try:
                val = float(trade_val)
            except (ValueError, TypeError):
                continue

            if val == 1 and posicao == 0:
                preco_entrada = row["Close"]
                qtde = int(saldo // preco_entrada)
                if qtde >= 1:
                    entradas.append({"data": i, "preco": preco_entrada, "qtde": qtde})
                    posicao = qtde
                    saldo -= qtde * preco_entrada
            elif val == -1 and posicao > 0 and preco_entrada is not None:
                preco_saida = row["Close"]
                entradas[-1].update({"data_saida": i, "preco_saida": preco_saida, "resultado": (preco_saida - preco_entrada) * posicao})
                saldo += preco_saida * posicao
                posicao = 0
                preco_entrada = None

        if posicao > 0 and preco_entrada is not None:
            preco_final = data.iloc[-1]["Close"]
            entradas[-1].update({"data_saida": data.index[-1], "preco_saida": preco_final, "resultado": (preco_final - preco_entrada) * posicao})
            saldo += preco_final * posicao

        resultado_total = saldo - (capital / len(symbols.split(",")))
        resultados.append({
            "Ação": symbol,
            "Retorno (USD)": resultado_total,
            "Retorno (%)": 100 * resultado_total / (capital / len(symbols.split(","))),
            "Qtd trades": len(entradas)
        })

        st.markdown(f"#### {symbol} - Sinais de Compra/Venda")
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=data.index, y=data["Close"], name="Preço de Fechamento", line=dict(color="black")))
        fig.add_trace(go.Scatter(x=data.index, y=data["MA_curta"], name=f"MA Curta ({janela_ma_curta})", line=dict(color="blue")))
        fig.add_trace(go.Scatter(x=data.index, y=data["MA_longa"], name=f"MA Longa ({janela_ma_longa})", line=dict(color="orange")))
        
        # Añadir marcadores de compra
        buy_signals = [ent["data"] for ent in entradas]
        fig.add_trace(go.Scatter(x=buy_signals, y=data.loc[buy_signals]["Close"], name="Compra", mode="markers", marker=dict(color="green", size=10, symbol="triangle-up")))
        
        # Añadir marcadores de venta
        sell_signals = [ent["data_saida"] for ent in entradas if "data_saida" in ent]
        if sell_signals:
            fig.add_trace(go.Scatter(x=sell_signals, y=data.loc[sell_signals]["Close"], name="Venda", mode="markers", marker=dict(color="red", size=10, symbol="triangle-down")))

        fig.update_layout(height=400, legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("### Resultados gerais")
    if resultados:
        resultados_df = pd.DataFrame(resultados)
        st.dataframe(resultados_df.style.format({
            "Retorno (USD)": "{:.2f}",
            "Retorno (%)": "{:.2f}%"
        }))
    
    st.markdown("""
    <div style='background-color:#dbeafe; padding:12px; border-radius:8px; margin-top: 20px;'>
    <b>Observação:</b> O sistema aqui simula apenas. Antes de operar em conta real, valide exaustivamente por meses e ajuste sua estratégia.<br>
    Em investimentos reais, ganhos passados não garantem resultados futuros.<br>
    Estratégias conservadoras tendem a proteger melhor o capital, mas nunca eliminam totalmente o risco.<br>
    Para execução automática/contínua, desenvolva um script Python separado (fora do Streamlit).
    </div>
    """, unsafe_allow_html=True)

    # El código de matplotlib que tenías al final parecía para experimentación,
    # lo he comentado para que no interfiera con la app de Streamlit.
    # Si lo necesitas, puedes descomentarlo o moverlo a otro script.
    
    # import matplotlib.pyplot as plt
    #
    # st.subheader("Análisis de Correlación Cruzada (Ejemplo con Matplotlib)")
    # try:
    #     tickers = ['AMZN', 'MSFT', 'WMT', 'BBY']
    #     data_corr = yf.download(tickers, period='2y')
    #
    #     if 'Adj Close' in data_corr.columns:
    #         precios = data_corr['Adj Close']
    #     elif 'Close' in data_corr.columns:
    #         precios = data_corr['Close']
    #     else:
    #         raise ValueError("No se encontró 'Adj Close' ni 'Close' en los datos descargados.")
    #
    #     precios = precios.dropna()
    #     returns = precios.pct_change().dropna()
    #
    #     fig_corr, ax_corr = plt.subplots()
    #     lags = range(-20, 21)
    #     for other in ['MSFT', 'WMT', 'BBY']:
    #         if 'AMZN' in returns.columns and other in returns.columns:
    #             corr_lags = [returns['AMZN'].corr(returns[other].shift(lag)) for lag in lags]
    #             ax_corr.plot(lags, corr_lags, label=other)
    #
    #     ax_corr.axvline(0, color='gray', linestyle='--')
    #     ax_corr.legend()
    #     ax_corr.set_title('Correlación cruzada de retornos con AMZN (lags)')
    #     ax_corr.set_xlabel('Desfase (días)')
    #     ax_corr.set_ylabel('Correlación')
    #     st.pyplot(fig_corr)
    #
    # except Exception as e:
    #     st.warning(f"No se pudo generar el gráfico de correlación: {e}")

# Esta llamada permite ejecutar el script directamente
if __name__ == "__main__":
    # He añadido una verificación para que la función render() solo se llame una vez.
    if 'render_called' not in st.session_state:
        st.session_state.render_called = True
        render()
    # Para evitar que se vuelva a llamar en cada re-run, reinicia el estado si es necesario
    # o estructura la app de manera que `render` no se llame múltiples veces.
    # Por simplicidad, esta estructura funciona para una página simple.
    # En una app multipágina, la gestión del estado es diferente.