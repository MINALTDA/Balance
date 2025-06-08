# import streamlit as st
# import pandas as pd
# import numpy as np
# import yfinance as yf
# import plotly.graph_objs as go
# from datetime import datetime, timedelta
# from alpaca_trade_api.rest import REST, TimeFrame

# def render():
#     st.title("💸 Teste1 - Estratégia de Trading Conservadora (Simulação Alpaca Paper)")

#     st.markdown("""
#     <div style='font-size:18px;'>
#     <b>Descrição:</b> <br>
#     Esta aba permite simular operações de compra e venda automatizada de ações usando a API da Alpaca em <b>modo paper (teste/simulação)</b>.<br>
#     <ul>
#       <li>Você pode usar de qualquer lugar do Brasil para testar estratégias e visualizar resultados sem risco real.</li>
#       <li>O objetivo desta estratégia é ser <b>conservadora</b>, usando médias móveis e filtros de volatilidade, visando operações com maior probabilidade de lucro no médio/longo prazo.</li>
#       <li><b>Observação:</b> Para operar em conta real na Alpaca, é necessário ser residente nos EUA, mas todo desenvolvimento e validação pode ser feito em paper trading!</li>
#     </ul>
#     </div>
#     """, unsafe_allow_html=True)

#     # --- Entradas do usuário ---
#     st.header("Configuração da Simulação")

#     API_KEY = st.text_input("Alpaca API Key", type="password")
#     API_SECRET = st.text_input("Alpaca API Secret", type="password")
#     BASE_URL = "https://paper-api.alpaca.markets"

#     symbols = st.text_input("Ações para simular (separadas por vírgula)", value="AAPL,MSFT,GOOGL,AMZN")
#     capital = st.number_input("Capital inicial de simulação (USD)", value=10000, min_value=1000, step=100)
#     janela_ma_curta = st.number_input("Período da Média Móvel Curta (dias)", min_value=3, max_value=20, value=5)
#     janela_ma_longa = st.number_input("Período da Média Móvel Longa (dias)", min_value=10, max_value=60, value=21)
#     periodo_analise = st.selectbox("Período histórico para análise", ["1y", "6mo", "3mo", "1mo"], index=0)

#     executar = st.button("Executar Simulação", type="primary")

#     if not (API_KEY and API_SECRET):
#         st.info("Por favor, informe sua API Key e Secret da Alpaca (modo paper).")
#         st.stop()

#     if not executar:
#         st.stop()

#     try:
#         api = REST(API_KEY, API_SECRET, BASE_URL)
#         st.success("Conexão bem-sucedida com a API Alpaca em modo paper.")
#     except Exception as e:
#         st.error(f"Erro ao conectar na API Alpaca: {str(e)}")
#         st.stop()

#     # --- Estratégia conservadora baseada em cruzamento de médias móveis ---
#     st.subheader("Resultados da Estratégia")

#     resultados = []
#     for symbol in [s.strip().upper() for s in symbols.split(",")]:
#         # 1. Baixar histórico com yfinance
#         data = yf.download(symbol, period=periodo_analise)
#         if len(data) < max(janela_ma_curta, janela_ma_longa):
#             st.warning(f"Dados insuficientes para {symbol}. Pulando.")
#             continue
#         # 2. Calcular médias móveis
#         data["MA_curta"] = data["Close"].rolling(janela_ma_curta).mean()
#         data["MA_longa"] = data["Close"].rolling(janela_ma_longa).mean()

#         # 3. Gerar sinais: compra quando MA_curta cruza acima da MA_longa, venda quando cruza abaixo
#         data["Signal"] = 0
#         data["Signal"][janela_ma_longa:] = np.where(
#             data["MA_curta"][janela_ma_longa:] > data["MA_longa"][janela_ma_longa:], 1, 0
#         )
#         data["Trade"] = data["Signal"].diff()

#         # 4. Simular trades
#         posicao = 0
#         saldo = capital / len(symbols.split(","))
#         entradas, saidas = [], []
#         for i, row in data.iterrows():
#             # if row["Trade"] == 1 and posicao == 0:
#             if not pd.isna(row["Trade"]) and row["Trade"] == 1 and posicao == 0:
#                 # Comprar tudo disponível (modo conservador: 100% do saldo para cada ação, sem alavancagem)
#                 preco_entrada = row["Close"]
#                 qtde = saldo // preco_entrada
#                 if qtde > 0:
#                     entradas.append({"data": i, "preco": preco_entrada, "qtde": qtde})
#                     posicao = qtde
#                     saldo -= qtde * preco_entrada
#             # elif row["Trade"] == -1 and posicao > 0:
#             elif not pd.isna(row["Trade"]) and row["Trade"] == -1 and posicao > 0:
#                 # Vender tudo
#                 preco_saida = row["Close"]
#                 entradas[-1].update({"data_saida": i, "preco_saida": preco_saida, "resultado": (preco_saida-preco_entrada)*posicao})
#                 saldo += preco_saida * posicao
#                 posicao = 0
#         # Fecha posição aberta no último dia
#         if posicao > 0:
#             preco_final = data.iloc[-1]["Close"]
#             entradas[-1].update({"data_saida": data.index[-1], "preco_saida": preco_final, "resultado": (preco_final-preco_entrada)*posicao})
#             saldo += preco_final * posicao
#             posicao = 0

#         resultado_total = saldo - (capital / len(symbols.split(",")))
#         resultados.append({
#             "Ação": symbol,
#             "Retorno (USD)": resultado_total,
#             "Retorno (%)": 100*resultado_total/(capital/len(symbols.split(","))),
#             "Qtd trades": len(entradas)
#         })

#         # Gráfico
#         st.markdown(f"#### {symbol} - Sinais de Compra/Venda")
#         fig = go.Figure()
#         fig.add_trace(go.Scatter(x=data.index, y=data["Close"], name="Preço de Fechamento", line=dict(color="black")))
#         fig.add_trace(go.Scatter(x=data.index, y=data["MA_curta"], name=f"MA Curta ({janela_ma_curta})", line=dict(color="blue")))
#         fig.add_trace(go.Scatter(x=data.index, y=data["MA_longa"], name=f"MA Longa ({janela_ma_longa})", line=dict(color="orange")))
#         for ent in entradas:
#             fig.add_vline(x=ent["data"], line_dash="dash", line_color="green")
#         for ent in entradas:
#             if "data_saida" in ent:
#                 fig.add_vline(x=ent["data_saida"], line_dash="dash", line_color="red")
#         fig.update_layout(height=350, legend=dict(font=dict(size=14)))
#         st.plotly_chart(fig, use_container_width=True)

#     # Tabela de resultados
#     st.markdown("### Resultados gerais")
#     resultados_df = pd.DataFrame(resultados)
#     st.dataframe(resultados_df.style.format({
#         "Retorno (USD)": "{:.2f}",
#         "Retorno (%)": "{:.2f}%"
#     }))

#     st.info("""
#     <b>Observação:</b> O sistema aqui simula apenas. Antes de operar em conta real, valide exaustivamente por meses e ajuste sua estratégia. 
#     Em investimentos reais, ganhos passados não garantem resultados futuros.<br>
#     Estratégias conservadoras tendem a proteger melhor o capital, mas nunca eliminam totalmente o risco.
#     """, unsafe_allow_html=True)







import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objs as go
from alpaca_trade_api.rest import REST
import time

def render():
    st.title("💸 Teste1 - Estratégia de Trading Conservadora (Simulação Alpaca Paper)")

    st.markdown("""
    <div style='font-size:18px;'>
    <b>Descrição:</b> <br>
    Simule operações de compra e venda automatizada de ações usando a API da Alpaca em <b>modo paper</b>.<br>
    Esta estratégia é conservadora, baseada em cruzamento de médias móveis.<br>
    <b>Observação:</b> Para operar em conta real na Alpaca, é necessário ser residente nos EUA.<br>
    <b>Importante:</b> Esta aba executa simulações históricas. Para execução contínua automática, utilize um script Python fora do Streamlit.<br>
    </div>
    """, unsafe_allow_html=True)

    st.header("Configuração da Simulação")

    API_KEY = st.text_input("Alpaca API Key", type="password")
    API_SECRET = st.text_input("Alpaca API Secret", type="password")
    BASE_URL = "https://paper-api.alpaca.markets"

    symbols = st.text_input("Ações para simular (separadas por vírgula)", value="AAPL,MSFT,GOOGL,AMZN")
    capital = st.number_input("Capital inicial de simulação (USD)", value=10000, min_value=1000, step=100)
    janela_ma_curta = st.number_input("Período da Média Móvel Curta (dias)", min_value=3, max_value=20, value=5)
    janela_ma_longa = st.number_input("Período da Média Móvel Longa (dias)", min_value=10, max_value=60, value=21)
    periodo_analise = st.selectbox("Período histórico para análise", ["1y", "6mo", "3mo", "1mo"], index=0)

    executar = st.button("Executar Simulação", type="primary")

    if not (API_KEY and API_SECRET):
        st.info("Por favor, informe sua API Key e Secret da Alpaca (modo paper).")
        st.stop()

    if not executar:
        st.stop()

    try:
        api = REST(API_KEY, API_SECRET, BASE_URL)
        st.success("Conexão bem-sucedida com a API Alpaca em modo paper.")
    except Exception as e:
        st.error(f"Erro ao conectar na API Alpaca: {str(e)}")
        st.stop()

    st.subheader("Resultados da Estratégia")

    resultados = []
    for symbol in [s.strip().upper() for s in symbols.split(",")]:
        # data = yf.download(symbol, period=periodo_analise)


        data = yf.download(symbol, period=periodo_analise)

        # # Debug visual
        # st.write(f"Debug - {symbol}", data.head())

        
        # # Si el DataFrame tiene MultiIndex en columnas, selecciona la columna 'Close'
        # if isinstance(data.columns, pd.MultiIndex):
        #     # Si solo descargaste 1 símbolo, deshaz MultiIndex
        #     try:
        #         data.columns = data.columns.get_level_values(1)
        #     except:
        #         st.warning(f"Erro ao ajustar as colunas para {symbol}. Pulei este ativo.")
        #         continue

        # if data is None or data.empty or "Close" not in data.columns:
        #     st.warning(f"Sem dados históricos válidos para {symbol}. Pulei este ativo.")
        #     continue

        # data = data.dropna(subset=["Close"])
        # if data.empty:
        #     st.warning(f"Todos os registros de {symbol} estão vazios após limpeza. Pulei este ativo.")
        #     continue



        # Si las columnas son MultiIndex, selecciona solo el nivel correspondiente
        if isinstance(data.columns, pd.MultiIndex):
            # Selecciona solo el símbolo actual
            if symbol in data.columns.get_level_values(1):
                # "Close" para este símbolo es una tupla ("Close", symbol)
                data = data.xs(symbol, axis=1, level=1, drop_level=False)
                data.columns = data.columns.get_level_values(0)  # Deja "Close", "Open", etc como columnas planas
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
        data["Signal"][janela_ma_longa:] = np.where(
            data["MA_curta"][janela_ma_longa:] > data["MA_longa"][janela_ma_longa:], 1, 0
        )
        data["Trade"] = data["Signal"].diff()

        posicao = 0
        saldo = capital / len(symbols.split(","))
        entradas = []
        preco_entrada = None

        # for i, row in data.iterrows():
        #     trade_val = row["Trade"]
        #     if not pd.isna(trade_val) and float(trade_val) == 1 and posicao == 0:
        #         preco_entrada = row["Close"]
        #         qtde = saldo // preco_entrada
        #         if qtde > 0:
        #             entradas.append({"data": i, "preco": preco_entrada, "qtde": qtde})
        #             posicao = qtde
        #             saldo -= qtde * preco_entrada
        #     elif not pd.isna(trade_val) and float(trade_val) == -1 and posicao > 0 and preco_entrada is not None:
        #         preco_saida = row["Close"]
        #         entradas[-1].update({"data_saida": i, "preco_saida": preco_saida, "resultado": (preco_saida-preco_entrada)*posicao})
        #         saldo += preco_saida * posicao
        #         posicao = 0
        #         preco_entrada = None

        # st.write(data.tail(20))

        for i, row in data.iterrows():
            trade_val = row["Trade"]
            try:
                # Asegúrate que trade_val es un número escalar (no Serie, ni nan)
                val = float(trade_val)
            except:
                continue  # Si no es convertible a float, ignora la fila

            if val == 1 and posicao == 0:
                preco_entrada = row["Close"]
                # qtde = saldo // preco_entrada
                # if qtde > 0:
                #     entradas.append({"data": i, "preco": preco_entrada, "qtde": qtde})
                #     posicao = qtde
                #     saldo -= qtde * preco_entrada
                qtde = int(saldo // preco_entrada)
                if qtde >= 1:
                    entradas.append({"data": i, "preco": preco_entrada, "qtde": qtde})
                    posicao = qtde
                    saldo -= qtde * preco_entrada

            elif val == -1 and posicao > 0 and preco_entrada is not None:
                preco_saida = row["Close"]
                entradas[-1].update({"data_saida": i, "preco_saida": preco_saida, "resultado": (preco_saida-preco_entrada)*posicao})
                saldo += preco_saida * posicao
                posicao = 0
                preco_entrada = None





        if posicao > 0 and preco_entrada is not None:
            preco_final = data.iloc[-1]["Close"]
            entradas[-1].update({"data_saida": data.index[-1], "preco_saida": preco_final, "resultado": (preco_final-preco_entrada)*posicao})
            saldo += preco_final * posicao
            posicao = 0
            preco_entrada = None

        resultado_total = saldo - (capital / len(symbols.split(",")))
        resultados.append({
            "Ação": symbol,
            "Retorno (USD)": resultado_total,
            "Retorno (%)": 100*resultado_total/(capital/len(symbols.split(","))),
            "Qtd trades": len(entradas)
        })

        st.markdown(f"#### {symbol} - Sinais de Compra/Venda")
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=data.index, y=data["Close"], name="Preço de Fechamento", line=dict(color="black")))
        fig.add_trace(go.Scatter(x=data.index, y=data["MA_curta"], name=f"MA Curta ({janela_ma_curta})", line=dict(color="blue")))
        fig.add_trace(go.Scatter(x=data.index, y=data["MA_longa"], name=f"MA Longa ({janela_ma_longa})", line=dict(color="orange")))
        for ent in entradas:
            fig.add_vline(x=ent["data"], line_dash="dash", line_color="green")
        for ent in entradas:
            if "data_saida" in ent:
                fig.add_vline(x=ent["data_saida"], line_dash="dash", line_color="red")
        fig.update_layout(height=350, legend=dict(font=dict(size=14)))
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("### Resultados gerais")
    resultados_df = pd.DataFrame(resultados)
    st.dataframe(resultados_df.style.format({
        "Retorno (USD)": "{:.2f}",
        "Retorno (%)": "{:.2f}%"
    }))

    st.markdown("""
    <div style='background-color:#dbeafe; padding:12px; border-radius:8px'>
    <b>Observação:</b> O sistema aqui simula apenas. Antes de operar em conta real, valide exaustivamente por meses e ajuste sua estratégia.<br>
    Em investimentos reais, ganhos passados não garantem resultados futuros.<br>
    Estratégias conservadoras tendem a proteger melhor o capital, mas nunca eliminam totalmente o risco.<br>
    Para execução automática/contínua, desenvolva um script Python separado (fora do Streamlit).
    </div>
    """, unsafe_allow_html=True)





    # # Descarga precios de cierre ajustados
    # tickers = ['AMZN', 'MSFT', 'WMT', 'BBY']
    # data = yf.download(tickers, period='2y')['Adj Close'].dropna()

    # # Ejemplo: calcular retornos diarios
    # returns = data.pct_change().dropna()

    # # Correlación cruzada entre AMZN y cada otro activo (lags de -20 a +20)
    # for other in ['MSFT', 'WMT', 'BBY']:
    #     lags = range(-20, 21)
    #     corr_lags = [returns['AMZN'].corr(returns[other].shift(lag)) for lag in lags]
    #     plt.plot(lags, corr_lags, label=other)
    # plt.axvline(0, color='gray', linestyle='--')
    # plt.legend()
    # plt.title('Correlación cruzada (lags)')
    # plt.xlabel('Desfase (días)')
    # plt.ylabel('Correlación')
    # plt.show()

    # # Análisis de causalidad de Granger (ejemplo con MSFT)
    # result = grangercausalitytests(returns[['AMZN', 'MSFT']], maxlag=10, verbose=True)



    import matplotlib.pyplot as plt

    tickers = ['AMZN', 'MSFT', 'WMT', 'BBY']
    data = yf.download(tickers, period='2y')
    # print(data.columns)  # Debug visual

    # ¿Qué columna usar?
    if 'Adj Close' in data.columns:
        # Múltiples tickers dan un MultiIndex
        precios = data['Adj Close']
    elif 'Close' in data.columns:
        precios = data['Close']
    else:
        raise ValueError("No se encontró 'Adj Close' ni 'Close' en los datos descargados.")

    precios = precios.dropna()

    returns = precios.pct_change().dropna()


    print(precios.head())

    lags = range(-20, 21)
    for other in ['MSFT', 'WMT', 'BBY']:
        corr_lags = [returns['AMZN'].corr(returns[other].shift(lag)) for lag in lags]
        plt.plot(lags, corr_lags, label=other)
    plt.axvline(0, color='gray', linestyle='--')
    plt.legend()
    plt.title('Correlación cruzada de retornos con AMZN (lags)')
    plt.xlabel('Desfase (días)')
    plt.ylabel('Correlación')
    plt.show()