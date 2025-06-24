# import streamlit as st
# import pandas as pd
# import numpy as np
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.metrics import classification_report
# import matplotlib.pyplot as plt

# def compute_signals(prices, fast=5, slow=21):
#     if not isinstance(prices, pd.Series):
#         raise ValueError("A variável 'prices' deve ser uma Series do pandas.")
#     df = pd.DataFrame({'close': prices})
#     df['ma_fast'] = df['close'].rolling(fast).mean()
#     df['ma_slow'] = df['close'].rolling(slow).mean()
#     df['signal'] = 0
#     df.loc[df['ma_fast'] > df['ma_slow'], 'signal'] = 1
#     df.loc[df['ma_fast'] < df['ma_slow'], 'signal'] = -1
#     df['cross'] = df['signal'].diff().fillna(0)
#     df['trade_signal'] = 0
#     df.loc[df['cross'] == 2, 'trade_signal'] = 1    # Golden cross (buy)
#     df.loc[df['cross'] == -2, 'trade_signal'] = -1   # Death cross (sell)
#     return df

# def build_features_multivariate(df_signals, prices_matrix, aligned_order, lookback=7, horizon=5):
#     feats = []
#     labels = []
#     rets = prices_matrix.pct_change().fillna(0).values
#     for i in range(lookback, len(df_signals) - horizon):
#         if df_signals['trade_signal'].iloc[i] == 0:
#             continue
#         window = rets[i - lookback:i, :].flatten()
#         signal = df_signals['trade_signal'].iloc[i]
#         future_ret = df_signals['close'].iloc[i+1:i+1+horizon].pct_change().sum()
#         label = 1 if (signal == 1 and future_ret > 0) or (signal == -1 and future_ret < 0) else 0
#         feats.append(window)
#         labels.append(label)
#     return np.array(feats), np.array(labels)

# def train_rf_classifier(X, y):
#     model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight="balanced")
#     model.fit(X, y)
#     return model

# def simulate_strategy(df_signals, prices_matrix, aligned_order, model, lookback=7, threshold=0.6, horizon=5):
#     df = df_signals.copy()
#     df['rf_pred'] = 0
#     df['rf_proba'] = 0.0
#     equity = [1.0]
#     signals = []
#     rets = prices_matrix.pct_change().fillna(0).values
#     for i in range(lookback, len(df)-1):
#         if df['trade_signal'].iloc[i] == 0:
#             equity.append(equity[-1] if equity else 1.0)
#             continue
#         window = rets[i - lookback:i, :].flatten().reshape(1, -1)
#         proba = model.predict_proba(window)[0,1]
#         pred = int(proba > threshold)
#         df.at[df.index[i], 'rf_pred'] = pred
#         df.at[df.index[i], 'rf_proba'] = proba
#         if pred:
#             ret = (df['close'].iloc[i+1] - df['close'].iloc[i]) / df['close'].iloc[i]
#             if df['trade_signal'].iloc[i] == 1:
#                 equity.append(equity[-1] * (1 + ret))
#                 signals.append((df.index[i], "BUY", round(100*ret,2), proba))
#             else:
#                 equity.append(equity[-1] * (1 - ret))
#                 signals.append((df.index[i], "SELL", round(-100*ret,2), proba))
#         else:
#             equity.append(equity[-1])
#     df['equity_hibrida'] = equity[:len(df)]
#     return df, signals

# def plot_equity_curve(df):
#     fig, ax = plt.subplots(figsize=(10, 4))
#     ax.plot(df.index, df['equity_hibrida'], label="Estratégia Híbrida", color="blue")
#     ax.set_ylabel("Equity (Crescimento do capital)")
#     ax.set_title("Curva de Equity - Estratégia Híbrida")
#     ax.legend()
#     st.pyplot(fig)

# def show():
#     st.title("Estratégia Híbrida Quant-IA com Alineamento")

#     # --- CARGA DAS VARIÁVEIS DE SESSÃO ---
#     if not all(key in st.session_state for key in ['tickers_alinhados', 'dados_alinhados', 'ordem_alinhada']):
#         st.error("Primeiro, gere o alinhamento na aba de Exploração para continuar.")
#         return
#     tickers = st.session_state['tickers_alinhados']
#     prices_matrix = st.session_state['dados_alinhados']
#     aligned_order = st.session_state['ordem_alinhada']

#     fast = st.slider("Período média móvel curta", 3, 15, 5)
#     slow = st.slider("Período média móvel longa", 10, 40, 21)
#     lookback = st.slider("Dias usados como histórico (features)", 3, 14, 7)
#     threshold = st.slider("Threshold mínimo do modelo (probabilidade)", 0.5, 0.95, 0.6)
#     horizon = st.slider("Janela para avaliar acerto do trade (dias)", 2, 10, 5)

#     st.write("Ações alinhadas:", " → ".join(aligned_order))
#     st.dataframe(prices_matrix)

#     # Usa a primeira ação do alinhamento como principal para sinais de trade
#     main_ticker = aligned_order[0]
#     main_prices = prices_matrix[main_ticker]
#     st.line_chart(main_prices, use_container_width=True)

#     # Compute trade signals (médias móveis) na ação principal
#     df_signals = compute_signals(main_prices, fast=fast, slow=slow)

#     # Gera features multivariadas usando as ações alinhadas
#     X, y = build_features_multivariate(df_signals, prices_matrix[aligned_order], aligned_order, lookback=lookback, horizon=horizon)
#     if len(X) < 10:
#         st.error("Poucos exemplos para treinar o modelo. Tente mais dias ou outro grupo de ações.")
#         return

#     model = train_rf_classifier(X, y)

#     st.subheader("Relatório do modelo Random Forest")
#     st.text(classification_report(y, model.predict(X)))

#     df_h, signals = simulate_strategy(df_signals, prices_matrix[aligned_order], aligned_order, model, lookback=lookback, threshold=threshold, horizon=horizon)
#     plot_equity_curve(df_h)

#     st.subheader("Sinais de trade confirmados pelo modelo")
#     st.dataframe(pd.DataFrame(signals, columns=["Data", "Tipo", "Retorno (%)", "Probabilidade modelo"]))

# if __name__ == "__main__":
#     show()







# import streamlit as st
# import pandas as pd
# import numpy as np
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.metrics import classification_report
# import matplotlib.pyplot as plt

# plt.style.use("seaborn-v0_8-darkgrid")

# def compute_signals(prices, fast=5, slow=21):
#     df = pd.DataFrame({'close': prices})
#     df['ma_fast'] = df['close'].rolling(fast).mean()
#     df['ma_slow'] = df['close'].rolling(slow).mean()
#     df['signal'] = 0
#     df.loc[df['ma_fast'] > df['ma_slow'], 'signal'] = 1
#     df.loc[df['ma_fast'] < df['ma_slow'], 'signal'] = -1
#     df['cross'] = df['signal'].diff().fillna(0)
#     df['trade_signal'] = 0
#     df.loc[df['cross'] == 2, 'trade_signal'] = 1    # Golden cross (compra)
#     df.loc[df['cross'] == -2, 'trade_signal'] = -1   # Death cross (venda)
#     return df

# def build_features_multivariate(df_signals, prices_matrix, aligned_order, lookback=7, horizon=5):
#     feats = []
#     labels = []
#     rets = prices_matrix.pct_change().fillna(0).values
#     for i in range(lookback, len(df_signals) - horizon):
#         if df_signals['trade_signal'].iloc[i] == 0:
#             continue
#         window = rets[i - lookback:i, :].flatten()
#         signal = df_signals['trade_signal'].iloc[i]
#         future_ret = df_signals['close'].iloc[i+1:i+1+horizon].pct_change().sum()
#         label = 1 if (signal == 1 and future_ret > 0) or (signal == -1 and future_ret < 0) else 0
#         feats.append(window)
#         labels.append(label)
#     return np.array(feats), np.array(labels)

# def train_rf_classifier(X, y):
#     model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight="balanced")
#     model.fit(X, y)
#     return model

# def simulate_strategy(df_signals, prices, model=None, prices_matrix=None, aligned_order=None, lookback=7, threshold=0.6, horizon=5):
#     df = df_signals.copy()
#     df['rf_pred'] = 0
#     df['rf_proba'] = 0.0
#     equity_ma = [1.0]
#     equity_hibrida = [1.0]
#     signals_ma = []
#     signals_rf = []
#     # returns multivariados
#     if model is not None and prices_matrix is not None:
#         rets = prices_matrix[aligned_order].pct_change().fillna(0).values
#     else:
#         rets = None

#     for i in range(len(df)-1):
#         # Media móvel clássica
#         if df['trade_signal'].iloc[i] != 0:
#             ret = (df['close'].iloc[i+1] - df['close'].iloc[i]) / df['close'].iloc[i]
#             if df['trade_signal'].iloc[i] == 1:
#                 equity_ma.append(equity_ma[-1] * (1 + ret))
#                 signals_ma.append((df.index[i], "BUY", round(100*ret,2)))
#             else:
#                 equity_ma.append(equity_ma[-1] * (1 - ret))
#                 signals_ma.append((df.index[i], "SELL", round(-100*ret,2)))
#         else:
#             equity_ma.append(equity_ma[-1])
#         # Estratégia híbrida
#         if model is not None and prices_matrix is not None and rets is not None and i >= lookback:
#             if df['trade_signal'].iloc[i] == 0:
#                 equity_hibrida.append(equity_hibrida[-1])
#                 continue
#             window = rets[i - lookback:i, :].flatten().reshape(1, -1)
#             proba = model.predict_proba(window)[0,1]
#             pred = int(proba > threshold)
#             df.at[df.index[i], 'rf_pred'] = pred
#             df.at[df.index[i], 'rf_proba'] = proba
#             if pred:
#                 if df['trade_signal'].iloc[i] == 1:
#                     equity_hibrida.append(equity_hibrida[-1] * (1 + ret))
#                     signals_rf.append((df.index[i], "BUY", round(100*ret,2), proba))
#                 else:
#                     equity_hibrida.append(equity_hibrida[-1] * (1 - ret))
#                     signals_rf.append((df.index[i], "SELL", round(-100*ret,2), proba))
#             else:
#                 equity_hibrida.append(equity_hibrida[-1])
#         elif model is not None:
#             equity_hibrida.append(equity_hibrida[-1])

#     df['equity_ma'] = equity_ma[:len(df)]
#     df['equity_hibrida'] = equity_hibrida[:len(df)]
#     return df, signals_ma, signals_rf

# def plot_comparison_equity(df, ticker):
#     fig, ax = plt.subplots(figsize=(12, 5))
#     ax.plot(df.index, df['equity_ma'], label="Média Móvel", lw=2, color="#efcb68")
#     ax.plot(df.index, df['equity_hibrida'], label="Média Móvel + IA (Híbrida)", lw=2, color="#0070F3")
#     ax.set_ylabel("Equity (Crescimento do capital)")
#     ax.set_xlabel("Data")
#     ax.set_title(f"Curva de Equity: {ticker}", fontsize=16, fontweight='bold', color='#222831')
#     ax.legend(fontsize=13, loc='upper left')
#     ax.grid(True, linestyle='--', alpha=0.7)
#     st.pyplot(fig)

# def show():
#     st.title("Estratégia Quantitativa Híbrida — Simulação Interativa")
#     st.write(
#         "Explore uma abordagem moderna de trading quantitativo, combinando o poder das médias móveis com um modelo preditivo multivariado. "
#         "Selecione o histórico, escolha ações alinhadas e visualize o impacto do suporte de IA na sua tomada de decisão."
#     )

#     # Checa dados da sessão
#     if not all(k in st.session_state for k in ['tickers_alinhados', 'dados_alinhados', 'ordem_alinhada', 'dias_historico']):
#         st.error("Primeiro, gere o alinhamento na aba de Exploração para continuar.")
#         return
#     aligned_order = st.session_state['ordem_alinhada']
#     prices_matrix = st.session_state['dados_alinhados']
#     dias_historico = st.session_state['dias_historico']

#     st.info(
#         f"Ações alinhadas por máxima correlação: {' → '.join(aligned_order)}"
#     )

#     # Multiselect para selecionar até 5 ações principais (pré-selecionadas as 5 primeiras)
#     ativos_default = aligned_order[:5]
#     ativos_escolhidos = st.multiselect(
#         "Selecione até 5 ações principais para simular a estratégia:",
#         aligned_order,
#         default=ativos_default,
#         max_selections=5,
#         help="Estas são as ações principais onde a estratégia será simulada. Altere à vontade!"
#     )

#     # Parâmetros ajustáveis
#     fast = st.slider("Período média móvel curta", 3, 15, 5)
#     slow = st.slider("Período média móvel longa", 10, 40, 21)
#     lookback = st.slider("Dias de histórico (features IA)", 3, 20, 7)
#     threshold = st.slider("Threshold mínimo do modelo IA (probabilidade)", 0.5, 0.95, 0.6)
#     horizon = st.slider("Janela para avaliar acerto do trade (dias)", 2, 10, 5)

#     for ticker in ativos_escolhidos:
#         st.subheader(f"Resultados para: {ticker}")
#         main_prices = prices_matrix[ticker]
#         st.line_chart(main_prices, use_container_width=True)

#         # Sinais médias móveis
#         df_signals = compute_signals(main_prices, fast=fast, slow=slow)

#         # Features multivariadas para IA (usa todas as ações alinhadas)
#         X, y = build_features_multivariate(df_signals, prices_matrix[aligned_order], aligned_order, lookback=lookback, horizon=horizon)
#         if len(X) < 10:
#             st.warning(
#                 "Poucos exemplos para treinar o modelo. "
#                 "Tente aumentar o período de dados históricos ou ajustar os parâmetros para este ativo."
#             )
#             continue

#         # Treina o modelo Random Forest para filtrar sinais
#         model = train_rf_classifier(X, y)

#         st.text("Desempenho do modelo Random Forest (treino):")
#         st.text(classification_report(y, model.predict(X)))

#         df_result, sinais_ma, sinais_rf = simulate_strategy(
#             df_signals, main_prices,
#             model=model,
#             prices_matrix=prices_matrix,
#             aligned_order=aligned_order,
#             lookback=lookback,
#             threshold=threshold,
#             horizon=horizon
#         )
#         plot_comparison_equity(df_result, ticker)

#         st.markdown(f"#### Sinais validados pelo modelo (IA):")
#         df_rf = pd.DataFrame(sinais_rf, columns=["Data", "Tipo", "Retorno (%)", "Prob. modelo"])
#         st.dataframe(df_rf, use_container_width=True)
#         st.markdown("---")

# if __name__ == "__main__":
#     show()







import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt

plt.style.use("seaborn-v0_8-darkgrid")

def compute_signals(prices, fast=5, slow=21):
    df = pd.DataFrame({'close': prices})
    df['ma_fast'] = df['close'].rolling(fast).mean()
    df['ma_slow'] = df['close'].rolling(slow).mean()
    df['signal'] = 0
    df.loc[df['ma_fast'] > df['ma_slow'], 'signal'] = 1
    df.loc[df['ma_fast'] < df['ma_slow'], 'signal'] = -1
    df['cross'] = df['signal'].diff().fillna(0)
    df['trade_signal'] = 0
    df.loc[df['cross'] == 2, 'trade_signal'] = 1    # Golden cross (compra)
    df.loc[df['cross'] == -2, 'trade_signal'] = -1   # Death cross (venda)
    return df

def build_features_multivariate(df_signals, prices_matrix, aligned_order, lookback=7, horizon=5):
    feats = []
    labels = []
    idxs = []
    rets = prices_matrix[aligned_order].pct_change().fillna(0).values
    for i in range(lookback, len(df_signals) - horizon):
        if df_signals['trade_signal'].iloc[i] == 0:
            continue
        window = rets[i - lookback:i, :].flatten()
        signal = df_signals['trade_signal'].iloc[i]
        # futuro retorno da ação individual:
        future_ret = df_signals['close'].iloc[i+1:i+1+horizon].pct_change().sum()
        label = 1 if (signal == 1 and future_ret > 0) or (signal == -1 and future_ret < 0) else 0
        feats.append(window)
        labels.append(label)
        idxs.append(i)
    return np.array(feats), np.array(labels), np.array(idxs)

def train_rf_classifier(X, y):
    model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight="balanced")
    model.fit(X, y)
    return model

def simulate_strategy(df_signals, prices, model=None, prices_matrix=None, aligned_order=None, lookback=7, threshold=0.6, horizon=5, idxs_test=None):
    df = df_signals.copy()
    df['rf_pred'] = 0
    df['rf_proba'] = 0.0
    equity_ma = [1.0]
    equity_hibrida = [1.0]
    signals_ma = []
    signals_rf = []
    # returns multivariados
    if model is not None and prices_matrix is not None:
        rets = prices_matrix[aligned_order].pct_change().fillna(0).values
    else:
        rets = None

    for i in range(len(df)-1):
        # Media móvel clássica
        if df['trade_signal'].iloc[i] != 0:
            ret = (df['close'].iloc[i+1] - df['close'].iloc[i]) / df['close'].iloc[i]
            if df['trade_signal'].iloc[i] == 1:
                equity_ma.append(equity_ma[-1] * (1 + ret))
                signals_ma.append((df.index[i], "BUY", round(100*ret,2)))
            else:
                equity_ma.append(equity_ma[-1] * (1 - ret))
                signals_ma.append((df.index[i], "SELL", round(-100*ret,2)))
        else:
            equity_ma.append(equity_ma[-1])
        # Estratégia híbrida (solo para teste)
        if model is not None and prices_matrix is not None and rets is not None and i >= lookback and idxs_test is not None:
            if df['trade_signal'].iloc[i] == 0 or i not in idxs_test:
                equity_hibrida.append(equity_hibrida[-1])
                continue
            window = rets[i - lookback:i, :].flatten().reshape(1, -1)
            proba = model.predict_proba(window)[0,1]
            pred = int(proba > threshold)
            df.at[df.index[i], 'rf_pred'] = pred
            df.at[df.index[i], 'rf_proba'] = proba
            if pred:
                if df['trade_signal'].iloc[i] == 1:
                    equity_hibrida.append(equity_hibrida[-1] * (1 + ret))
                    signals_rf.append((df.index[i], "BUY", round(100*ret,2), proba))
                else:
                    equity_hibrida.append(equity_hibrida[-1] * (1 - ret))
                    signals_rf.append((df.index[i], "SELL", round(-100*ret,2), proba))
            else:
                equity_hibrida.append(equity_hibrida[-1])
        elif model is not None:
            equity_hibrida.append(equity_hibrida[-1])

    df['equity_ma'] = equity_ma[:len(df)]
    df['equity_hibrida'] = equity_hibrida[:len(df)]
    return df, signals_ma, signals_rf

def plot_comparison_equity(df, ticker):
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(df.index, df['equity_ma'], label="Média Móvel", lw=2, color="#efcb68")
    ax.plot(df.index, df['equity_hibrida'], label="Média Móvel + IA (Híbrida)", lw=2, color="#0070F3")
    ax.set_ylabel("Equity (Crescimento do capital)")
    ax.set_xlabel("Data")
    ax.set_title(f"Curva de Equity: {ticker}", fontsize=16, fontweight='bold', color='#222831')
    ax.legend(fontsize=13, loc='upper left')
    ax.grid(True, linestyle='--', alpha=0.7)
    st.pyplot(fig)

def show():
    st.title("Estratégia Quantitativa Híbrida — Simulação Interativa")
    st.write(
        "Explore uma abordagem moderna de trading quantitativo, combinando o poder das médias móveis com um modelo preditivo multivariado. "
        "Selecione o histórico, escolha ações alinhadas e visualize o impacto do suporte de IA na sua tomada de decisão."
    )

    # Checa dados da sessão
    if not all(k in st.session_state for k in ['tickers_alinhados', 'dados_alinhados', 'ordem_alinhada', 'dias_historico']):
        st.error("Primeiro, gere o alinhamento na aba de Exploração para continuar.")
        return
    aligned_order = st.session_state['ordem_alinhada']
    prices_matrix = st.session_state['dados_alinhados']
    dias_historico = st.session_state['dias_historico']

    st.info(
        f"Ações alinhadas por máxima correlação: {' → '.join(aligned_order)}"
    )

    # Multiselect para selecionar até 5 ações principais (pré-selecionadas as 5 primeiras)
    ativos_default = aligned_order[:5]
    ativos_escolhidos = st.multiselect(
        "Selecione até 5 ações principais para simular a estratégia:",
        aligned_order,
        default=ativos_default,
        max_selections=5,
        help="Estas são as ações principais onde a estratégia será simulada. Altere à vontade!"
    )

    # Parâmetros ajustáveis
    fast = st.slider("Período média móvel curta", 3, 15, 5)
    slow = st.slider("Período média móvel longa", 10, 40, 21)
    lookback = st.slider("Dias de histórico (features IA)", 3, 20, 7)
    threshold = st.slider("Threshold mínimo do modelo IA (probabilidade)", 0.5, 0.95, 0.6)
    horizon = st.slider("Janela para avaliar acerto do trade (dias)", 2, 10, 5)

    for ticker in ativos_escolhidos:
        st.subheader(f"Resultados para: {ticker}")
        main_prices = prices_matrix[ticker]
        st.line_chart(main_prices, use_container_width=True)

        # Sinais médias móveis
        df_signals = compute_signals(main_prices, fast=fast, slow=slow)

        # Features multivariadas para IA (usa todas as ações alinhadas)
        X, y, idxs = build_features_multivariate(df_signals, prices_matrix[aligned_order], aligned_order, lookback=lookback, horizon=horizon)
        if len(X) < 10:
            st.warning(
                "Poucos exemplos para treinar o modelo. "
                "Tente aumentar o período de dados históricos ou ajustar os parâmetros para este ativo."
            )
            continue

        # Split: 60% treino, 20% validação, 20% teste
        n = len(X)
        idx_train = int(n * 0.6)
        idx_val = int(n * 0.8)
        X_train, y_train = X[:idx_train], y[:idx_train]
        X_val, y_val = X[idx_train:idx_val], y[idx_train:idx_val]
        X_test, y_test = X[idx_val:], y[idx_val:]
        idxs_test = idxs[idx_val:]

        model = train_rf_classifier(X_train, y_train)

        st.markdown("#### Desempenho do modelo Random Forest (teste - dados futuros):")
        y_pred_test = model.predict(X_test)
        st.text(classification_report(y_test, y_pred_test))

        df_result, sinais_ma, sinais_rf = simulate_strategy(
            df_signals, main_prices,
            model=model,
            prices_matrix=prices_matrix,
            aligned_order=aligned_order,
            lookback=lookback,
            threshold=threshold,
            horizon=horizon,
            idxs_test=idxs_test
        )
        plot_comparison_equity(df_result, ticker)

        st.markdown(f"#### Sinais validados pelo modelo (IA) no teste:")
        df_rf = pd.DataFrame(sinais_rf, columns=["Data", "Tipo", "Retorno (%)", "Prob. modelo"])
        st.dataframe(df_rf, use_container_width=True)
        st.markdown("---")

if __name__ == "__main__":
    show()
