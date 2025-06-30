#menu/quant_ia.py
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

def extract_trades(df_signals):
    """Identifica trades completos: cada compra até a próxima venda e vice-versa."""
    trades = []
    entries = df_signals.index[df_signals['trade_signal'] != 0].tolist()
    signals = df_signals.loc[entries, 'trade_signal'].tolist()
    # Só consideramos posições long (compra->venda)
    i = 0
    while i < len(entries) - 1:
        if signals[i] == 1:  # Buy
            entry_idx = entries[i]
            # Busca próxima venda
            for j in range(i+1, len(entries)):
                if signals[j] == -1:
                    exit_idx = entries[j]
                    trades.append((entry_idx, exit_idx, 1))  # 1 = long
                    i = j
                    break
            else:
                break
        else:
            i += 1
    return trades

def build_features_multivariate_trades(df_signals, prices_matrix, aligned_order, lookback=7):
    """Constrói features apenas para os trades completos de compra→venda."""
    trades = extract_trades(df_signals)
    feats, labels, idxs = [], [], []
    rets = prices_matrix[aligned_order].pct_change().fillna(0).values
    close_prices = df_signals['close'].values
    for entry_idx, exit_idx, _ in trades:
        i = df_signals.index.get_loc(entry_idx)
        j = df_signals.index.get_loc(exit_idx)
        if i < lookback or j <= i:
            continue
        # Features: janela antes da entrada (multivariada)
        window = rets[i - lookback:i, :].flatten()
        # Label: trade deu lucro?
        ret = (close_prices[j] - close_prices[i]) / close_prices[i]
        label = 1 if ret > 0 else 0
        feats.append(window)
        labels.append(label)
        idxs.append((i, j, ret))  # início, fim, retorno real
    return np.array(feats), np.array(labels), np.array(idxs)

def train_rf_classifier(X, y):
    model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight="balanced")
    model.fit(X, y)
    return model

def simulate_strategy_trades(df_signals, prices, model=None, prices_matrix=None, aligned_order=None, lookback=7, threshold=0.6, idxs_test=None, feats_test=None):
    """Simula o equity de cada estratégia: MA puro vs Híbrida (só trades aprovados pela IA)."""
    close_prices = df_signals['close'].values
    equity_ma = [1.0]
    equity_hibrida = [1.0]
    trades_ma = []
    trades_hibrida = []

    # Extrai trades completos
    trades = extract_trades(df_signals)
    # Se fornecido, apenas os trades do teste
    if idxs_test is not None:
        test_set = set(tuple(x[:2]) for x in idxs_test)
        trades = [t for t in trades if (df_signals.index.get_loc(t[0]), df_signals.index.get_loc(t[1])) in test_set]

    # Simulação
    pos_ma = 1.0
    pos_hib = 1.0
    for k, (entry_idx, exit_idx, _) in enumerate(trades):
        i = df_signals.index.get_loc(entry_idx)
        j = df_signals.index.get_loc(exit_idx)
        entry_price = close_prices[i]
        exit_price = close_prices[j]
        ret = (exit_price - entry_price) / entry_price
        # Média móvel clássica: entra sempre
        pos_ma = pos_ma * (1 + ret)
        equity_ma.append(pos_ma)
        trades_ma.append((df_signals.index[i], df_signals.index[j], round(100*ret,2)))
        # Híbrida: só se a IA aprova
        if model is not None and feats_test is not None:
            window = feats_test[k].reshape(1, -1)
            proba = model.predict_proba(window)[0,1]
            pred = int(proba > threshold)
            if pred:
                pos_hib = pos_hib * (1 + ret)
                equity_hibrida.append(pos_hib)
                trades_hibrida.append((df_signals.index[i], df_signals.index[j], round(100*ret,2), proba))
            else:
                equity_hibrida.append(pos_hib)
        else:
            equity_hibrida.append(pos_hib)
    return equity_ma, equity_hibrida, trades_ma, trades_hibrida

def plot_comparison_equity_trades(df_signals, equity_ma, equity_hibrida, ticker):
    fig, ax = plt.subplots(figsize=(12, 5))
    points = np.linspace(0, len(df_signals.index), num=len(equity_ma))
    ax.plot(df_signals.index, np.interp(range(len(df_signals)), points, equity_ma), label="Média Móvel (Todos os trades)", lw=2, color="#efcb68")
    ax.plot(df_signals.index, np.interp(range(len(df_signals)), points, equity_hibrida), label="Média Móvel + IA (Apenas trades aprovados)", lw=2, color="#0070F3")
    ax.set_ylabel("Equity (Crescimento do capital)")
    ax.set_xlabel("Data")
    ax.set_title(f"Curva de Equity: {ticker}", fontsize=16, fontweight='bold', color='#222831')
    ax.legend(fontsize=13, loc='upper left')
    ax.grid(True, linestyle='--', alpha=0.7)
    st.pyplot(fig)

def show():
    st.title("Estratégia Quantitativa Híbrida — Trades Completos")
    st.write(
        "Agora a IA só valida operações completas: cada trade é uma sequência de compra (cruzamento de médias) até a próxima venda. "
        "A estratégia híbrida só executa trades que o modelo prevê como vencedores. Acompanhe o desempenho comparativo!"
    )

    if not all(k in st.session_state for k in ['tickers_alinhados', 'dados_alinhados', 'ordem_alinhada', 'dias_historico']):
        st.error("Primeiro, gere o alinhamento na aba de Exploração para continuar.")
        return

    aligned_order = st.session_state['ordem_alinhada']
    prices_matrix = st.session_state['dados_alinhados']

    st.info(
        f"Ações alinhadas por máxima correlação: {' → '.join(aligned_order)}"
    )

    ativos_default = aligned_order[:5]
    ativos_escolhidos = st.multiselect(
        "Selecione até 5 ações principais para simular a estratégia:",
        aligned_order,
        default=ativos_default,
        max_selections=5,
        help="Estas são as ações principais onde a estratégia será simulada. Altere à vontade!"
    )

    fast = st.slider("Período média móvel curta", 3, 15, 5)
    slow = st.slider("Período média móvel longa", 10, 40, 21)
    lookback = st.slider("Dias de histórico (features IA)", 3, 20, 7)
    threshold = st.slider("Threshold mínimo do modelo IA (probabilidade)", 0.5, 0.95, 0.6)

    for ticker in ativos_escolhidos:
        st.subheader(f"Resultados para: {ticker}")
        main_prices = prices_matrix[ticker]
        st.line_chart(main_prices, use_container_width=True)

        df_signals = compute_signals(main_prices, fast=fast, slow=slow)

        # Features para trades completos
        X, y, idxs = build_features_multivariate_trades(df_signals, prices_matrix[aligned_order], aligned_order, lookback=lookback)
        if len(X) < 5:
            st.warning(
                "Poucos trades completos para treinar o modelo. "
                "Tente aumentar o período de dados históricos ou ajustar os parâmetros para este ativo."
            )
            continue

        # Split: 60-20-20
        n = len(X)
        idx_train = int(n * 0.6)
        idx_val = int(n * 0.8)
        X_train, y_train = X[:idx_train], y[:idx_train]
        X_val, y_val = X[idx_train:idx_val], y[idx_train:idx_val]
        X_test, y_test = X[idx_val:], y[idx_val:]
        idxs_test = idxs[idx_val:]

        model = train_rf_classifier(X_train, y_train)

        st.markdown("#### Desempenho do modelo Random Forest (teste - trades futuros):")
        y_pred_test = model.predict(X_test)
        st.text(classification_report(y_test, y_pred_test))

        # Equity simulada (apenas nos trades de teste)
        equity_ma, equity_hibrida, trades_ma, trades_hibrida = simulate_strategy_trades(
            df_signals,
            main_prices,
            model=model,
            prices_matrix=prices_matrix,
            aligned_order=aligned_order,
            lookback=lookback,
            threshold=threshold,
            idxs_test=idxs_test,
            feats_test=X_test
        )
        plot_comparison_equity_trades(df_signals, equity_ma, equity_hibrida, ticker)

        st.markdown("#### Trades executados no período de teste")
        df_ma = pd.DataFrame(trades_ma, columns=["Entrada", "Saída", "Retorno (%)"])
        st.dataframe(df_ma, use_container_width=True)
        st.markdown("#### Trades aprovados pela IA no período de teste")
        df_rf = pd.DataFrame(trades_hibrida, columns=["Entrada", "Saída", "Retorno (%)", "Prob. modelo"])
        st.dataframe(df_rf, use_container_width=True)
        st.markdown("---")

if __name__ == "__main__":
    show()
