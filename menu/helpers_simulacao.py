import yfinance as yf
import pandas as pd
import numpy as np
from itertools import combinations

def load_prices(tickers):
    df = yf.download(tickers, start="2018-01-01", progress=False)["Close"]
    if isinstance(df, pd.Series):
        df = df.to_frame()
    return df

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

def calcular_top5(prices, top_tickers, lookback, z_entry, z_exit, commission, max_lag):
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
                "lag_exp": lag_exp,
                "sharpe_exp": res_exp["sharpe"],
                "crescimento_exp": res_exp["growth"],
                "sharpe_trad": res_trad["sharpe"],
                "crescimento_trad": res_trad["growth"],
                "df_lag": best_result["df"],
                "df_trad": res_trad["df"],
                "df_exp": res_exp["df"]
            })
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values(by="sharpe_lag", ascending=False)
    results_df["selector"] = results_df["par"] + " (lag=" + results_df["lag"].astype(str) + ")"
    return results_df
