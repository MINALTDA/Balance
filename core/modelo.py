import pandas as pd
from statsmodels.tsa.api import VAR
from statsmodels.tsa.stattools import grangercausalitytests

# def select_var_order(df, maxlags=8):
#     """
#     Seleciona o lag ótimo com base no critério AIC.
#     """
#     model = VAR(df)
#     order = model.select_order(maxlags)
#     ic_df = pd.DataFrame({
#         'AIC': order.aic,
#         'BIC': order.bic,
#         'FPE': order.fpe,
#         'HQIC': order.hqic
#     })
#     p_opt = ic_df.idxmin()['AIC']  # normalmente o AIC é usado
#     return int(p_opt), ic_df

# def select_var_order(df, maxlags=3):
#     """
#     Seleciona o lag ótimo baseado nos critérios de informação.
#     Suporta qualquer versão do statsmodels.
#     """
#     model = VAR(df)
#     order = model.select_order(maxlags)
    
#     # Função auxiliar para garantir que tudo é Series
#     def as_series(val, name):
#         if isinstance(val, (pd.Series, dict)):
#             return pd.Series(val)
#         else:
#             # É escalar: só 1 lag foi avaliado (ex: poucos dados)
#             return pd.Series({0: val})
    
#     aic = as_series(order.aic, "AIC")
#     bic = as_series(order.bic, "BIC")
#     fpe = as_series(order.fpe, "FPE")
#     hqic = as_series(order.hqic, "HQIC")
    
#     ic_df = pd.DataFrame({
#         'AIC': aic,
#         'BIC': bic,
#         'FPE': fpe,
#         'HQIC': hqic
#     })
#     # drop duplicates de índices se vierem, só para garantir
#     ic_df = ic_df[~ic_df.index.duplicated(keep='first')]
    
#     # O p ótimo é o lag com menor AIC
#     p_opt = ic_df['AIC'].idxmin()
#     return int(p_opt), ic_df






def select_var_order(df, maxlags=5, min_lag=1):
    """
    Seleciona o lag ótimo com base no critério de informação (AIC), 
    mas impondo lag mínimo (default = 1).
    """
    model = VAR(df)
    order = model.select_order(maxlags)
    
    def as_series(val, name):
        if isinstance(val, (pd.Series, dict)):
            return pd.Series(val)
        else:
            return pd.Series({0: val})
    
    aic = as_series(order.aic, "AIC")
    bic = as_series(order.bic, "BIC")
    fpe = as_series(order.fpe, "FPE")
    hqic = as_series(order.hqic, "HQIC")
    
    ic_df = pd.DataFrame({
        'AIC': aic,
        'BIC': bic,
        'FPE': fpe,
        'HQIC': hqic
    })
    ic_df = ic_df[~ic_df.index.duplicated(keep='first')]
    
    # Aqui impomos lag mínimo
    ic_df = ic_df[ic_df.index >= min_lag]
    
    # Se ficou vazio (exemplo: só havia p=0), então força p=1
    if ic_df.empty:
        p_opt = 1
    else:
        p_opt = ic_df['AIC'].idxmin()
    return int(p_opt), ic_df






def fit_var(df, p):
    """
    Ajusta o modelo VAR(p).
    """
    model = VAR(df)
    results = model.fit(p)
    return results

def granger_matrix(df, maxlag):
    """
    Matriz de causalidade de Granger: valor-p para cada par.
    """
    result = pd.DataFrame(index=df.columns, columns=df.columns)
    for caused in df.columns:
        for causing in df.columns:
            if caused != causing:
                test = grangercausalitytests(df[[caused, causing]], maxlag=maxlag, verbose=False)
                pvals = [test[i + 1][0]['ssr_ftest'][1] for i in range(maxlag)]
                result.loc[caused, causing] = round(min(pvals), 4)
            else:
                result.loc[caused, causing] = '-'
    return result

def get_irf(var_model, impulse, response, periods=8):
    """
    Calcula a função impulso-resposta.
    """
    irf = var_model.irf(periods)
    irf_values = irf.orth_irfs[:, var_model.names.index(response), var_model.names.index(impulse)]
    index = [f"T+{i}" for i in range(periods + 1)]
    return pd.DataFrame({'Resposta': irf_values}, index=index)
