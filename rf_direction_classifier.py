# rf_direction_classifier.py
import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score, roc_curve
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="Random Forest Dirección (Sube/Baja)", layout="wide")
st.title("🔺 Random Forest para Predicción de Dirección (Sube/Baja)")

# 1. Configuración de tickers y parámetros
st.sidebar.header("Configuración")
default_tickers = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "META", "NVDA"]
tickers = st.sidebar.multiselect("Selecciona los tickers a usar como predictores", default_tickers, default=default_tickers)
target = st.sidebar.selectbox("Selecciona el ticker objetivo", tickers, index=3)  # Ej: AMZN
max_lag = st.sidebar.slider("Máximo lag por ticker", 1, 20, 10)
period_years = st.sidebar.slider("Años de histórico", 1, 10, 3)
n_estimators = st.sidebar.slider("N° de árboles en el Random Forest", 10, 200, 100, step=10)
test_size = st.sidebar.slider("Proporción de datos para test", 0.1, 0.5, 0.3)

st.write(f"**Objetivo:** predecir si {target} sube o baja usando {', '.join(tickers)} y hasta {max_lag} lags por acción.")

# 2. Descarga y preparación de datos
@st.cache_data(show_spinner=False)
def load_data(tickers, years):
    end = datetime.today()
    start = end - timedelta(days=365*years)
    df = yf.download(tickers, start=start, end=end, progress=False)["Close"]
    if isinstance(df, pd.Series):
        df = df.to_frame()
    return df

df = load_data(tickers, period_years)
returns = np.log(df / df.shift(1)).dropna()

# 3. Features y objetivo binario (sube=1, baja=0)
features = []
for ticker in tickers:
    for lag in range(1, max_lag+1):
        col = f"{ticker}_lag{lag}"
        returns[col] = returns[ticker].shift(lag)
        features.append(col)
returns["target_up"] = (returns[target] > 0).astype(int)
model_data = returns[features + ["target_up"]].dropna()

# 4. Split train/test
split_idx = int((1 - test_size) * len(model_data))
X_train = model_data[features].iloc[:split_idx].values
X_test = model_data[features].iloc[split_idx:].values
y_train = model_data["target_up"].iloc[:split_idx].values
y_test = model_data["target_up"].iloc[split_idx:].values

# 5. Entrenamiento y predicción
rf = RandomForestClassifier(n_estimators=n_estimators, max_depth=7, random_state=42, n_jobs=-1)
rf.fit(X_train, y_train)
y_pred = rf.predict(X_test)
y_proba = rf.predict_proba(X_test)[:, 1]

# 6. Métricas
st.subheader("Métricas de clasificación")
acc = accuracy_score(y_test, y_pred)
auc = roc_auc_score(y_test, y_proba)
st.write(f"**Accuracy:** {acc:.4f}")
st.write(f"**AUC (ROC):** {auc:.4f}")
st.text("Reporte de clasificación:\n" + classification_report(y_test, y_pred))

# 7. Matriz de confusión
st.subheader("Matriz de confusión")
fig, ax = plt.subplots()
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Blues', cbar=False,
            xticklabels=['Baja', 'Sube'], yticklabels=['Baja', 'Sube'], ax=ax)
ax.set_xlabel("Predicho")
ax.set_ylabel("Real")
st.pyplot(fig)

# 8. Curva ROC
st.subheader("Curva ROC")
fpr, tpr, _ = roc_curve(y_test, y_proba)
fig2, ax2 = plt.subplots()
ax2.plot(fpr, tpr, label=f"AUC = {auc:.2f}")
ax2.plot([0, 1], [0, 1], 'k--')
ax2.set_xlabel("False Positive Rate")
ax2.set_ylabel("True Positive Rate")
ax2.set_title("ROC Curve")
ax2.legend()
st.pyplot(fig2)

# 9. Importancia de predictores
importancias = pd.Series(rf.feature_importances_, index=features)
st.subheader("Top 10 predictores más importantes según Random Forest")
st.dataframe(importancias.abs().sort_values(ascending=False).head(10))

st.caption("Si el accuracy supera el 51%-55% y se mantiene estable en diferentes periodos, es potencialmente útil para señales de trading.")

