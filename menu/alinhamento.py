import streamlit as st
import pandas as pd

from menu.funcoes_alinhamento import (
    carregar_e_limpar_r3,
    carregar_e_limpar_r3_consolidado,
    carregar_vitals,
    ajuste_tempo_zero,
    alinhar_sensores_heuristico,
    visualizar_ajuste_tempo_zero_interativo,
    visualizar_alinhamento,
    aplicar_desfase,
)

def render():
    st.markdown("# 🔍 Alinhamento de sensores")
    try:
        # df_r3 = carregar_e_limpar_r3("data/R3.csv")
        df_r3 = carregar_e_limpar_r3_consolidado("data/R_alinhados.csv")
        df_vitals = carregar_vitals("data/vitals.csv")
        st.subheader("Informações dos dados")
        col1, col2 = st.columns(2)
        with col1:
            st.write("Sensor R3 (após limpeza):")
            st.write(f"- Registros: {len(df_r3)}")
            st.write(f"- Período: {df_r3['Epoch'].min()} a {df_r3['Epoch'].max()} ms")
        with col2:
            st.write("Sensor vitals:")
            st.write(f"- Registros: {len(df_vitals)}")
            st.write(f"- Período: {df_vitals['TimeStamp'].min()} a {df_vitals['TimeStamp'].max()} ms")
        df_r3_ajustado, df_vitals_ajustado = ajuste_tempo_zero(df_r3, df_vitals)
        st.subheader("Visualização após ajuste de tempo zero")
        fig = visualizar_ajuste_tempo_zero_interativo(df_r3_ajustado, df_vitals_ajustado)
        st.plotly_chart(fig, use_container_width=True)
        st.subheader("Parâmetros para o alinhamento heurístico")
        col1, col2 = st.columns(2)
        with col1:
            num_pontos_aleatorios = st.slider("Número de pontos aleatórios", 2, 50, 2)
            max_desvio = st.slider("Desvio máximo para valores similares (bpm)", 0, 5, 0)
        with col2:
            max_diff_tempo = st.slider("Diferença máxima de tempo (ms)", 100, 1000, 500, 50)
            limiar_penalizacao = st.slider("Limiar de penalização (bpm)", 1, 10, 3)
        min_comparacoes_percentual = st.slider("Percentual mínimo de comparações", 0.05, 0.3, 0.05, 0.01)
        df_vitals_ajustado_subconjunto = df_vitals_ajustado.sample(frac=min_comparacoes_percentual*3).sort_index().reset_index(drop=False)
        if st.button("Executar alinhamento heurístico"):
            with st.spinner("Calculando o alinhamento..."):
                resultado = alinhar_sensores_heuristico(
                    df_r3_ajustado,
                    df_vitals_ajustado_subconjunto,
                    num_pontos_aleatorios=num_pontos_aleatorios,
                    max_desvio=max_desvio,
                    max_diff_tempo=max_diff_tempo,
                    limiar_penalizacao=limiar_penalizacao,
                    min_comparacoes_percentual=min_comparacoes_percentual
                )
                if resultado:
                    st.subheader("Resultados do alinhamento")
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Desfase (segundos)", f"{resultado['desfase']/1000:.3f}")
                    with col2:
                        st.metric("Média de erros (bpm)", f"{resultado['media_erros']:.2f}")
                    with col3:
                        st.metric("Número de penalizações", resultado['num_penalizacoes'])
                    st.metric("Número de Comparações", resultado['num_comparacoes'])
                    st.metric("Score de alinhamento", f"{resultado['metrica']:.4f}")
                    st.subheader("Visualização do alinhamento aprimorado")
                    fig_alinhado = visualizar_alinhamento(df_r3_ajustado, df_vitals_ajustado, resultado['desfase'])
                    st.plotly_chart(fig_alinhado, use_container_width=True)
                    df_r3_alinhado = aplicar_desfase(df_r3_ajustado, resultado['desfase'])
                    st.subheader("Download dos dados alinhados")
                    @st.cache_data
                    def convert_df_to_csv(df):
                        return df.to_csv(index=False).encode('utf-8')                 

                    # 1. Tiempo inicial alineado:
                    tiempo_inicial_vitals_alinhado = df_r3['Epoch'].iloc[0] - resultado['desfase']

                    # 2. Calcula el delta de tiempo entre registros de vitals
                    delta_vitals = df_vitals['TimeStamp'].values - df_vitals['TimeStamp'].values[0]

                    # 3. Nuevo tiempo alineado:
                    nuevo_tiempo_vitals = tiempo_inicial_vitals_alinhado + delta_vitals

                    # 4. Crea una copia para no modificar df_vitals original
                    df_vitals_alinhado = df_vitals.copy()
                    df_vitals_alinhado['TimeStamp (mS)'] = nuevo_tiempo_vitals

                    
                    # Elimina columna auxiliar si existe
                    if 'tempo_relativo' in df_vitals_alinhado.columns:
                        df_vitals_alinhado = df_vitals_alinhado.drop(columns=['tempo_relativo'])
                    if 'TimeStamp' in df_vitals_alinhado.columns:
                        df_vitals_alinhado = df_vitals_alinhado.drop(columns=['TimeStamp'])
                    if 'HeartRate' in df_vitals_alinhado.columns:
                        df_vitals_alinhado = df_vitals_alinhado.drop(columns=['HeartRate'])

                    # 5. Descarga todas las columnas
                    csv_vitals = convert_df_to_csv(df_vitals_alinhado)

                    st.download_button(
                        "Download dados Vitals alinhados",
                        csv_vitals,
                        "vitals_alinhado.csv",
                        "text/csv",
                        key='download-vitals-alinhado-todos'
                    )

                else:
                    st.error("Não foi possível encontrar um alinhamento adequado com os parâmetros fornecidos. Tente ajustar os parâmetros.")
    except FileNotFoundError:
        st.error("Erro ao carregar arquivos. Verifique se os CSVs estão na pasta `data/`.")
    except Exception as e:
        st.error(f"Ocorreu um erro: {str(e)}")
        st.exception(e)

if __name__ == "__main__":
    render()
