import streamlit as st

def render(create_qr_code=None, add_vertical_space=None):
    # Cabeçalho inicial
    st.header("Otimização do Uso de Sensores Fisiológicos em Cenários com Recursos Limitados")
    st.markdown("*Avaliação comparativa de sensores tipo R na predição do MAP medido por sensor Vitals*")

    # Criando duas colunas lado a lado
    col1, col2 = st.columns(2)

    # Primeira coluna: Contexto, Problema e Objetivo
    with col1:
        st.markdown("### 🎯 Contexto")
        st.markdown("""
        - Devido a cortes previstos no programa de pesquisa, será necessário otimizar o uso dos sensores tipo **R**.  
        - Atualmente usamos **3 sensores R** e **1 sensor Vital**, que fornece o **MAP (Pressão Arterial Média)** como referência.
        """)

        st.markdown("### ❗ Problema")
        st.markdown("""
        - Com menos recursos disponíveis, precisamos reduzir o número de sensores R sem comprometer a qualidade dos dados fisiológicos coletados.
        """)

        st.markdown("### 🧠 Objetivo")
        st.markdown("""
        - Avaliar qual dos sensores R, em sua posição específica, fornece a melhor estimativa do **MAP** medido pelo sensor Vital.
        """)

    # Segunda coluna: Abordagem e Benefícios
    with col2:
        st.markdown("### 🛠️ Abordagem")
        st.markdown("""
        - Usar dados fisiológicos já coletados dos sensores R e do sensor Vital.  
        - Treinar modelos de predição para estimar o MAP a partir de cada sensor R.  
        - Comparar o desempenho dos sensores.  
        - Indicar o sensor R mais eficiente para uso isolado.
        """)

        st.markdown("### ✅ Benefícios esperados")
        st.markdown("""
        - Redução de custos operacionais.  
        - Manutenção da qualidade nas medições fisiológicas.  
        - Base científica para decisões em cenários com restrição de recursos.
        """)

    # Espaço opcional
    if add_vertical_space:
        add_vertical_space(2)

    # Exibição do QR code opcional (com verificação de None)
    # if create_qr_code:
    #     st.markdown("### 🔗 Acesso rápido ao projeto")
    #     qr_img = create_qr_code("https://seu-link-do-projeto.com")
    #     if qr_img is not None:
    #         st.image(qr_img, caption="Acesse o painel completo", width=8) 
    #     else:
    #         st.warning("QR code não pôde ser gerado.")
