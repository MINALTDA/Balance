import streamlit as st
from PIL import Image

def show():
    st.markdown(
        "<h1 style='color:#0070F3; font-weight:800;'>Estratégia Macro Waves</h1>", 
        unsafe_allow_html=True
    )

    st.markdown(
        "<h3 style='color:#222831;'>Investimento baseado em ciência, não em achismo.</h3>", 
        unsafe_allow_html=True
    )

    st.markdown("---")

    # Filosofia central em destaque
    st.markdown(
        f"""
        <div style='background-color:#EFCB68; border-radius:12px; padding:22px 32px; margin-bottom: 20px;'>
        <b>Nossa Tese:</b> <br>
        O mercado não é totalmente eficiente no curto e médio prazo. Shocks macroeconômicos geram ondas previsíveis que percorrem setores da economia de forma quantificável.<br>
        Nosso objetivo não é prever o preço exato de um ativo, mas antecipar a <b>direção relativa de setores inteiros</b> diante de choques reais de mercado.
        </div>
        """,
        unsafe_allow_html=True
    )

    # Passos do método
    st.markdown("### Como funciona a estratégia?")
    st.info("""
    1️⃣ **Identificamos um choque macroeconômico** (taxa de juros, volatilidade, petróleo).  
    2️⃣ **Usamos modelos econométricos** (VAR) para quantificar como o choque se propaga entre setores.  
    3️⃣ **Geramos sinais claros de investimento**: long & short, sempre com embasamento estatístico, não em intuição.  
    4️⃣ **Acompanhamos e ajustamos posições** de acordo com a dinâmica prevista pela análise impulso-resposta.
    """)

    # Infográfico animado ou imagem ilustrativa
    st.markdown("### Onda de Impacto Macro")
    # Exemplo: carregue uma imagem do assets se já tiver
    try:
        img = Image.open("assets/onda_choque_macro.png")
        st.image(img, caption="Visualização da propagação de um choque econômico por diferentes setores.", use_column_width=True)
    except:
        st.info("Aqui você pode adicionar um infográfico animado mostrando como um choque macroeconômico afeta setores ao longo do tempo.")

    # Frase final de impacto
    st.markdown(
        "<div style='color:#0070F3; font-size:22px; margin-top:32px;'>"
        "🌊 Invista com lógica, robustez e transparência. Não siga a maré – antecipe a onda.</div>",
        unsafe_allow_html=True
    )
