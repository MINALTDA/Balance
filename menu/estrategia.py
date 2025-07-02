import streamlit as st

def show():
    st.markdown(
        """
        <style>
        .big-title {
            font-size: 2.7em !important;
            color: #0070F3 !important;
            font-weight: 900 !important;
            letter-spacing: -1px;
            margin-bottom: 0.1em;
        }
        .subtitle {
            font-size: 1.45em !important;
            color: #222831 !important;
            font-weight: 600 !important;
            margin-bottom: 1em;
        }
        .highlight-box {
            background: #f0f4ff;
            border-left: 8px solid #0070F3;
            padding: 1.3em 1.7em 1.3em 1.7em;
            margin-top: 1.2em;
            margin-bottom: 2em;
            border-radius: 16px;
            font-size: 1.17em;
            font-weight: 500;
        }
        .destaque {
            color: #0070F3;
            font-weight: bold;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    st.markdown(
        "<div class='big-title'>Revolucione sua Gestão de Investimentos com Inteligência Quantitativa</div>",
        unsafe_allow_html=True
    )

    st.image("assets/banner.jpg", use_column_width=True, caption="")

    st.markdown(
        "<div class='subtitle'>Gestão de risco, performance e inovação em um só ambiente. Plataforma pensada para investidores exigentes que não aceitam resultados medianos.</div>",
        unsafe_allow_html=True
    )    

    st.markdown(
        """
        <div class="highlight-box">
            <span class="destaque">Você está pronto para operar no próximo nível?</span>
            <br><br>
            <ul>
            <li>Infraestrutura robusta e segura, desenvolvida por especialistas em algoritmos e mercados financeiros.</li>
            <li>Estratégias baseadas em evidências, com backtests auditáveis e métricas avançadas de risco-retorno.</li>
            <li>Monitoramento em tempo real e ferramentas para análise profunda de pares, sinais e desempenho.</li>
            <li>Gestão integrada com tecnologia de alinhamento temporal (<b>lead-lag</b>) e simulação de cenários.</li>
            <li>Transparência, governança e suporte dedicado para clientes institucionais, family offices e traders profissionais.</li>
            </ul>
            <span class="destaque">Maximize sua vantagem. Invista com quem transforma dados em performance.</span>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown(
        """
        ### Por que escolher nossa plataforma?
        - **Metodologia científica, auditável e customizável.**
        - **Equipe multidisciplinar com trajetoria comprovada.**
        - **Relatórios visuais e insights acionáveis em tempo real.**

        ---
        """,
        unsafe_allow_html=True
    )

    col1, col2, col3 = st.columns([1,2,1])
    with col2:
        st.markdown(
            """
            <div style='text-align:center;margin-top:2.2em;'>
                <a href="https://www.minavital.com" target="_blank" style="
                    display:inline-block;
                    background:#0070F3;
                    color:#fff;
                    font-weight:700;
                    font-size:1.20em;
                    padding:0.85em 2.4em;
                    border-radius:14px;
                    text-decoration:none;
                    box-shadow:0 4px 18px rgba(0,112,243,0.07);
                    margin-bottom:1em;
                    transition:background 0.2s;
                " onmouseover="this.style.background='#005bb5'" onmouseout="this.style.background='#0070F3'">
                    🚀 Agende uma demonstração exclusiva
                </a>
                <br>
                <span style='font-size:1.05em;color:#222831;'>
                    Conheça pessoalmente como nossa tecnologia pode alavancar seus resultados.<br>
                    <b>Seu próximo salto de performance começa aqui.</b>
                </span>
            </div>
            """,
            unsafe_allow_html=True
        )
