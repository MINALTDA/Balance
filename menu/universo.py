import streamlit as st
import pandas as pd
from PIL import Image

# Dados do universo (exemplo)
UNIVERSO = [
    {
        "nome": "Títulos do Tesouro Americano (20+ anos)",
        "ticker": "TLT",
        "tipo": "Impulso",
        "descricao": "Indicador principal do custo do dinheiro e apetite ao risco no mercado global.",
        "cor": "#0070F3"
    },
    {
        "nome": "Índice de Volatilidade (VIX)",
        "ticker": "VIX",
        "tipo": "Impulso",
        "descricao": "Reflete o 'preço do medo' – volatilidade e incerteza do mercado.",
        "cor": "#33D69F"
    },
    {
        "nome": "Petróleo (USO)",
        "ticker": "USO",
        "tipo": "Impulso",
        "descricao": "Indicador global de custos energéticos e pressão inflacionária.",
        "cor": "#EFCB68"
    },
    {
        "nome": "Setor de Tecnologia (QQQ)",
        "ticker": "QQQ",
        "tipo": "Resposta",
        "descricao": "Sensível a juros e sentimento de risco – representa o crescimento.",
        "cor": "#7B61FF"
    },
    {
        "nome": "Setor Financeiro (XLF)",
        "ticker": "XLF",
        "tipo": "Resposta",
        "descricao": "Reflete a saúde dos bancos e do crédito na economia americana.",
        "cor": "#FF5964"
    },
    {
        "nome": "Consumo Discricionário (XLY)",
        "ticker": "XLY",
        "tipo": "Resposta",
        "descricao": "Indica confiança e poder de compra do consumidor.",
        "cor": "#F7B32B"
    }
]

def show():
    st.markdown("<h1 style='color:#0070F3; font-weight:800;'>Universo de Ativos</h1>", unsafe_allow_html=True)
    st.markdown(
        "A estratégia Macro Waves atua sobre **setores inteiros da economia** representados por ETFs. "
        "Esses ativos traduzem grandes movimentos de capital, filtrando o 'ruído' de empresas individuais."
    )
    st.markdown("---")

    # Cards dos ativos
    st.markdown("### Ativos Selecionados")
    colunas = st.columns(3)
    for idx, ativo in enumerate(UNIVERSO):
        with colunas[idx % 3]:
            st.markdown(
                f"""
                <div style='background-color:{ativo['cor']}22; border-radius:10px; padding:16px; margin-bottom:10px;'>
                    <span style='font-size:22px; font-weight:bold; color:{ativo['cor']}'>{ativo['ticker']}</span><br>
                    <span style='font-size:16px; font-weight:600;'>{ativo['nome']}</span><br>
                    <span style='background-color:{"#FFF" if ativo["tipo"] == "Impulso" else "#222831"}; 
                                 color:{"#0070F3" if ativo["tipo"] == "Impulso" else "#FFF"}; 
                                 padding:3px 12px; border-radius:6px; font-size:12px; font-weight:700;'>{ativo['tipo']}</span>
                    <div style='margin-top:10px; font-size:14px;'>{ativo['descricao']}</div>
                </div>
                """,
                unsafe_allow_html=True
            )

    st.markdown("---")

    # Diagrama visual
    st.markdown("### Relação entre os Setores")
    try:
        img = Image.open("assets/relacoes_setores.png")
        st.image(img, caption="Fluxo macroeconômico entre setores: como um choque em um pilar impacta os demais.", use_column_width=True)
    except:
        st.info("Adicione aqui um diagrama simples mostrando a conexão entre os setores (Ex: TLT → QQQ, USO → XLY, VIX → QQQ etc). Dica: crie facilmente no Canva, Figma ou PowerPoint.")

    # Frase final
    st.markdown(
        "<div style='color:#222831; font-size:20px; margin-top:28px;'>"
        "A força da estratégia está na <b>diversificação inteligente</b> e na <b>ciência do fluxo de capitais</b>."
        "</div>", unsafe_allow_html=True
    )
