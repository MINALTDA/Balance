# import streamlit as st
# from streamlit_option_menu import option_menu

# # Paleta personalizada
# PRIMARY_COLOR = "#0070F3"
# ACCENT_COLOR = "#EFCB68"
# DARK_COLOR = "#222831"
# SUCCESS_COLOR = "#33D69F"
# BG_COLOR = "#F7F7F9"

# # Configuración general
# st.set_page_config(
#     page_title="VAR Macro Waves – Dashboard de Estrategia Cuantitativa",
#     page_icon="💹",
#     layout="wide",
#     initial_sidebar_state="expanded",
#     menu_items={
#         'Get Help': 'mailto:tu-email@ejemplo.com',
#         'Report a bug': "mailto:tu-email@ejemplo.com"
#     }
# )

# # CSS personalizado para el estilo diferenciado
# st.markdown(f"""
#     <style>
#     body, .stApp {{
#         background-color: {BG_COLOR} !important;
#         color: {DARK_COLOR};
#         font-family: 'Montserrat', 'Lato', 'Roboto', sans-serif;
#     }}
#     .sidebar .sidebar-content {{
#         background: {BG_COLOR} !important;
#     }}
#     .stButton>button {{
#         background-color: {PRIMARY_COLOR};
#         color: white;
#         border-radius: 8px;
#         font-size: 18px;
#         font-weight: 600;
#     }}
#     .stDataFrame {{
#         border-radius: 16px;
#         box-shadow: 0 2px 8px rgba(34,40,49,0.06);
#     }}
#     .block-container {{
#         padding-top: 2rem;
#     }}
#     .css-1aumxhk {{
#         background-color: {ACCENT_COLOR} !important;
#     }}
#     .stTabs [data-baseweb="tab"] {{
#         font-size:18px;
#         padding: 8px 24px;
#     }}
#     </style>
#     """, unsafe_allow_html=True
# )

# # Menú lateral elegante
# with st.sidebar:
#     st.image("assets/logo.png", width=200)  # Si tienes un logo, ponlo aquí
#     menu = option_menu(
#         "Navegación",
#         ["Estrategia", "Universo", "Modelo", "Señales", "Resultados"],
#         icons=["lightbulb", "layers", "cpu", "activity", "bar-chart-line"],
#         menu_icon="cast",
#         default_index=0,
#         styles={
#             "container": {"padding": "0!important", "background-color": BG_COLOR},
#             "icon": {"color": PRIMARY_COLOR, "font-size": "20px"},
#             "nav-link": {"font-size": "18px", "text-align": "left", "margin":"0px", "--hover-color": ACCENT_COLOR},
#             "nav-link-selected": {"background-color": PRIMARY_COLOR, "color": "white"},
#         }
#     )

# # Renderizar la página seleccionada
# if menu == "Estrategia":
#     import menu.estrategia as page
#     page.show()
# elif menu == "Universo":
#     import menu.universo as page
#     page.show()
# elif menu == "Modelo":
#     import menu.modelo as page
#     page.show()
# elif menu == "Señales":
#     import menu.senales as page
#     page.show()
# elif menu == "Resultados":
#     import menu.resultados as page
#     page.show()




import streamlit as st
from streamlit_option_menu import option_menu

# Paleta personalizada
PRIMARY_COLOR = "#0070F3"
ACCENT_COLOR = "#EFCB68"
DARK_COLOR = "#222831"
SUCCESS_COLOR = "#33D69F"
BG_COLOR = "#F7F7F9"

# Configuração geral
st.set_page_config(
    page_title="VAR Macro Waves – Dashboard Quantitativo",
    page_icon="💹",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'mailto:suporte@exemplo.com',
        'Report a bug': "mailto:suporte@exemplo.com"
    }
)

# CSS personalizado para o visual diferenciado
st.markdown(f"""
    <style>
    body, .stApp {{
        background-color: {BG_COLOR} !important;
        color: {DARK_COLOR};
        font-family: 'Montserrat', 'Lato', 'Roboto', sans-serif;
    }}
    .sidebar .sidebar-content {{
        background: {BG_COLOR} !important;
    }}
    .stButton>button {{
        background-color: {PRIMARY_COLOR};
        color: white;
        border-radius: 8px;
        font-size: 18px;
        font-weight: 600;
    }}
    .stDataFrame {{
        border-radius: 16px;
        box-shadow: 0 2px 8px rgba(34,40,49,0.06);
    }}
    .block-container {{
        padding-top: 2rem;
    }}
    .css-1aumxhk {{
        background-color: {ACCENT_COLOR} !important;
    }}
    .stTabs [data-baseweb="tab"] {{
        font-size:18px;
        padding: 8px 24px;
    }}
    /* Remove o título "Navegação" do menu lateral */
    .css-1v0mbdj > div:first-child {{
        display: none;
    }}
    </style>
    """, unsafe_allow_html=True
)

# Menu lateral sem o título "Navegação"
with st.sidebar:
    st.image("assets/logo.png", width=200)  # Coloque o seu logo se desejar
    menu = option_menu(
        None,  # Remove o título!
        # ["Estratégia", "Universo", "Modelo", "Sinais", "Resultados"],
        # icons=["lightbulb", "layers", "cpu", "activity", "bar-chart-line"],
        ["Estratégia", "Universo", "Modelo", "Señales", "Resultados", "Simulação"],
        icons=["lightbulb", "layers", "cpu", "activity", "bar-chart-line", "repeat"],
        menu_icon=None,
        default_index=0,
        styles={
            "container": {"padding": "0!important", "background-color": BG_COLOR},
            "icon": {"color": PRIMARY_COLOR, "font-size": "20px"},
            "nav-link": {"font-size": "18px", "text-align": "left", "margin":"0px", "--hover-color": ACCENT_COLOR},
            "nav-link-selected": {"background-color": PRIMARY_COLOR, "color": "white"},
        }
    )

# Renderizar a página selecionada
if menu == "Estratégia":
    import menu.estrategia as page
    page.show()
elif menu == "Universo":
    import menu.universo as page
    page.show()
elif menu == "Modelo":
    import menu.modelo as page
    page.show()
elif menu == "Sinais":
    import menu.sinais as page
    page.show()
elif menu == "Resultados":
    import menu.resultados as page
    page.show()
elif menu == "Simulação":
    import menu.simulacao as page
    page.show()
