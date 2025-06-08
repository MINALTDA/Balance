def create_qr_code(link):
    import qrcode
    import streamlit as st
    import io

    img = qrcode.make(link)
    buf = io.BytesIO()
    img.save(buf)
    st.image(buf.getvalue(), caption="QR Code", use_column_width=True)

def add_vertical_space(lines=1):
    import streamlit as st
    for _ in range(lines):
        st.markdown("<br>", unsafe_allow_html=True)
