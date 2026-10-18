"""Entry point:  streamlit run app/main.py"""
import streamlit as st

st.set_page_config(page_title="Trợ lý Pháp luật đại cương", page_icon="⚖️", layout="wide")

pages = st.navigation([
    st.Page("chat_page.py", title="Hỏi đáp", icon="💬", default=True),
    st.Page("quiz_page.py", title="Luyện trắc nghiệm", icon="📝"),
])
pages.run()
