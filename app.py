import streamlit as st

# Set up Title
st.set_page_config(page_title="WYN AI", page_icon=":robot_face:")
st.markdown(
    f"""
        <h1 style='text-align: center;'>W.Y.N. Artificial Intelligence😬</h1>
    """,
    unsafe_allow_html=True,
)

## Set up Sidebar
st.sidebar.title("Sidebar")
stocks = st.sidebar.text_input('Enter stocks (sep. by comma)', 'AAPL, MSFT, NVDA, TSLA')

