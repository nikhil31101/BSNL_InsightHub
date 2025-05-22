import os
import streamlit as st
import requests
from dotenv import load_dotenv
import base64

# Load environment variables
load_dotenv()

# Constants
BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000")

def set_bg(png_file):
    with open(png_file, "rb") as file:
        encoded = base64.b64encode(file.read()).decode()
    css = f"""
    <style>
    .stApp {{
        background-image: url("data:image/png;base64,{encoded}");
        background-size: cover;
        background-position: center;
        background-repeat: no-repeat;
    }}
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)

set_bg("BSNL_background.png")

def check_backend_status():
    try:
        response = requests.get(f"{BACKEND_URL}/status")
        return response.json()["initialized"]
    except:
        return False

def query_backend(question, include_sources=False):
    try:
        response = requests.post(
            f"{BACKEND_URL}/query",
            data={"question": question, "include_sources": include_sources}
        )
        return response.json()
    except Exception as e:
        return {"status": "error", "message": str(e)}

# Streamlit UI
st.set_page_config(page_title="iBSNL App", layout="centered", page_icon="bsnl_logo.png")

st.title("iBSNL Customer Support Assistant")
st.markdown("""
<style>
    .stTextInput>div>div>input {
        background-color: rgba(255, 255, 255, 0.9);
        border-radius: 10px;
        padding: 10px;
    }
    .stButton>button {
        width: 100%;
        border-radius: 10px;
        padding: 10px;
        background-color: #FF4B4B;
        color: white;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if "initialized" not in st.session_state:
    st.session_state.initialized = check_backend_status()

if not st.session_state.initialized:
    st.error("The AI engine is still initializing. Please wait...")
    st.spinner("Connecting to backend services...")
else:
    user_question = st.text_input("🔍 Ask your question:", placeholder="Type your question about BSNL services here...")

    if st.button("Generate Response") and user_question.strip():
        with st.spinner("Thinking..."):
            response = query_backend(user_question)
            
            if response.get("status") == "success":
                st.success("Response:")
                st.markdown(response["answer"])
                
                if st.checkbox("Show sources"):
                    for source in response.get("sources", []):
                        st.text(source)
            else:
                st.error(f"Error: {response.get('message', 'Unknown error occurred')}")

    st.sidebar.header("About")
    st.sidebar.info("""
    This is the official BSNL customer support assistant. 
    It can help you with:
    - Broadband services
    - Mobile services
    - New connections
    - Complaint registration
    - Plans and tariffs
    """)

    st.sidebar.header("Quick Links")
    st.sidebar.markdown("""
    - [Broadband Recharge](https://bsnl.co.in/broadband/FTTHbillpay)
    - [Complaint Registration](https://bsnl.co.in/support/contact)
    - [New Connection](https://bsnl.co.in/broadband/bharatfiber)
    - [Plans and Tariffs](https://bsnl.co.in/mobile/prepaid)
    - [Mobile Services](https://bsnl.co.in/mobile/porttobsnl)
    """)

if __name__ == "__main__":
    st.write("iBSNL App is running!")