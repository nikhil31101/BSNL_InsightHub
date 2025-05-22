import os
import json
import streamlit as st
from langchain_aws import BedrockLLM
from langchain_aws.embeddings import BedrockEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
import pdfplumber
import base64
import boto3
import tempfile
import pandas as pd
from langchain.schema import Document
from botocore.exceptions import ClientError, BotoCoreError
from langchain.text_splitter import RecursiveCharacterTextSplitter
from concurrent.futures import ThreadPoolExecutor
import shutil

# Load environment variables
load_dotenv()

# Constants
MAX_INPUT_CHARS = 49000
CHUNK_SIZE = 1500
CHUNK_OVERLAP = 150
S3_BUCKET = "bsnlinsighthub"
FAISS_INDEX_KEY = "faiss_index"

# Prompt Template
custom_prompt = """You are a helpful AI assistant for BSNL Customers.

Answer the following question using the information from the provided documents. Be polite and concise.

give the below information in the answer only if asked for:

- Broadband Recharge: https://bsnl.co.in/broadband/FTTHbillpay

- Complaint Registration: https://bsnl.co.in/support/contact

- New Connection: https://bsnl.co.in/broadband/bharatfiber

- Plans and Tariffs: https://bsnl.co.in/mobile/prepaid

- Mobile Services: https://bsnl.co.in/mobile/porttobsnl


Question: {question}

Answer:"""

# Streamlit Config - minimal layout
st.set_page_config(
    page_title="iBSNL App",
    layout="centered",
    page_icon="bsnl_logo.png",
    initial_sidebar_state="collapsed"  # Hide sidebar by default
)

def set_bg(png_file):
    try:
        with open(png_file, "rb") as f:
            encoded = base64.b64encode(f.read()).decode()
        
        css = f"""
        <style>
        /* Full bleed background with no cropping */
        .stApp {{
            background-image: url("data:image/png;base64,{encoded}");
            background-size: cover;
            background-position: top center;
            background-repeat: no-repeat;
            background-attachment: fixed;
            padding: 0 !important;
            margin: 0 !important;
        }}
        
        /* Remove all containers' padding */
        .main .block-container, .st-emotion-cache-1y4p8pa, .st-emotion-cache-1v0mbdj {{
            padding: 0 !important;
            max-width: 100% !important;
        }}
        
        /* Remove default Streamlit spacing */
        .st-emotion-cache-1r4qj8v {{
            padding-top: 0 !important;
        }}
        
        /* Header adjustments */
        header {{
            background-color: transparent !important;
        }}
        
        /* Full width content */
        .st-emotion-cache-1dj0hjr {{
            width: 95% !important;
            padding: 0 5px !important;
        }}
        </style>
        """
        st.markdown(css, unsafe_allow_html=True)
    except Exception as e:
        st.error(f"Background error: {str(e)}")

# Set background - ensure image path is correct
set_bg("BSNL_background1.png")

# Your content - now flush with edges
st.markdown(
    """
    <div style='padding: 9px;'>
            </div>
    """,
    unsafe_allow_html=True
)


def process_pdf_page(page, pdf_key, page_num):
    """Process a single PDF page in parallel"""
    text = page.extract_text() or ""
    tables = []
    for table in page.extract_tables():
        try:
            df = pd.DataFrame(table[1:], columns=table[0])
            tables.append(df.to_string(index=False))
        except Exception:
            continue
    return text, tables, f"{pdf_key}-Page{page_num}"

def load_or_create_index(embeddings):
    """Load cached FAISS index or create new"""
    s3 = boto3.client('s3')
    temp_dir = tempfile.mkdtemp()
    
    try:
        s3.download_file(S3_BUCKET, f"{FAISS_INDEX_KEY}.index", f"{temp_dir}/index.faiss")
        s3.download_file(S3_BUCKET, f"{FAISS_INDEX_KEY}.pkl", f"{temp_dir}/index.pkl")
        faiss_store = FAISS.load_local(temp_dir, embeddings)
        st.sidebar.success("Loaded cached embeddings")
        return faiss_store
    except Exception as e:
        st.sidebar.warning(f"Building new embeddings: {str(e)}")
        return None

@st.cache_resource(ttl="24h")
def initialize_rag():
    # Initialize clients
    s3 = boto3.client("s3")
    bedrock = boto3.client("bedrock-runtime")

    # Try loading cached index
    embeddings = BedrockEmbeddings(client=bedrock, model_id="amazon.titan-embed-text-v2:0")
    faiss_store = load_or_create_index(embeddings)

    if faiss_store is None:
        all_chunks = []
        pdf_files = ["BSNL_CDA.pdf"]
        json_files = [
            "bbc_data.json",
            "EB_NAM_details_with_metadata.json",
            "Franchise_managers_data_with_metadata.json"
        ]

        # Process PDFs
        """for pdf_key in pdf_files:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_pdf:
                s3.download_fileobj(S3_BUCKET, pdf_key, temp_pdf)
                temp_pdf_path = temp_pdf.name

            with pdfplumber.open(temp_pdf_path) as pdf:
                with ThreadPoolExecutor() as executor:
                    futures = []
                    for page_num, page in enumerate(pdf.pages, 1):
                        futures.append(executor.submit(
                            process_pdf_page, page, pdf_key, page_num))
                    
                    for future in futures:
                        text, tables, source = future.result()
                        if text:
                            all_chunks.append(f"[{source}]\n{text}")
                        for table in tables:
                            all_chunks.append(f"[{source}-Table]\n{table}")"""

        # Process JSON files
        for json_key in json_files:
            json_obj = s3.get_object(Bucket=S3_BUCKET, Key=json_key)
            json_data = json.load(json_obj["Body"])
            
            records = json_data.get("records", [json_data] if not isinstance(json_data, list) else json_data)
            for i, record in enumerate(records, 1):
                text = "\n".join([f"{k}: {v}" for k, v in record.items() if v])
                if text:
                    all_chunks.append(f"[{json_key}-Record{i}]\n{text}")

        # ✨ Chunk the documents
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP
        )
        split_docs = []
        for chunk in all_chunks:
            split_docs.extend(text_splitter.create_documents([chunk]))

        # Build FAISS index
        faiss_store = FAISS.from_documents(split_docs, embeddings)

        # Save FAISS index to S3
        temp_dir = tempfile.mkdtemp()
        faiss_store.save_local(temp_dir)
        s3.upload_file(f"{temp_dir}/index.faiss", S3_BUCKET, f"{FAISS_INDEX_KEY}.index")
        s3.upload_file(f"{temp_dir}/index.pkl", S3_BUCKET, f"{FAISS_INDEX_KEY}.pkl")
        shutil.rmtree(temp_dir)

    # Load LLM from Bedrock
    llm = BedrockLLM(
        client=bedrock,
        model_id="meta.llama3-70b-instruct-v1:0",
        model_kwargs={
            "temperature": 0.2,
            "top_p": 0.9,
            "max_gen_len": 1024
        }
    )

    return RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=faiss_store.as_retriever(search_kwargs={"k": 3}),
        return_source_documents=True
    )

# UI - Lazy initialization
if "rag_initialized" not in st.session_state:
    if st.button("Initialize AI Engine"):
        with st.spinner("One-time setup (takes 1-2 minutes)..."):
            st.session_state.rag_chain = initialize_rag()
            st.session_state.rag_initialized = True
            st.rerun()
else:
    user_question = st.text_input("🔍 Ask your question:")
    if st.button("Generate Response") and user_question.strip():
        with st.spinner("Thinking..."):
            try:
                response = st.session_state.rag_chain({"query": custom_prompt.format(question=user_question[:500])})
                answer = response["result"]
                
                st.success("Response:")
                st.markdown(answer)
                
                if st.checkbox("Show sources"):
                    for doc in response["source_documents"]:
                        st.text(doc.page_content[:200] + "...")
                        
            except Exception as e:
                st.error(f"Error: {str(e)}")
