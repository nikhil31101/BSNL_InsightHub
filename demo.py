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
import uuid
from datetime import datetime
import random
import time
from botocore.exceptions import ClientError
from boto3.dynamodb.conditions import Key

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

# Add these constants after your existing ones
DYNAMODB_TABLE = "bsnl_conversations"
COGNITO_USER_POOL_ID = "your-cognito-pool-id"  # Replace with your Cognito User Pool ID
COGNITO_CLIENT_ID = "your-cognito-client-id"  # Replace with your Cognito App Client ID
OTP_EXPIRY_MINUTES = 5

# Initialize AWS clients (add after load_dotenv())
cognito = boto3.client('cognito-idp')
dynamodb = boto3.resource('dynamodb')
conversation_table = dynamodb.Table(DYNAMODB_TABLE)

# Add these new functions
def generate_otp():
    """Generate a 6-digit OTP"""
    return str(random.randint(100000, 999999))

def send_otp(mobile_number):
    """Send OTP via SMS using Cognito"""
    try:
        # Remove +91 if present and add country code
        if mobile_number.startswith('+91'):
            mobile_number = mobile_number[3:]
        elif mobile_number.startswith('0'):
            mobile_number = mobile_number[1:]
        
        response = cognito.sign_up(
            ClientId=COGNITO_CLIENT_ID,
            Username=mobile_number,
            Password=str(uuid.uuid4()),  # Dummy password
            UserAttributes=[
                {'Name': 'phone_number', 'Value': f'+91{mobile_number}'}
            ]
        )
        st.session_state.otp_code = generate_otp()
        st.session_state.otp_expiry = time.time() + (OTP_EXPIRY_MINUTES * 60)
        st.session_state.mobile_number = mobile_number
        
        # In production, you would actually send the OTP via SMS
        # This is just for demo - in real app use AWS SNS or Cognito's built-in SMS
        st.success(f"Demo OTP sent to +91{mobile_number}: {st.session_state.otp_code}")
        return True
    except ClientError as e:
        st.error(f"Error sending OTP: {e.response['Error']['Message']}")
        return False

def verify_otp(user_otp):
    """Verify the OTP entered by user"""
    if 'otp_code' not in st.session_state:
        return False
        
    if time.time() > st.session_state.otp_expiry:
        st.error("OTP has expired. Please request a new one.")
        return False
        
    if user_otp == st.session_state.otp_code:
        st.session_state.authenticated = True
        st.session_state.session_id = str(uuid.uuid4())
        return True
    else:
        st.error("Invalid OTP. Please try again.")
        return False

def store_conversation(question, answer, sources=None):
    """Store conversation in DynamoDB"""
    if 'session_id' not in st.session_state:
        return
        
    try:
        item = {
            'SessionID': st.session_state.session_id,
            'Timestamp': datetime.utcnow().isoformat(),
            'MobileNumber': st.session_state.mobile_number,
            'Question': question[:500],
            'Answer': answer[:2000],
            'Sources': sources or []
        }
        conversation_table.put_item(Item=item)
    except Exception as e:
        st.error(f"Failed to save conversation: {str(e)}")

def get_conversation_history():
    """Retrieve user's conversation history"""
    if 'session_id' not in st.session_state:
        return []
        
    try:
        response = conversation_table.query(
            KeyConditionExpression=Key('SessionID').eq(st.session_state.session_id),
            Limit=10,
            ScanIndexForward=False
        )
        return response.get('Items', [])
    except Exception as e:
        st.error(f"Failed to load history: {str(e)}")
        return []

# Authentication UI - Add this before your main content
if 'authenticated' not in st.session_state:
    st.session_state.authenticated = False

if not st.session_state.authenticated:
    st.title("BSNL Customer Login")
    mobile_number = st.text_input("Enter Mobile Number (10 digits)", max_chars=10)
    
    if st.button("Send OTP"):
        if len(mobile_number) == 10 and mobile_number.isdigit():
            if send_otp(mobile_number):
                st.session_state.otp_sent = True
        else:
            st.error("Please enter a valid 10-digit mobile number")
    
    if st.session_state.get('otp_sent'):
        user_otp = st.text_input("Enter OTP", max_chars=6)
        if st.button("Verify OTP"):
            if verify_otp(user_otp):
                st.success("Authentication successful!")
                time.sleep(1)
                st.rerun()
    st.stop()

# Main App UI (only shown after authentication)
st.markdown(
    """
    <div style='padding: 9px;'>
            </div>
    """,
    unsafe_allow_html=True
)

# Add conversation history panel
history = get_conversation_history()
if history:
    with st.expander("📜 Conversation History", expanded=False):
        for item in history:
            st.markdown(f"**Q:** {item['Question']}")
            st.markdown(f"**A:** {item['Answer']}")
            if item.get('Sources'):
                with st.expander("View Sources"):
                    st.write(item['Sources'])
            st.markdown("---")

# Modify your existing QA section
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
                
                # Store conversation
                sources = [doc.page_content[:200] + "..." for doc in response["source_documents"]]
                store_conversation(
                    question=user_question,
                    answer=answer,
                    sources=sources
                )
                
                st.success("Response:")
                st.markdown(answer)
                
                if st.checkbox("Show sources"):
                    for doc in response["source_documents"]:
                        st.text(doc.page_content[:200] + "...")
                        
            except Exception as e:
                st.error(f"Error: {str(e)}")

# Add logout button
if st.button("Logout"):
    for key in list(st.session_state.keys()):
        del st.session_state[key]
    st.rerun()