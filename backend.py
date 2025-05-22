import os
import json
from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from langchain_aws import BedrockLLM
from langchain_aws.embeddings import BedrockEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
import pdfplumber
import boto3
import tempfile
import pandas as pd
from langchain.schema import Document
from botocore.exceptions import ClientError, BotoCoreError
from langchain.text_splitter import RecursiveCharacterTextSplitter
from concurrent.futures import ThreadPoolExecutor
import shutil
import uvicorn

# Load environment variables
load_dotenv()

# Initialize FastAPI app
app = FastAPI(title="iBSNL API", description="API for BSNL customer support chatbot")

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Constants
MAX_INPUT_CHARS = 49000
CHUNK_SIZE = 1500
CHUNK_OVERLAP = 150
S3_BUCKET = "bsnlinsighthub"
FAISS_INDEX_KEY = "faiss_index"

class BSNLAssistant:
    def __init__(self):
        self.rag_chain = None
        self.initialized = False
        self.s3 = boto3.client('s3')
        self.bedrock = boto3.client('bedrock-runtime')
        self.custom_prompt = """You are a helpful AI assistant for BSNL Customers.

Answer the following question using the information from the provided documents. Be polite and concise.

give the below information in the answer only if asked for:

- Broadband Recharge: https://bsnl.co.in/broadband/FTTHbillpay

- Complaint Registration: https://bsnl.co.in/support/contact

- New Connection: https://bsnl.co.in/broadband/bharatfiber

- Plans and Tariffs: https://bsnl.co.in/mobile/prepaid

- Mobile Services: https://bsnl.co.in/mobile/porttobsnl


Question: {question}

Answer:"""
        
    def process_pdf_page(self, page, pdf_key, page_num):
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

    def load_or_create_index(self, embeddings):
        """Load cached FAISS index or create new"""
        temp_dir = tempfile.mkdtemp()
        
        try:
            self.s3.download_file(S3_BUCKET, f"{FAISS_INDEX_KEY}.index", f"{temp_dir}/index.faiss")
            self.s3.download_file(S3_BUCKET, f"{FAISS_INDEX_KEY}.pkl", f"{temp_dir}/index.pkl")
            faiss_store = FAISS.load_local(temp_dir, embeddings)
            return faiss_store
        except Exception as e:
            print(f"Building new embeddings: {str(e)}")
            return None

    def initialize_rag(self):
        """Initialize the RAG system"""
        if self.initialized:
            return True
            
        try:
            # Try loading cached index
            embeddings = BedrockEmbeddings(client=self.bedrock, model_id="amazon.titan-embed-text-v2:0")
            faiss_store = self.load_or_create_index(embeddings)

            if faiss_store is None:
                all_chunks = []
                pdf_files = ["BSNL_CDA.pdf"]
                json_files = [
                    "bbc_data.json",
                    "EB_NAM_details_with_metadata.json",
                    "Franchise_managers_data_with_metadata.json"
                ]

                # Process PDFs
                for pdf_key in pdf_files:
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_pdf:
                        self.s3.download_fileobj(S3_BUCKET, pdf_key, temp_pdf)
                        temp_pdf_path = temp_pdf.name

                    with pdfplumber.open(temp_pdf_path) as pdf:
                        with ThreadPoolExecutor() as executor:
                            futures = []
                            for page_num, page in enumerate(pdf.pages, 1):
                                futures.append(executor.submit(
                                    self.process_pdf_page, page, pdf_key, page_num))
                            
                            for future in futures:
                                text, tables, source = future.result()
                                if text:
                                    all_chunks.append(f"[{source}]\n{text}")
                                for table in tables:
                                    all_chunks.append(f"[{source}-Table]\n{table}")

                # Process JSON files
                for json_key in json_files:
                    json_obj = self.s3.get_object(Bucket=S3_BUCKET, Key=json_key)
                    json_data = json.load(json_obj["Body"])
                    
                    records = json_data.get("records", [json_data] if not isinstance(json_data, list) else json_data)
                    for i, record in enumerate(records, 1):
                        text = "\n".join([f"{k}: {v}" for k, v in record.items() if v])
                        if text:
                            all_chunks.append(f"[{json_key}-Record{i}]\n{text}")

                # Chunk the documents
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
                self.s3.upload_file(f"{temp_dir}/index.faiss", S3_BUCKET, f"{FAISS_INDEX_KEY}.index")
                self.s3.upload_file(f"{temp_dir}/index.pkl", S3_BUCKET, f"{FAISS_INDEX_KEY}.pkl")
                shutil.rmtree(temp_dir)

            # Load LLM from Bedrock
            llm = BedrockLLM(
                client=self.bedrock,
                model_id="mistral.mixtral-8x7b-instruct-v0:1",
                model_kwargs={
                    "temperature": 0.2,
                    "top_p": 0.9,
                    "max_gen_len": 1024
                }
            )

            self.rag_chain = RetrievalQA.from_chain_type(
                llm=llm,
                chain_type="stuff",
                retriever=faiss_store.as_retriever(search_kwargs={"k": 3}),
                return_source_documents=True
            )
            
            self.initialized = True
            return True
            
        except Exception as e:
            print(f"Initialization failed: {str(e)}")
            return False

# Global assistant instance
assistant = BSNLAssistant()

@app.on_event("startup")
async def startup_event():
    """Initialize the assistant on startup"""
    if not assistant.initialize_rag():
        raise RuntimeError("Failed to initialize AI engine")

@app.get("/status")
async def get_status():
    """Check service status"""
    return {
        "status": "running",
        "initialized": assistant.initialized
    }

@app.post("/query")
async def handle_query(question: str = Form(...), include_sources: bool = Form(False)):
    """Handle user queries"""
    if not assistant.initialized:
        raise HTTPException(status_code=503, detail="Service not initialized")
    
    try:
        if len(question) > MAX_INPUT_CHARS:
            question = question[:MAX_INPUT_CHARS]
            
        response = assistant.rag_chain({"query": assistant.custom_prompt.format(question=question)})
        
        result = {
            "answer": response["result"],
            "status": "success"
        }
        
        if include_sources:
            result["sources"] = [doc.page_content[:200] + "..." for doc in response["source_documents"]]
            
        return JSONResponse(content=result)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)