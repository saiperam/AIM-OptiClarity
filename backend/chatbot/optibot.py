import os
from fastapi import FastAPI, Request
from pydantic import BaseModel
import uvicorn
from fastapi.middleware.cors import CORSMiddleware
from langchain_huggingface import HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from langchain.retrievers.document_compressors import CrossEncoderReranker
from langchain.retrievers import EnsembleRetriever, ContextualCompressionRetriever
from langchain_community.retrievers import BM25Retriever
from langchain.chains import RetrievalQA
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from langchain_community.vectorstores import FAISS

# FastAPI app setup
app = FastAPI()

# Allow CORS from the frontend (localhost:5173)
origins = [
    "http://localhost:5173",  # Your frontend URL
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load HuggingFace Token and Model ID
HF_TOKEN = os.environ.get("HF_TOKEN")
HUGGINGFACE_REPO_ID = "mistralai/Mistral-7B-Instruct-v0.3"
DB_FAISS_PATH = "vectorstore/db_faiss"

# Step 1: Setup LLM (Mistral with HuggingFace)
def load_llm(huggingface_repo_id):
    llm = HuggingFaceEndpoint(
        repo_id=huggingface_repo_id,
        task="text-generation",
        temperature=0.4,
        model_kwargs={"token": HF_TOKEN, "max_length": "512"}
    )
    return llm

# Step 2: Define custom prompt template
CUSTOM_PROMPT_TEMPLATE = """
Use the pieces of information provided in the context to answer user's question.
If you don't know the answer, just say that you don't know, don't try to make up an answer.
Don't provide anything out of the given context.

Context: {context}
Question: {question}

Start the answer directly. No small talk please.
"""
def set_custom_prompt(custom_prompt_template):
    prompt = PromptTemplate(template=custom_prompt_template, input_variables=["context", "question"])
    return prompt

# Load Database
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vector_db = FAISS.load_local(DB_FAISS_PATH, embedding_model, allow_dangerous_deserialization=True)

# Step 3: Create ensemble retriever along with cross-encoder re-ranking algorithm
dense_retriever = vector_db.as_retriever(search_kwargs={'k': 15})
documents = list(vector_db.docstore._dict.values())
sparse_retriever = BM25Retriever.from_documents(documents)
sparse_retriever.k = 15

ensemble_retriever = EnsembleRetriever(
    retrievers=[dense_retriever, sparse_retriever],
    weights=[.7, .3]
)

cross_encoder = HuggingFaceCrossEncoder(model_name="BAAI/bge-reranker-base")
compressor = CrossEncoderReranker(model=cross_encoder, top_n=3)
reranking_retriever = ContextualCompressionRetriever(
    base_compressor=compressor,
    base_retriever=ensemble_retriever
)

# Create QA chain
qa_chain = RetrievalQA.from_chain_type(
    llm=load_llm(HUGGINGFACE_REPO_ID),
    chain_type="stuff",
    retriever=reranking_retriever,
    return_source_documents=False,
    chain_type_kwargs={'prompt': set_custom_prompt(CUSTOM_PROMPT_TEMPLATE)}
)

# Define request schema
class QueryRequest(BaseModel):
    query: str

# FastAPI route to handle chat query
@app.post("/chat")
async def chat(request: QueryRequest):
    result = qa_chain.invoke({'query': request.query})
    return {"response": result["result"]}

# Run FastAPI app
if __name__ == "__main__":
    uvicorn.run("optibot:app", host="localhost", port=8000, reload=True)











"""from fastapi import FastAPI, Request
from pydantic import BaseModel
import uvicorn
from fastapi.middleware.cors import CORSMiddleware
from langchain_huggingface import HuggingFaceEndpoint, HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain_core.prompts import PromptTemplate
from langchain_community.vectorstores import FAISS

import os

# Load HuggingFace Token and Model ID
HF_TOKEN = os.environ.get("HF_TOKEN")
HUGGINGFACE_REPO_ID = "mistralai/Mistral-7B-Instruct-v0.3"
DB_FAISS_PATH = "vectorstore/db_faiss"

# Load Embedding Model and FAISS Vector Store
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
db = FAISS.load_local(DB_FAISS_PATH, embedding_model, allow_dangerous_deserialization=True)

# Define custom prompt template
CUSTOM_PROMPT_TEMPLATE = 
Use the pieces of information provided in the context to answer user's question.
If you don't know the answer, just say that you don't know. Don't try to make up an answer.
Don't provide anything out of the given context.

Context: {context}
Question: {question}

Start the answer directly. No small talk please.

prompt = PromptTemplate(template=CUSTOM_PROMPT_TEMPLATE, input_variables=["context", "question"])

# Load the LLM
def load_llm():
    llm = HuggingFaceEndpoint(
        repo_id=HUGGINGFACE_REPO_ID,
        task="text-generation",
        temperature=0.4,
        model_kwargs={"token": HF_TOKEN, "max_length": "512"}
    )
    return llm

# Setup RetrievalQA Chain
qa_chain = RetrievalQA.from_chain_type(
    llm=load_llm(),
    chain_type="stuff",
    retriever=db.as_retriever(search_kwargs={"k": 4}),
    return_source_documents=False,
    chain_type_kwargs={"prompt": prompt}
)

# FastAPI app setup
app = FastAPI()

# Allow CORS from the frontend (localhost:5173)
origins = [
    "http://localhost:5173",  # Your frontend URL
]

# Allow CORS for frontend connection (e.g., React)
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request schema
class QueryRequest(BaseModel):
    query: str

# Route to handle chat query
@app.post("/chat")
async def chat(request: QueryRequest):
    result = qa_chain.invoke({'query': request.query})
    return {"response": result["result"]}

if __name__ == "__main__":
    uvicorn.run("optibot:app", host="localhost", port=8000, reload=True)"""