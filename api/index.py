from fastapi import FastAPI, HTTPException, Request, UploadFile, File, Form
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
import os
import requests
import io
from typing import Optional

from langchain_groq import ChatGroq
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from langchain.text_splitter import CharacterTextSplitter
from langchain.document_loaders import PyPDFLoader
from langchain.chains import RetrievalQA
from langchain.agents import initialize_agent, Tool
from langchain.agents import AgentType
from langchain.memory import ConversationBufferMemory
from langchain.utilities import DuckDuckGoSearchAPIWrapper
from langchain.prompts import PromptTemplate
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

app = FastAPI()

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")

# Set up Jinja2 templates
templates = Jinja2Templates(directory="templates")

# Initialize components with environment variables
llm = ChatGroq(
    api_key=os.environ.get('GROQ_API_KEY'),
    model_name="llama3-8b-8192",
    max_tokens=1024,
)
embeddings = HuggingFaceEmbeddings()

# Get the subscription key from environment variables
SUBSCRIPTION_KEY = os.environ.get('SUBSCRIPTION_KEY')

# Define your Query model
class Query(BaseModel):
    text: str

# Function to load PDF from URL
def load_pdf_from_url(url):
    response = requests.get(url)
    if response.status_code == 200:
        pdf_file = io.BytesIO(response.content)
        loader = PyPDFLoader(pdf_file)
        documents = loader.load()
        return documents
    else:
        raise Exception(f"Failed to download PDF from {url}")

# Function to initialize vector store
def initialize_vector_store(documents, embeddings):
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    texts = text_splitter.split_documents(documents)
    vectorstore = Chroma.from_documents(documents=texts, embedding=embeddings)
    return vectorstore.as_retriever(search_kwargs={"k": 3})

# Set up DuckDuckGo search
search = DuckDuckGoSearchAPIWrapper()

def web_search(query: str) -> str:
    results = search.run(query)
    return f"Web search results: {results}"

# Function to decide which tool to use
def decide_tool(query: str, documents) -> str:
    query_embedding = embeddings.embed_query(query)
    pdf_embeddings = [embeddings.embed_query(doc.page_content) for doc in documents]
    
    similarities = cosine_similarity([query_embedding], pdf_embeddings)[0]
    max_similarity = np.max(similarities)
    
    similarity_threshold = 0.5  # Adjust this threshold as needed
    
    if max_similarity > similarity_threshold:
        return "VectorDB QA System"
    else:
        return "Web Search"

# Custom prompt template
custom_prompt_template = """Use the following pieces of context to answer the question at the end.
If you don't know the answer, just say that you don't know, don't try to make up an answer.

{context}

Question: {question}
Answer: """
CUSTOM_PROMPT = PromptTemplate(
    template=custom_prompt_template, input_variables=["context", "question"]
)

# Serve the index.html file
@app.get("/", response_class=HTMLResponse)
async def read_index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

# RAG chatbot endpoint
@app.post("/api/rag-chatbot")
async def rag_chatbot_endpoint(query: Query):
    pdf_url = os.environ.get('PDF_URL')
    documents = load_pdf_from_url(pdf_url)
    retriever = initialize_vector_store(documents, embeddings)
    
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        chain_type_kwargs={"prompt": CUSTOM_PROMPT}
    )
    
    response = qa_chain.run(query.text)
    return {"response": response}

# Agent endpoint
@app.post("/api/agent")
async def agent_endpoint(query: Query):
    pdf_url = os.environ.get('PDF_URL')
    documents = load_pdf_from_url(pdf_url)
    retriever = initialize_vector_store(documents, embeddings)
    
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        chain_type_kwargs={"prompt": CUSTOM_PROMPT}
    )
    
    tools = [
        Tool(
            name="VectorDB QA System",
            func=qa_chain.run,
            description="Useful for answering questions about specific document content, topics from the PDF."
        ),
        Tool(
            name="Web Search",
            func=web_search,
            description="Useful for general questions or current information not found in the document."
        )
    ]
    
    system_message = """You are an intelligent assistant with access to two tools: a VectorDB QA System for answering questions about specific document content (topics from the PDF), and a Web Search tool for general questions or current information. 
    Your first step for any query is to use the decide_tool function to determine which tool is most appropriate. Always use the tool suggested by decide_tool.
    If VectorDB QA System is suggested, use it to answer questions related to the document content. 
    If Web Search is suggested, use it for general questions or information not likely to be in the document."""

    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    agent = initialize_agent(
        tools,
        llm,
        agent=AgentType.CHAT_CONVERSATIONAL_REACT_DESCRIPTION,
        verbose=True,
        memory=memory,
        handle_parsing_errors=True,
        agent_kwargs={
            "system_message": system_message,
        },
    )
    
    tool_to_use = decide_tool(query.text, documents)
    response = f"Based on content similarity, I'll use the {tool_to_use}.\n"
    response += agent.run(f"Use the {tool_to_use} to answer: {query.text}")
    return {"response": response}

# Speech-to-text endpoint
@app.post("/api/speech-to-text")
async def speech_to_text(file: UploadFile = File(...), language_code: str = Form(...)):
    try:
        content = await file.read()
        headers = {
            "API-Subscription-Key": SUBSCRIPTION_KEY,
            "Content-Type": "audio/wav"
        }
        params = {
            "language_code": language_code,
            "model": "saarika:v1"
        }
        response = requests.post(
            "https://api.sarvam.ai/speech-to-text",
            headers=headers,
            params=params,
            data=content
        )
        response.raise_for_status()
        return response.json()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Healthcheck endpoint for Vercel
@app.get("/api/healthcheck")
def healthcheck():
    return {"status": "ok"}

# Run the FastAPI app
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)