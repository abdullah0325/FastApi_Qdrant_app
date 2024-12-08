
import os
import streamlit as st 
from langchain_community.document_loaders import PyMuPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_qdrant import QdrantVectorStore
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough

# Configuration
OPENAI_API_KEY ="openai_api_key"
QDRANT_URL = "Qdrant_url"
QDRANT_KEY = "Qdrant_key"
COLLECTION_NAME = "fastapi"
MODEL_NAME = "gpt-4o-mini"

# Initialize embedding model
embed_model = HuggingFaceEmbeddings(model_name='BAAI/bge-small-en-v1.5')

def process_pdf(file_path):
    """Load and split PDF into chunks."""
    loader = PyMuPDFLoader(file_path)
    pages = loader.load()
    text_splitter = CharacterTextSplitter(chunk_size=100, chunk_overlap=10)
    splits = text_splitter.split_documents(pages)
    return pages, splits

def create_vectorstore(splits):
    """Create and return a Qdrant vector store."""
    return QdrantVectorStore.from_documents(
        splits, embed_model, url=QDRANT_URL, api_key=QDRANT_KEY, collection_name=COLLECTION_NAME
    )

def generate_response(vectorstore, question):
    """Retrieve context and generate a response."""
    # Set up the retriever
    retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 3})
    
    # Create a prompt template
    prompt_template = PromptTemplate.from_template(
        "Answer the question based on the context:\n\n{context}\n\nQuestion: {question}"
    )
    
    # Create the LLM
    chat_llm = ChatOpenAI(model=MODEL_NAME, openai_api_key=OPENAI_API_KEY, temperature=0)
    
    # Construct the LCEL chain
    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)
    
    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt_template
        | chat_llm
    )
    
    # Invoke the chain and return the content
    response = rag_chain.invoke(question)
    return response.content