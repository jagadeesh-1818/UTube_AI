
import streamlit as st

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings


# Load and cache the multilingual embedding model
@st.cache_resource(show_spinner="Loading multilingual embedding model...")
def get_embeddings():

    embeddings = HuggingFaceEmbeddings(

        model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",

        model_kwargs={
            "device": "cpu"
        },

        encode_kwargs={
            "normalize_embeddings": True
        }
    )

    return embeddings


def create_vector_store(documents):

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )

    chunks = splitter.split_documents(documents)

    # Assign unique IDs
    for i, chunk in enumerate(chunks):
        chunk.metadata["chunk_id"] = f"chunk_{i:03d}"

    if not chunks:
        raise ValueError("No chunks generated from transcript.")

    vector_store = FAISS.from_documents(
        documents=chunks,
        embedding=get_embeddings()
    )

    print("FAISS vector database created successfully.")

    return vector_store


def create_retriever(vector_store):

    return vector_store.as_retriever(
        search_type="mmr",
        search_kwargs={
            "k": 8,
            "fetch_k": 25,
            "lambda_mult": 0.95
        }
    )
