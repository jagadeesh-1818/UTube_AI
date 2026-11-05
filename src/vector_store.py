from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings


# Initialize multilingual embedding model

embeddings = HuggingFaceEmbeddings(

    model_name="BAAI/bge-m3",

    model_kwargs={
        "device": "cpu"
    },

    encode_kwargs={
        "normalize_embeddings": True
    }
)



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
        embedding=embeddings
    )

    print("FAISS vector database created successfully.")

    return vector_store



def create_retriever(vector_store):
    return vector_store.as_retriever(
        search_type="mmr",
        search_kwargs={
            "k": 6,
            "fetch_k": 20,
            "lambda_mult": 0.7
        }
    )
