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

    # Text splitting

    splitter = RecursiveCharacterTextSplitter(

        chunk_size=1000,

        chunk_overlap=200

    )

    chunks = splitter.split_documents(documents)

    if not chunks:

        raise ValueError("No chunks generated from transcript.")

    print(f"\nTotal chunks created: {len(chunks)}")

    # Generate embeddings and create FAISS database

    vector_store = FAISS.from_documents(

        documents=chunks,

        embedding=embeddings

    )

    print("FAISS vector database created successfully.")

    return vector_store


def create_retriever(vector_store):

    retriever = vector_store.as_retriever(

        search_type="similarity",

        search_kwargs={
            "k": 4
        }

    )

    return retriever