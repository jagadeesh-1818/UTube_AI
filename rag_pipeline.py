from youtube_transcript_api import YouTubeTranscriptApi
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from dotenv import load_dotenv
from urllib.parse import urlparse, parse_qs
from langchain_core.runnables import RunnableParallel, RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

#initialise LLM
llm = ChatGroq(
    model="openai/gpt-oss-120b",
    temperature=0,
    max_tokens=512,
    reasoning_effort="low",
    reasoning_format="hidden",
    max_retries=2,
    timeout=30
)

#initialise Embedding model
embeddings = HuggingFaceEmbeddings(
    model_name= "BAAI/bge-small-en-v1.5",
    model_kwargs={"device": "cpu"},
    encode_kwargs={"normalize_embeddings": True}
)

#extract video transcript
def extract_video_id(url):

    parsed_url = urlparse(url)

    if parsed_url.hostname in ["youtu.be", "www.youtu.be"]:
        return parsed_url.path.lstrip("/")

    if parsed_url.hostname in ["youtube.com", "www.youtube.com", "m.youtube.com"]:

        if parsed_url.path == "/watch":
            return parse_qs(parsed_url.query).get("v", [None])[0]

        if parsed_url.path.startswith(("/shorts/", "/embed/", "/live/")):
            return parsed_url.path.split("/")[2]

    return None


url = input("Enter YouTube URL: ")

video_id = extract_video_id(url)

try:
    ytt_api = YouTubeTranscriptApi()

    transcript_list = ytt_api.fetch(
        video_id,
        languages=["en"]
    )

    transcript = " ".join(
        snippet.text for snippet in transcript_list
    )

    #print(transcript)

except Exception as e:
    print(type(e).__name__, e)

#making chunks and embedding vector of chunks
splitter = RecursiveCharacterTextSplitter(chunk_size= 1000, chunk_overlap= 200)
chunks = splitter.create_documents([transcript])

vector_store = FAISS.from_documents(documents= chunks, embedding=embeddings)

#Retrieval
retriever = vector_store.as_retriever(search_type= 'similarity', search_kwargs= {'k':4})

#Prompt

prompt = PromptTemplate(template="You are a helpful assistant." \
"answer only from the provided transcript context." \
"if the context is insufficient, just say you don't know." \
"{context}" \
"Question: {question}",
input_variables=['context', 'question']
)

parser = StrOutputParser()

question = input('Enter your query:')

def format_docs(retrieved_docs):
    context_text = "\n\n".join(doc.page_content for doc in retrieved_docs)
    return context_text

#Chain
parallel_chain = RunnableParallel({
    'context' : retriever | RunnableLambda(format_docs),
    'question': RunnablePassthrough()
})

chain = parallel_chain | prompt | llm | parser

#LLM Response Generation
response = chain.invoke(question)
print(f'\nAI: {response}')