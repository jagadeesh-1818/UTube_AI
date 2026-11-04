from dotenv import load_dotenv

# Load environment variables first
load_dotenv()

import os

from langchain_core.messages import HumanMessage, AIMessage
from src.transcript import extract_video_id, extract_transcript
from src.vector_store import (
    create_vector_store,
    create_retriever
)
from src.rag_chain import get_rag_response


# ---------------------------------
# 1. Check API Key
# ---------------------------------

if not os.getenv("GROQ_API_KEY"):
    raise ValueError(
        "GROQ_API_KEY not found. Check your .env file."
    )


# ---------------------------------
# 2. Get YouTube URL
# ---------------------------------

url = input("\nEnter YouTube URL: ").strip()

video_id = extract_video_id(url)

if not video_id:
    raise ValueError("Invalid YouTube URL.")


# ---------------------------------
# 3. Extract Transcript
# ---------------------------------

print("\nExtracting transcript...")

documents, transcript_language = extract_transcript(video_id)

print("\nTranscript extracted successfully!")

print("Transcript language:", transcript_language)


# ---------------------------------
# 4. Create Vector Database
# ---------------------------------

print("\nCreating vector database...")

vector_store = create_vector_store(documents)

retriever = create_retriever(vector_store)


# ---------------------------------
# 5. Initialize Chat History
# ---------------------------------

chat_history = []


# ---------------------------------
# 6. Start Chatbot
# ---------------------------------

print("\n====================================")
print("          Welcome to UTube AI")
print("====================================")

print("\nAsk questions about your YouTube video.")

print("Type 'exit' or 'quit' to stop.")


while True:

    question = input("\nYou: ").strip()

    if not question:
        continue

    if question.lower() in ["exit", "quit"]:
        print("\nThank you for using UTube AI!")
        break

    try:

        response = get_rag_response(
            question=question,
            chat_history=chat_history,
            retriever=retriever
        )

        print("\nUTube AI:", response)

        # Update conversation history

        chat_history.append(
            HumanMessage(content=question)
        )

        chat_history.append(
            AIMessage(content=response)
        )

    except Exception as e:

        print(
            f"\nError: {type(e).__name__}: {e}"
        )