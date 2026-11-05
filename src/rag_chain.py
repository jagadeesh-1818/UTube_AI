from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda
from langdetect import detect
from langdetect.lang_detect_exception import LangDetectException


# Initialize LLM

llm = ChatGroq(

    model="openai/gpt-oss-120b",

    temperature=0,

    max_tokens=1024,

    reasoning_effort="low",

    reasoning_format="hidden",

    max_retries=2,

    timeout=60

)


parser = StrOutputParser()


# Format retrieved documents

def format_docs(docs):

    formatted_docs = []

    for doc in docs:

        timestamp = doc.metadata.get("start", 0)

        minutes = int(timestamp // 60)

        seconds = int(timestamp % 60)

        language = doc.metadata.get("language", "unknown")

        formatted_docs.append(
            f"""
[Timestamp: {minutes:02d}:{seconds:02d}]
[Language: {language}]

{doc.page_content}
"""
        )

    return "\n\n".join(formatted_docs)


# Format conversation history

def format_history(chat_history):

    formatted_history = []

    for message in chat_history:

        role = (
            "User"
            if message.type == "human"
            else "Assistant"
        )

        formatted_history.append(
            f"{role}: {message.content}"
        )

    return "\n".join(formatted_history)


# ---------------------------------
# 1. Question Rewriting Prompt
# ---------------------------------

rewrite_prompt = PromptTemplate(

    template="""
You are a multilingual question rewriting assistant.

Your task is to convert the user's latest question
into a standalone question using conversation history.

Rules:

1. Preserve the language of the current question.
2. Resolve references such as "it", "that", "they",
   "its benefits", and "the second point".
3. Do not answer the question.
4. If the question is already standalone, return it unchanged.
5. Return only the rewritten question.

Conversation History:
{chat_history}

Current Question:
{question}

Standalone Question:
""",

    input_variables=[
        "chat_history",
        "question"
    ]

)


rewrite_chain = rewrite_prompt | llm | parser


# ---------------------------------
# 2. Answer Generation Prompt
# ---------------------------------

answer_prompt = PromptTemplate(

    template="""
You are UTube AI, a multilingual YouTube video assistant.

You answer questions based on YouTube video transcripts.

Instructions:
1. Answer using ONLY the provided transcript context.
2. Understand the transcript regardless of its language.
3. The transcript language and response language can be different.
4. You MUST respond entirely in {response_language}.
5. NEVER copy the transcript's language for your answer
   unless it matches the requested response language.
6. Ignore the language used in the transcript when deciding
   the response language.
7. Never invent information unavailable in the transcript.
8. If the transcript does not contain the answer,
   clearly say so in {response_language}.
9. Include relevant timestamps whenever possible.

Conversation History:

{chat_history}

Retrieved Transcript Context:

{context}

Current Question:

{question}

Answer:
""",

    input_variables=[
    "chat_history",
    "context",
    "question",
    "response_language"
]

)


answer_chain = answer_prompt | llm | parser


# ---------------------------------
# 3. Build Complete RAG Function
# ---------------------------------


def get_rag_response(question, chat_history, retriever, return_details=False):

    history_text = format_history(chat_history)

    # Detect the language of the user's current question
    response_language = get_response_language(question)

    print(f"\nResponse language: {response_language}")

    # Rewrite follow-up questions if chat history exists
    if chat_history:

        standalone_question = rewrite_chain.invoke({
            "chat_history": history_text,
            "question": question
        })

    else:
        standalone_question = question

    print("\nRetrieval query:", standalone_question)

    # Retrieve relevant transcript chunks
    retrieved_docs = retriever.invoke(standalone_question)

    if not retrieved_docs:
        return "I couldn't find relevant information in this video."

    print("\nRetrieved chunks:")

    for i, doc in enumerate(retrieved_docs):
        print(f"\nChunk {i + 1}:")
        print(doc.page_content[:300])
        print("Metadata:", doc.metadata)

    context = format_docs(retrieved_docs)

    # Generate answer in the user's question language
    response = answer_chain.invoke({

        "chat_history": history_text,
        "context": context,
        "question": question,
        "response_language": response_language

    })

    if return_details:

        return {
            "answer": response,
            "context": context,
            "retrieved_chunks": [
                {
                    "chunk_id": doc.metadata.get("chunk_id"),
                    "timestamp": doc.metadata.get("start"),
                    "content": doc.page_content
                }
                for doc in retrieved_docs
            ]
        }

    return response


def get_response_language(question):

    language_names = {
        "en": "English",
        "hi": "Hindi",
        "kn": "Kannada",
        "ta": "Tamil",
        "te": "Telugu",
        "ml": "Malayalam",
        "bn": "Bengali",
        "mr": "Marathi",
        "zh-cn": "Chinese",
        "zh-tw": "Traditional Chinese",
        "es": "Spanish",
        "fr": "French",
        "de": "German",
        "ja": "Japanese"
    }

    try:
        detected_language = detect(question)
        return language_names.get(
            detected_language,
            detected_language
        )

    except LangDetectException:
        return "English"