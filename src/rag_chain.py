from langchain_groq import ChatGroq

from langchain_core.prompts import PromptTemplate

from langchain_core.output_parsers import StrOutputParser

from langchain_core.runnables import RunnableLambda


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
3. The user's question may be in a different language
   from the transcript.
4. Respond in the same language as the user's
   CURRENT question.
5. Use conversation history to understand follow-up questions.
6. Never invent information that is not available in the transcript.
7. If the transcript does not contain the answer,
   clearly say that you don't know based on this video.
8. When possible, include the relevant video timestamp.
9. Give clear and understandable answers.

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
        "question"
    ]

)


answer_chain = answer_prompt | llm | parser


# ---------------------------------
# 3. Build Complete RAG Function
# ---------------------------------

def get_rag_response(
    question,
    chat_history,
    retriever
):

    # Format history

    history_text = format_history(chat_history)

    # Rewrite follow-up questions

    if chat_history:

        standalone_question = rewrite_chain.invoke({

            "chat_history": history_text,

            "question": question

        })

    else:

        standalone_question = question

    print("\nRetrieval query:", standalone_question)

    # Retrieve relevant transcript chunks

    retrieved_docs = retriever.invoke(
        standalone_question
    )

    if not retrieved_docs:

        return "I couldn't find relevant information in this video."

    # Debug retrieved documents

    print("\nRetrieved chunks:")

    for i, doc in enumerate(retrieved_docs):

        print(f"\nChunk {i + 1}:")

        print(doc.page_content[:300])

        print("Metadata:", doc.metadata)

    # Format context

    context = format_docs(retrieved_docs)

    # Generate final answer

    response = answer_chain.invoke({

        "chat_history": history_text,

        "context": context,

        "question": question

    })

    return response