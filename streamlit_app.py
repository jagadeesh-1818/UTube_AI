import streamlit as st
from dotenv import load_dotenv
import os
from langchain_core.messages import HumanMessage, AIMessage

# Load .env before importing the RAG chain
load_dotenv(override=True)

from src.transcript import extract_video_id, extract_transcript
from src.vector_store import create_vector_store, create_retriever
from src.rag_chain import get_rag_response


# ---------------------------------
# 1. Configuration
# ---------------------------------

st.set_page_config(
    page_title="UTube AI",
    page_icon="🎥",
    layout="wide",
    initial_sidebar_state="expanded"
)

if not os.getenv("GROQ_API_KEY"):
    st.error("GROQ_API_KEY not found. Please check your .env file.")
    st.stop()


# ---------------------------------
# 2. Session State
# ---------------------------------

if "retriever" not in st.session_state:
    st.session_state.retriever = None

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "video_url" not in st.session_state:
    st.session_state.video_url = None

if "video_language" not in st.session_state:
    st.session_state.video_language = None


# ---------------------------------
# 3. Custom Styling
# ---------------------------------

st.markdown("""
<style>
    .main-title {
        font-size: 42px;
        font-weight: 800;
        text-align: center;
        margin-bottom: 0;
    }

    .subtitle {
        text-align: center;
        color: #999999;
        font-size: 16px;
        margin-bottom: 30px;
    }

    .stButton button {
        border-radius: 10px;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)


# ---------------------------------
# 4. Header
# ---------------------------------

st.markdown(
    '<div class="main-title">🎥 UTube AI</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'Your multilingual AI assistant for YouTube videos'
    '</div>',
    unsafe_allow_html=True
)


# ---------------------------------
# 5. Sidebar - Video Input
# ---------------------------------

with st.sidebar:

    st.title("⚙️ Video Settings")

    youtube_url = st.text_input(
        "Enter YouTube URL",
        placeholder="https://www.youtube.com/watch?v=..."
    )

    process_video = st.button(
        "🚀 Process Video",
        use_container_width=True
    )

    if process_video:

        if not youtube_url.strip():
            st.warning("Please enter a YouTube URL.")

        else:

            video_id = extract_video_id(youtube_url.strip())

            if not video_id:
                st.error("Invalid YouTube URL.")

            else:

                # Clear old video data
                st.session_state.retriever = None
                st.session_state.chat_history = []
                st.session_state.video_url = None
                st.session_state.video_language = None

                try:

                    with st.spinner("Extracting YouTube transcript..."):

                        documents, language = extract_transcript(video_id)

                    with st.spinner("Creating vector database..."):

                        vector_store = create_vector_store(documents)
                        retriever = create_retriever(vector_store)

                    st.session_state.retriever = retriever
                    st.session_state.video_url = youtube_url.strip()
                    st.session_state.video_language = language

                    st.success("Video processed successfully!")

                except Exception as e:

                    st.error(
                        f"Processing failed: {type(e).__name__}: {e}"
                    )

    st.divider()

    st.subheader("🌍 Supported Languages")

    st.caption(
        "English, Hindi, Kannada, Tamil, Telugu, "
        "Malayalam, Bengali, Marathi and more, "
        "depending on transcript availability."
    )

    if st.session_state.video_language:

        st.success(
            f"Transcript language: "
            f"{st.session_state.video_language}"
        )

    if st.button("🗑️ Clear Chat", use_container_width=True):
        st.session_state.chat_history = []
        st.rerun()


# ---------------------------------
# 6. Video Preview
# ---------------------------------

if st.session_state.video_url:

    st.subheader("📺 Your Video")

    st.video(st.session_state.video_url)

    st.divider()



# ---------------------------------
# 7. Chat Interface
# ---------------------------------

st.subheader("💬 Chat with your video")

if st.session_state.retriever is None:

    st.info(
        "👈 Enter a YouTube URL in the sidebar "
        "and click Process Video to get started."
    )

else:

    st.caption(
        "Ask questions in your preferred language."
    )

    # Display previous messages
    for message in st.session_state.chat_history:

        role = (
            "user"
            if isinstance(message, HumanMessage)
            else "assistant"
        )

        with st.chat_message(role):
            st.markdown(message.content)

    question = st.chat_input(
        "Ask anything about this video..."
    )

    if question:

        # Display user question
        with st.chat_message("user"):
            st.markdown(question)

        # Generate response
        with st.chat_message("assistant"):

            try:

                with st.spinner("UTube AI is thinking..."):

                    response = get_rag_response(
                        question=question,
                        chat_history=st.session_state.chat_history,
                        retriever=st.session_state.retriever
                    )

                st.markdown(response)

                # Save conversation only after successful response
                st.session_state.chat_history.append(
                    HumanMessage(content=question)
                )

                st.session_state.chat_history.append(
                    AIMessage(content=response)
                )

            except Exception as e:

                st.error(
                    f"Error: {type(e).__name__}: {e}"
                )

                st.warning(
                    "Please check your terminal for the full error."
                )



# ---------------------------------
# 8. Footer
# ---------------------------------

st.divider()

st.markdown(
    "<center>Built with ❤️ using Streamlit, "
    "LangChain, FAISS and Groq</center>",
    unsafe_allow_html=True
)
