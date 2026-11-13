import streamlit as st
from dotenv import load_dotenv
import os
from langchain_core.messages import HumanMessage, AIMessage

# Load .env before importing the RAG chain
load_dotenv(override=True)

from src.transcript import (
    extract_video_id,
    extract_transcript,
    transcript_text_to_documents
)
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
    st.error("GROQ_API_KEY not found. Please check your .env file or Streamlit secrets.")
    st.stop()


# ---------------------------------
# 2. Session State
# ---------------------------------

DEFAULT_SESSION_STATE = {
    "retriever": None,
    "chat_history": [],
    "video_url": None,
    "video_language": None,
    "manual_transcript_required": False,
    "pending_video_url": None,
    "transcript_error": None
}

for key, default_value in DEFAULT_SESSION_STATE.items():
    if key not in st.session_state:
        st.session_state[key] = default_value


def build_and_store_retriever(documents, video_url, language):
    """Build the existing FAISS retriever and save the processed video state."""
    vector_store = create_vector_store(documents)
    retriever = create_retriever(vector_store)

    st.session_state.retriever = retriever
    st.session_state.video_url = video_url
    st.session_state.video_language = language
    st.session_state.chat_history = []
    st.session_state.manual_transcript_required = False
    st.session_state.pending_video_url = None
    st.session_state.transcript_error = None


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
            video_url = youtube_url.strip()
            video_id = extract_video_id(video_url)

            if not video_id:
                st.error("Invalid YouTube URL.")

            else:
                # Clear the previous video before processing a new one.
                st.session_state.retriever = None
                st.session_state.chat_history = []
                st.session_state.video_url = None
                st.session_state.video_language = None
                st.session_state.pending_video_url = video_url
                st.session_state.manual_transcript_required = False
                st.session_state.transcript_error = None

                try:
                    with st.spinner("Extracting YouTube transcript..."):
                        documents, language = extract_transcript(video_id)

                except Exception as e:
                    # Keep the URL so the user can paste the matching transcript.
                    st.session_state.manual_transcript_required = True
                    st.session_state.transcript_error = (
                        f"{type(e).__name__}: {e}"
                    )

                    st.warning(
                        "Automatic transcript extraction failed. "
                        "You can paste the transcript below to continue."
                    )

                else:
                    try:
                        with st.spinner("Creating vector database..."):
                            build_and_store_retriever(
                                documents,
                                video_url,
                                language
                            )

                        st.success("Video processed successfully!")

                    except Exception as e:
                        st.error(
                            f"Transcript was extracted, but vector database "
                            f"creation failed: {type(e).__name__}: {e}"
                        )

    # Manual transcript fallback appears only after automatic extraction fails.
    if st.session_state.manual_transcript_required:

        st.divider()
        st.subheader("📝 Paste Transcript")

        st.warning(
            "UTube AI couldn't retrieve captions automatically. "
            "Open the video on YouTube, choose **Show transcript**, "
            "copy the transcript, and paste it here."
        )

        with st.expander("Why did automatic extraction fail?"):
            st.caption(st.session_state.transcript_error or "")
            st.caption(
                "YouTube may restrict transcript requests from cloud-hosted "
                "servers. Pasting the transcript lets you continue without "
                "changing the RAG pipeline."
            )

        transcript_language = st.selectbox(
            "Transcript language",
            options=[
                ("Unknown / not sure", "unknown"),
                ("English", "en"),
                ("Hindi", "hi"),
                ("Kannada", "kn"),
                ("Tamil", "ta"),
                ("Telugu", "te"),
                ("Malayalam", "ml"),
                ("Bengali", "bn"),
                ("Marathi", "mr"),
                ("Spanish", "es"),
                ("French", "fr"),
                ("German", "de"),
                ("Japanese", "ja"),
                ("Chinese", "zh")
            ],
            format_func=lambda item: item[0],
            index=0
        )

        pasted_transcript = st.text_area(
            "Transcript text",
            placeholder=(
                "Paste the transcript copied from YouTube here. "
                "Timestamps are supported but not required."
            ),
            height=220,
            key="manual_transcript_input"
        )

        if st.button(
            "📚 Process Pasted Transcript",
            use_container_width=True
        ):

            if not pasted_transcript.strip():
                st.warning("Please paste the transcript first.")

            else:
                try:
                    with st.spinner("Preparing pasted transcript..."):
                        documents = transcript_text_to_documents(
                            pasted_transcript,
                            language=transcript_language[1]
                        )

                    with st.spinner("Creating vector database..."):
                        build_and_store_retriever(
                            documents,
                            st.session_state.pending_video_url,
                            transcript_language[1]
                        )

                    st.success("Transcript processed successfully!")
                    st.rerun()

                except Exception as e:
                    st.error(
                        f"Transcript processing failed: "
                        f"{type(e).__name__}: {e}"
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

    if st.session_state.manual_transcript_required:
        st.info(
            "Paste the transcript in the sidebar and click "
            "**Process Pasted Transcript** to start chatting."
        )
    else:
        st.info(
            "👈 Enter a YouTube URL in the sidebar "
            "and click Process Video to get started."
        )

else:

    st.caption("Ask questions in your preferred language.")

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

        with st.chat_message("user"):
            st.markdown(question)

        with st.chat_message("assistant"):

            try:
                with st.spinner("UTube AI is thinking..."):
                    response = get_rag_response(
                        question=question,
                        chat_history=st.session_state.chat_history,
                        retriever=st.session_state.retriever
                    )

                st.markdown(response)

                st.session_state.chat_history.append(
                    HumanMessage(content=question)
                )

                st.session_state.chat_history.append(
                    AIMessage(content=response)
                )

            except Exception as e:
                st.error(f"Error: {type(e).__name__}: {e}")
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
