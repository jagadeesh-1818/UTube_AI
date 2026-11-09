# UTube AI – Multilingual YouTube Video Chatbot

**Chat with YouTube videos using AI. Ask questions, understand video content, and receive context-aware answers grounded in video transcripts.**

UTube AI is a multilingual Retrieval-Augmented Generation (RAG) application that allows users to interact with YouTube videos through natural language conversations.

Instead of watching an entire video to find specific information, users can provide a YouTube URL and ask questions about its content. The application retrieves relevant transcript segments and generates answers using a Large Language Model.

## 🚀 Live Demo

**Streamlit App:** [https://utubeai.streamlit.app/]

## ✨ Features

- **Multilingual Support:** Process YouTube transcripts in multiple languages and answer questions in the language used by the user.
- **Conversational Q&A:** Ask questions and follow-up questions about video content.
- **Retrieval-Augmented Generation:** Retrieves relevant transcript segments before generating answers.
- **Semantic Search:** Uses BAAI/bge-m3 embeddings to understand the semantic meaning of queries.
- **MMR Retrieval:** Uses Maximum Marginal Relevance to retrieve relevant and diverse transcript chunks.
- **Timestamp-Aware Answers:** Includes transcript timestamps where relevant.
- **Hallucination Control:** Instructs the language model to answer using transcript context and abstain when information is unavailable.
- **Interactive UI:** Built with Streamlit for a simple conversational experience.

## 🛠️ Tech Stack

| Component | Technology |
|---|---|
| Programming Language | Python |
| Frontend | Streamlit |
| LLM | Groq API – GPT-OSS 120B |
| LLM Framework | LangChain |
| Embedding Model | BAAI/bge-m3 |
| Vector Database | FAISS |
| Retrieval Strategy | Maximum Marginal Relevance (MMR) |
| Transcript Extraction | YouTube Transcript API |
| Language Detection | Langdetect |
| Environment Management | Python Dotenv |

## 🧠 System Architecture

The application follows a Retrieval-Augmented Generation pipeline.

```text
             YouTube URL
                  |
                  v
       YouTube Transcript API
                  |
                  v
         Transcript Processing
                  |
                  v
          Text Chunking
                  |
                  v
       BAAI/bge-m3 Embeddings
                  |
                  v
          FAISS Vector Store
                  |
                  |
         User Question
                  |
                  v
       Language Detection
                  |
                  v
       Query Rewriting
       (when chat history exists)
                  |
                  v
        MMR-based Retrieval
                  |
                  v
       Relevant Transcript
              Chunks
                  |
                  v
           Groq LLM
        (GPT-OSS 120B)
                  |
                  v
          Context-Aware
              Answer
```

### How It Works

1. **Video Input:** The user provides a YouTube video URL.
2. **Transcript Extraction:** The YouTube Transcript API retrieves the available transcript.
3. **Text Processing:** The transcript is divided into smaller overlapping chunks.
4. **Embedding Generation:** BAAI/bge-m3 converts transcript chunks into vector embeddings.
5. **Vector Storage:** FAISS stores the embeddings for efficient similarity search.
6. **Question Processing:** The user's question is detected and, when necessary, rewritten using conversation history.
7. **Retrieval:** MMR retrieves relevant and diverse transcript chunks.
8. **Answer Generation:** Groq's GPT-OSS 120B generates an answer grounded in the retrieved transcript context.

## 📂 Project Structure

```text
UTube_AI/
│
├── src/
│   ├── main.py
│   ├── rag_chain.py
│   ├── transcript.py
│   └── vector_store.py
│
├── evals/
│   ├── dataset.py
│   ├── evals_generation.py
│   ├── evals_retrieval.py
│   ├── evaluate_answers.py
│   ├── test_regression.py
│   ├── answer_evaluation.json
│   └── generation_results.json
│
├── streamlit_app.py
├── requirements.txt
├── .gitignore
└── README.md
```

## ⚙️ Installation and Setup

### Prerequisites

- Python 3.14
- Git
- A Groq API key

### 1. Clone the Repository

```bash
git clone https://github.com/jagadeesh-1818/UTube_AI.git
cd UTube_AI
```

### 2. Create a Virtual Environment

```bash
python -m venv .venv
```

Activate it:

**Windows:**
```powershell
.venv\Scripts\activate
```

**Linux / macOS:**
```bash
source .venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure Environment Variables

Create a `.env` file in the project root:

```env
GROQ_API_KEY=your_groq_api_key
```

Get your API key from [Groq Console](https://console.groq.com/).

**Important:** Never commit your `.env` file or expose your API key publicly.

### 5. Run the Application

```bash
streamlit run streamlit_app.py
```

Open the local URL displayed in your terminal, usually:

```text
http://localhost:8501
```

## 📊 Evaluation

The application includes an evaluation pipeline to assess retrieval and answer-generation quality.

The answer-generation benchmark was tested using a dataset of 18 questions based on a YouTube video.

### Evaluation Results

| Metric | Average Score |
|---|---:|
| Correctness | 5.0 / 5 |
| Faithfulness | 4.6 / 5 |
| Relevance | 5.0 / 5 |
| Abstention | 5.0 / 5 |

**Evaluation Summary:**
- Total questions: 18
- Successfully evaluated: 18
- Evaluation failures: 0
- Answerable questions: 15
- Unanswerable questions: 3

The scores are based on an LLM-as-a-judge evaluation using a 1–5 scale. They represent benchmark evaluation results, not independently verified accuracy percentages.

### Running the Evaluation

Run the answer evaluation:

```bash
python evals/evaluate_answers.py
```

Run regression tests:

```bash
python evals/test_regression.py
```

## 🔐 Security

- API keys are stored in environment variables.
- `.env` files are excluded through `.gitignore`.
- Streamlit Community Cloud secrets are used for deployment.
- API credentials should never be committed to the repository.

## ☁️ Deployment

The application is deployed using **Streamlit Community Cloud**.

Deployment configuration:
- Repository: `jagadeesh-1818/UTube_AI`
- Branch: `main`
- Main file: `streamlit_app.py`
- Secrets: `GROQ_API_KEY`

## ⚠️ Limitations

- The application depends on YouTube transcript availability.
- Videos without accessible transcripts cannot be processed.
- Answer quality depends on transcript quality and retrieval relevance.
- The application is designed to answer questions about information present in the video transcript, not everything shown visually in the video.
- Initial execution may take longer because the embedding model needs to be downloaded.

## 🔮 Future Improvements

- Improve retrieval accuracy through advanced reranking techniques.
- Add support for video frame and visual-content understanding.
- Optimize embedding generation and vector retrieval.
- Expand evaluation datasets across different languages and video categories.
- Improve transcript timestamp precision.
- Add persistent conversation history.

## 👨‍💻 Author

**Jagadeesh B R**

GitHub: [@jagadeesh-1818](https://github.com/jagadeesh-1818)

---

⭐ If you find this project interesting, consider giving the repository a star!