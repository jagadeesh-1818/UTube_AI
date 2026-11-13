# UTube AI – Multilingual YouTube Video Chatbot

**Chat with YouTube videos using AI. Ask questions in your preferred language and receive context-aware answers grounded in video transcripts.**

UTube AI is a multilingual Retrieval-Augmented Generation (RAG) application that allows users to interact with YouTube videos through natural language conversations.

Simply provide a YouTube URL, let the application retrieve its transcript automatically, or paste the transcript manually if automatic extraction fails. Start asking questions and receive context-aware answers.

## 🚀 Live Demo

**[Try UTube AI](https://utubeai.streamlit.app/)**

## ✨ Features

- **Multilingual Support:** Processes available transcripts in multiple languages and generates responses in the language of the user's question.
- **Automatic Transcript Extraction:** Retrieves available YouTube transcripts using the YouTube Transcript API.
- **Manual Transcript Input:** Allows users to paste transcripts manually when automatic extraction fails.
- **Transcript Format Support:** Accepts plain-text and timestamped transcripts.
- **Conversational Q&A:** Supports questions and contextual follow-up questions about video content.
- **Retrieval-Augmented Generation:** Grounds generated answers in relevant transcript segments.
- **Multilingual Semantic Search:** Uses multilingual MiniLM embeddings to understand queries across supported languages.
- **MMR Retrieval:** Uses Maximum Marginal Relevance to retrieve relevant and diverse transcript chunks.
- **Timestamp-Aware Answers:** Provides video timestamps where relevant.
- **Hallucination Control:** Instructs the LLM to use transcript context and abstain when information is unavailable.
- **Interactive Interface:** Provides a conversational experience through Streamlit.

## 🛠️ Tech Stack

| Component | Technology |
|---|---|
| Programming Language | Python |
| Frontend | Streamlit |
| LLM | Groq API – GPT-OSS 120B |
| LLM Framework | LangChain |
| Embedding Model | `paraphrase-multilingual-MiniLM-L12-v2` |
| Vector Database | FAISS |
| Retrieval Strategy | Maximum Marginal Relevance (MMR) |
| Transcript Extraction | YouTube Transcript API |
| Language Detection | Langdetect |
| Environment Management | Python Dotenv |

## 🧠 System Architecture

UTube AI follows a Retrieval-Augmented Generation pipeline with automatic and manual transcript input options.

```text
                   YouTube URL
                        |
                        v
             YouTube Transcript API
                        |
                Extraction Failure?
                   /         \
                 No           Yes
                 |             |
                 |       Manual Transcript
                 |             |
                 \             /
                  \           /
                   v         v
              Transcript Processing
                        |
                        v
                  Text Chunking
                        |
                        v
              Multilingual MiniLM
                  Embeddings
                        |
                        v
                 FAISS Vector Store
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
                 MMR Retrieval
                        |
                        v
              Relevant Transcript Chunks
                        |
                        v
                     Groq LLM
                  GPT-OSS 120B
                        |
                        v
               Context-Aware Answer
```

### How It Works

1. **Video Input:** The user provides a YouTube video URL.
2. **Automatic Transcript Extraction:** The YouTube Transcript API attempts to retrieve an available transcript.
3. **Manual Transcript Fallback:** If automatic extraction fails, users can copy the transcript using YouTube's Show transcript feature and paste it into UTube AI.
4. **Transcript Processing:** Plain-text and timestamped transcripts are converted into LangChain Documents and divided into smaller overlapping chunks.
5. **Embedding Generation:** The multilingual MiniLM model converts transcript chunks into vector embeddings.
6. **Vector Storage:** FAISS stores the embeddings for efficient similarity search.
7. **Question Processing:** The user's question is language-detected and rewritten into a standalone question when conversation history is available.
8. **Retrieval:** MMR retrieves relevant and diverse transcript chunks.
9. **Answer Generation:** Groq's GPT-OSS 120B generates an answer grounded in the retrieved transcript context.

### Retrieval Configuration

The current retrieval configuration uses:

| Parameter | Value |
|---|---:|
| Retrieval strategy | MMR |
| Number of retrieved chunks (`k`) | 8 |
| Candidate chunks (`fetch_k`) | 25 |
| Diversity parameter (`lambda_mult`) | 0.7 |
| Chunk size | 1000 |
| Chunk overlap | 200 |

The MMR configuration balances relevance and diversity when selecting transcript chunks.

## 📝 Using Manual Transcript Input

YouTube may restrict automated transcript requests originating from cloud-hosted IP addresses. When this happens, UTube AI provides an alternative manual transcript input option.

**How to use it:**

1. Open the YouTube video in your browser.
2. Expand the video description and select **Show transcript**, if available.
3. Copy the transcript text.
4. Enter the YouTube URL in UTube AI and click **Process Video**.
5. If automatic extraction fails, the manual transcript input section appears.
6. Select the transcript language.
7. Paste the copied transcript and click **Process Pasted Transcript**.
8. Once processing completes, start asking questions about the video.

**Supported input:**
- Plain-text transcripts.
- Timestamped transcripts.
- SRT/VTT-style timestamped text pasted into the input area.

Timestamps are preserved where available to support timestamp-aware answers.

**Note:** Manual transcript input requires the user to obtain and provide the transcript. Videos without accessible captions or transcripts may still require another transcript source.

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
- Groq API key

### 1. Clone the Repository

```bash
git clone https://github.com/jagadeesh-1818/UTube_AI.git
cd UTube_AI
```

### 2. Create a Virtual Environment

```bash
python -m venv .venv
```

Activate the environment:

**Windows (PowerShell):**

```powershell
.venv\Scripts\Activate.ps1
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

Get your API key from the [Groq Console](https://console.groq.com/).

**Important:** Never commit your `.env` file or expose your API key publicly.

### 5. Run the Application

```bash
streamlit run streamlit_app.py
```

Open the local URL displayed in your terminal, usually:

```text
http://localhost:8501
```

## 📊 Evaluation and Benchmarking

The project includes a dedicated evaluation pipeline to measure retrieval performance and answer-generation quality.

The answer-generation benchmark uses a dataset of **18 questions** based on a YouTube video:

[How to Build a V-TAPER in 90 Days (Become UNRECOGNIZABLE)](https://youtu.be/6E_5dW13R24)

The dataset contains:
- 15 answerable questions
- 3 unanswerable questions

### Answer Generation Results

| Metric | Average Score |
|---|---:|
| Correctness | **5.00 / 5** |
| Faithfulness | **4.47 / 5** |
| Relevance | **5.00 / 5** |
| Abstention | **5.00 / 5** |

**Evaluation Summary:**

| Metric | Result |
|---|---:|
| Total questions | 18 |
| Successfully evaluated | 18 |
| Evaluation failures | 0 |
| Answerable questions | 15 |
| Unanswerable questions | 3 |

Scores are based on an LLM-as-a-judge evaluation using a 1–5 scale. They represent benchmark evaluation results and should not be interpreted as independently verified accuracy percentages.

### Retrieval Evaluation

The retrieval pipeline was evaluated using Hit Rate, Recall@K, and Mean Reciprocal Rank (MRR).

**MiniLM with MMR (K=8):**

| Metric | Score |
|---|---:|
| Hit Rate | **100.00%** |
| Recall@K | **96.67%** |
| MRR | **68.06%** |

These results measure retrieval performance on the project's evaluation dataset.

### Running Evaluations

Run generation evaluation:

```bash
python -m evals.evals_generation
```

Run retrieval evaluation:

```bash
python -m evals.evals_retrieval
```

Run answer-quality evaluation:

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

| Configuration | Value |
|---|---|
| Platform | Streamlit Community Cloud |
| Repository | `jagadeesh-1818/UTube_AI` |
| Branch | `main` |
| Main file | `streamlit_app.py` |
| API secrets | `GROQ_API_KEY` |

The application can be accessed here:

**[https://utubeai.streamlit.app/](https://utubeai.streamlit.app/)**

## ⚠️ Limitations

- Automatic transcript extraction depends on YouTube transcript availability and accessibility.
- YouTube may restrict transcript requests originating from cloud-hosted IP addresses.
- Manual transcript input is available as a fallback when automatic extraction fails.
- Manual input requires users to obtain and paste the transcript themselves.
- Videos without accessible transcripts cannot be processed unless a transcript is supplied through another source.
- Answer quality depends on transcript quality and retrieval relevance.
- The application answers questions about information available in the transcript, not everything visually presented in the video.
- Initial execution may take longer because the embedding model needs to be downloaded.

## 🔮 Future Improvements

- Improve retrieval using advanced reranking techniques.
- Add video-frame and visual-content understanding.
- Improve transcript timestamp precision.
- Expand evaluation datasets across multiple languages and video categories.
- Optimize embedding generation and vector retrieval.
- Add persistent conversation history.
- Explore more reliable transcript extraction options for cloud deployment.

## 👨‍💻 Author

**Jagadeesh B R**

GitHub: [@jagadeesh-1818](https://github.com/jagadeesh-1818)

---

⭐ If you find this project interesting, consider giving the repository a star!
