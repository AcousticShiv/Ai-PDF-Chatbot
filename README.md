# 🤖 AI PDF Chat — RAG-Powered Document Assistant

A conversational AI application that lets you upload any PDF and ask questions about it in plain English. Built as a learning project exploring Retrieval-Augmented Generation (RAG) architecture.

🔗 **Live App:** [Click here to try it](https://ai-pdf-chatbot-r5ydxnb3ol2lyasu2w8tgi.streamlit.app)

---

## 📌 What It Does

- Upload any PDF document
- Ask questions in natural language
- Get AI-powered answers with source references and page numbers
- Supports follow-up questions with chat history context
- Confidence scoring on every answer
- Dark / Light mode toggle

---

## 🧠 How It Works (RAG Architecture)

# 🤖 AI PDF Chat — RAG-Powered Document Assistant

A conversational AI application that lets you upload any PDF and ask questions about it in plain English. Built as a learning project exploring Retrieval-Augmented Generation (RAG) architecture.

🔗 **Live App:** [Click here to try it](https://ai-pdf-chatbot-r5ydxnb3ol2lyasu2w8tgi.streamlit.app)

---

## 📌 What It Does

- Upload any PDF document
- Ask questions in natural language
- Get AI-powered answers with source references and page numbers
- Supports follow-up questions with chat history context
- Confidence scoring on every answer
- Dark / Light mode toggle

---

## 🧠 How It Works (RAG Architecture)

1. **PDF is processed** into text chunks (750 chars with 150 char overlap)
2. **Chunks are embedded** using `sentence-transformers/all-MiniLM-L6-v2`
3. **Stored in FAISS** vector database for fast similarity search
4. **User query** is matched against chunks using semantic search
5. **Top 5 chunks** are sent to LLaMA 3 with the question
6. **Answer is returned** with source page references and confidence score

---

## 🛠️ Tech Stack

| Layer | Technology |
|-------|-----------|
| Frontend / UI | Streamlit |
| LLM | LLaMA 3 8B (via OpenRouter) |
| Embeddings | sentence-transformers (HuggingFace) |
| Vector Store | FAISS (Facebook AI) |
| PDF Loader | LangChain + PyPDF |
| Text Splitting | LangChain RecursiveCharacterTextSplitter |
| API Client | OpenAI SDK (OpenRouter compatible) |
| Language | Python 3.10+ |

---

## ✨ Features

- ✅ RAG pipeline with semantic search
- ✅ Query intent detection (numeric / procedural / list / policy / general)
- ✅ Context-aware follow-up questions
- ✅ Confidence scoring (answer grounded in sources)
- ✅ Prompt injection protection
- ✅ API error handling (rate limit / timeout / auth)
- ✅ Temp file cleanup (Windows safe)
- ✅ File size guard (20MB max)
- ✅ Dark and Light mode UI
- ✅ Remove PDF clears all state dynamically

---

## 🚀 Run Locally

**1. Clone the repo**
```bash
git clone https://github.com/acousticshiv/pdf-chat-app.git
cd pdf-chat-app
```

**2. Install dependencies**
```bash
pip install -r requirements.txt
```

**3. Add your API key**

Create a `.env` file in the project folder:
