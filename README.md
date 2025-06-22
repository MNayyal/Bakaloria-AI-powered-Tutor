# 🧠 Bakaloria AI Powered Tutor

**Bakaloria AI** is a Retrieval-Augmented Generation (RAG)-based chatbot designed to help Syrian high school students prepare for national exams using only official curriculum material.

Currently, it supports the **12th-grade English curriculum** as a proof of concept. The system retrieves relevant content from the official textbooks and answers questions using a language model — ensuring that all answers are grounded in the actual material.

---

## 🚀 Motivation

Private tutoring is expensive and inaccessible to many students in Syria. This tool aims to level the playing field by giving all students free access to reliable academic support, especially for the بــكالوريا exams.

---

## 🧰 Tech Stack

- **LangChain** for building the RAG pipeline
- **ChromaDB** for vector storage
- **SentenceTransformers** for embeddings
- **Google FLAN-T5 (via HuggingFace)** as the LLM
- **Gradio** for the user interface
- **Google Colab** for development/testing

---

## 📚 Dataset

This MVP uses markdown-formatted text from the **12th-grade English Student Book** (Module 1, Unit 1). The content is split into text chunks and embedded for retrieval.

---

## 🛠 How It Works

1. **Load & Preprocess Curriculum Texts**
2. **Split into Chunks (with Overlap)**
3. **Embed using SentenceTransformers**
4. **Store in ChromaDB**
5. **Connect to FLAN-T5 via HuggingFace Pipeline**
6. **Wrap in a RetrievalQA chain using LangChain**
7. **Serve via a Gradio Chat Interface**

---

## 🖥️ Run It on Google Colab

```python
# Mount Drive
from google.colab import drive
drive.mount('/content/drive')

# Install dependencies
!pip install langchain chromadb sentence-transformers transformers accelerate gradio plotly scikit-learn python-dotenv
!pip install -U langchain-community

