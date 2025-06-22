

from google.colab import drive
drive.mount('/content/drive')

# Install required packages for Colab
!pip install langchain chromadb sentence-transformers transformers accelerate gradio plotly scikit-learn python-dotenv

!pip install -U langchain-community

# Imports
import os
import glob
import gradio as gr

from langchain.document_loaders import DirectoryLoader, TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.schema import Document
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.llms import HuggingFaceHub
from transformers import pipeline
import torch
# Wrap the pipeline in a LangChain compatible LLM
from langchain_community.llms import HuggingFacePipeline
# 3. Update your conversation chain
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate


import numpy as np

# Set your Hugging Face token here if needed
os.environ["HUGGINGFACEHUB_API_TOKEN"] = "*******************"

# Paths

folder_path = "/content/drive/MyDrive/student book/module 1 learning for life/unit 1"


loader = DirectoryLoader(
    folder_path,
    glob="**/*.md",  # Will catch all markdown files
    loader_cls=TextLoader,
    loader_kwargs={'encoding': 'utf-8'}
)

documents = loader.load()
for doc in documents:
    doc.metadata["doc_type"] = "unit_1"

# Split into chunks
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
chunks = text_splitter.split_documents(documents)

# Embeddings using SentenceTransformers (free)
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

DB_NAME = "/content/english_unit_one_vector_db"

vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    persist_directory=DB_NAME
)

print(f"Vectorstore created with {vectorstore._collection.count()} chunks")

collection = vectorstore._collection
result = collection.get(include=['embeddings', 'documents', 'metadatas'])

vectors = np.array(result['embeddings'])
documents_text = result['documents']
doc_types = [metadata.get('doc_type', 'unknown') for metadata in result['metadatas']]
colors = ['blue' for _ in doc_types]



# 1. Initialize HF pipeline (works even with free Colab T4 GPU)
llm = pipeline(
    "text2text-generation",
    model="google/flan-t5-base",
    device="cuda" if torch.cuda.is_available() else "cpu",
    torch_dtype=torch.float16  # Saves memory
)

# 2. Test the pipeline directly
test_prompt = "Answer this: What is the capital of France?"
response = llm(test_prompt, max_length=512)[0]["generated_text"]
print("LLM response:", response)  # Should output "Paris"

# Create a prompt template to force the LLM to use retrieved context
prompt = PromptTemplate(
    input_variables=["context", "question"],
    template="Use this context to answer: {context}\nQuestion: {question}\nAnswer:"
)

# Create conversation memory and chain
memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)
retriever = vectorstore.as_retriever()

llm_chain = HuggingFacePipeline(pipeline=llm)

# 4. Modify your RAG chain
from langchain.chains import RetrievalQA

qa_chain = RetrievalQA.from_chain_type(
    llm=llm_chain,
    chain_type="stuff",
    retriever=retriever,
    chain_type_kwargs={"prompt": prompt}
)

# Chat function for Gradio
def chat(message, history):
    result = conversation_chain.invoke({"question": message})
    return result["answer"]

# 5. Update Gradio chat function
def chat(message, history):
    try:
        result = qa_chain.invoke({"query": message})
        return result["result"]
    except Exception as e:
        return f"Error: {str(e)}"

gr.ChatInterface(chat).launch(inbrowser=True)