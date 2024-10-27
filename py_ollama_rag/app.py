import streamlit as st
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
import ollama
import os

# Clear chat history function
def clear_chat_history():
    st.session_state.messages = [{"role": "assistant", "content": "How may I assist you?"}]

# Sidebar configuration for user settings and model parameters
with st.sidebar:
    st.title('✅🦙 Ollama Settings ⚙️')

    # User's username input
    user_username = st.text_input("Username", "Your Name")
    st.markdown("---")

    # Initialize and select LLM models dynamically from Ollama
    st.subheader('Open LLM Models')
    if "model" not in st.session_state:
        st.session_state["model"] = ""

    # Get the list of available models from Ollama and select one
    models = [model["name"] for model in ollama.list()["models"]]
    st.session_state["model"] = st.selectbox("Choose a Model via Ollama Framework", models)
    model_chosen = st.session_state["model"]

    st.markdown("---")
    st.subheader('Chat History')
    st.button('Click to Clear Chat History', on_click=clear_chat_history)

    st.markdown("---")
    st.write(f"Logged in as: {user_username}")
    st.markdown('📖 Opensource Code and ReadMe available app via this [Github Repo](https://github.com/kunalsuri/kllama/)!')

# Main app title
st.title("Document Chat with RAG and Hugging Face Embeddings")
st.write("Upload a text file, choose a model, and ask questions based on your document's content.")

# File uploader for user to upload a text file
uploaded_file = st.file_uploader("Upload your text file", type=["txt"])
if uploaded_file is not None:
    document_text = uploaded_file.read().decode("utf-8")
else:
    document_text = None

# Function to split text and create vectorstore
def ingest_text(text):
    # Initialize the text splitter and split the document
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = text_splitter.split_text(text)
    
    # Wrap each chunk in a Document object
    docs = [Document(page_content=chunk) for chunk in chunks]
    
    # Use Hugging Face Embeddings and store them in Chroma vector database
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")  # Adjust model if desired
    vectorstore = Chroma.from_documents(docs, embedding=embeddings)
    return vectorstore

# Initialize the vectorstore only if a document is uploaded
vectorstore = None
if document_text:
    with st.spinner("Processing document..."):
        vectorstore = ingest_text(document_text)
    st.success("Document ingested successfully!")

# Function to handle user queries
def ask_query(query, vectorstore, model_name):
    retriever = vectorstore.as_retriever(search_type="similarity_score_threshold", search_kwargs={"k": 3, "score_threshold": 0.5})
    docs = retriever.invoke(query)
    context = "\n".join([doc.page_content for doc in docs])

    # Create a query with context to ask the model
    response = ollama.chat(model=model_name, messages=[{"role": "user", "content": f"Question: {query} \nContext: {context}"}])
    return response['message']['content']

# Input for user query and display of response
if vectorstore is not None:
    user_query = st.text_input("Ask a question about the document:")
    if user_query:
        with st.spinner("Thinking..."):
            answer = ask_query(user_query, vectorstore, model_chosen)
        st.write("### Answer:")
        st.write(answer)
