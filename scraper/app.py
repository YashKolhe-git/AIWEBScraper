import streamlit as st
import os
from dotenv import load_dotenv
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain

# Load environment variables
load_dotenv()

# Explicitly set OpenAI API key
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

# Initialize session state
if 'vectorstore' not in st.session_state:
    st.session_state.vectorstore = None
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

def process_url(url):
    # Load and process the webpage
    loader = WebBaseLoader(url)
    docs = loader.load()
    
    # Split the documents
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    documents = text_splitter.split_documents(docs)
    
    # Create embeddings and vectorstore
    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_documents(documents, embeddings)
    return vectorstore

def setup_chain():
    llm = ChatOpenAI(model="gpt-4")
    
    prompt = ChatPromptTemplate.from_template("""
    Answer the following question based on the provided context, which is easy for user to understand:
    <context>
    {context}
    </context>
    
    Question: {input}
    """)
    
    document_chain = create_stuff_documents_chain(llm, prompt)
    return document_chain

# Streamlit UI
st.title("Website Chat Assistant")

# URL input
url_input = st.text_input("Enter website URL:")
if url_input and st.button("Process URL"):
    with st.spinner("Processing website content..."):
        try:
            st.session_state.vectorstore = process_url(url_input)
            st.success("Website processed successfully! You can now ask questions.")
        except Exception as e:
            st.error(f"Error processing URL: {str(e)}")

# Chat interface
if st.session_state.vectorstore is not None:
    user_question = st.text_input("Ask a question about the website content:")
    
    if user_question:
        try:
            # Setup retrieval chain
            document_chain = setup_chain()
            retriever = st.session_state.vectorstore.as_retriever()
            retrieval_chain = create_retrieval_chain(retriever, document_chain)
            
            # Get response
            with st.spinner("Thinking..."):
                response = retrieval_chain.invoke({"input": user_question})
                
            # Display response
            st.write("Answer:", response['answer'])
            
            # Store in chat history
            st.session_state.chat_history.append(("You", user_question))
            st.session_state.chat_history.append(("Assistant", response['answer']))
            
        except Exception as e:
            st.error(f"Error generating response: {str(e)}")
    
    # Display chat history
    if st.session_state.chat_history:
        st.subheader("Chat History")
        for role, message in st.session_state.chat_history:
            if role == "You":
                st.write(f"👤 **You:** {message}")
            else:
                st.write(f"🤖 **Assistant:** {message}")
