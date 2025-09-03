import streamlit as st
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain import hub

# Constants
PERSIST_DIRECTORY = "./chroma_db"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
GROQ_MODEL = "llama-3.3-70b-versatile"  # Or any other Llama model available on Groq
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

# key = "na....."

# Initialize session state
if "articles" not in st.session_state:
    st.session_state.articles = []  # List of dicts: {'url': url, 'title': title}
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = Chroma(
        persist_directory=PERSIST_DIRECTORY,
        embedding_function=HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    )
if "messages" not in st.session_state:
    st.session_state.messages = []
if "groq_api_key" not in st.session_state:
    st.session_state.groq_api_key = ""

# Function to add article
def add_article(url):
    try:
        loader = WebBaseLoader(url)
        docs = loader.load()
        if not docs:
            return None, "No content loaded from URL."
        
        # Extract title (simplified; assumes first doc has metadata with title)
        title = docs[0].metadata.get('title', url)
        
        # Split documents
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
        splits = text_splitter.split_documents(docs)
        
        # Add to vectorstore with metadata
        for split in splits:
            split.metadata['source_url'] = url
            split.metadata['title'] = title
        
        st.session_state.vectorstore.add_documents(splits)
        st.session_state.vectorstore.persist()
        
        # Add to articles list if not already present
        if not any(a['url'] == url for a in st.session_state.articles):
            st.session_state.articles.append({'url': url, 'title': title})
        
        return title, None
    except Exception as e:
        return None, str(e)

# Setup LLM and RAG chain
def setup_rag_chain():
    if not st.session_state.groq_api_key:
        return None, "Please enter your Groq API key in the sidebar."
    
    llm = ChatGroq(
        groq_api_key=st.session_state.groq_api_key,
        model_name=GROQ_MODEL
    )
    
    retriever = st.session_state.vectorstore.as_retriever(search_kwargs={"k": 5})
    
    prompt = hub.pull("rlm/rag-prompt")
    
    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)
    
    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    
    return rag_chain, None

# Sidebar
with st.sidebar:
    st.title("Settings")
    
    # API Key input
    groq_api_key = st.text_input("Groq API Key", type="password", value=st.session_state.groq_api_key)
    if groq_api_key:
        st.session_state.groq_api_key = groq_api_key
    
    # Articles list
    with st.expander("Articles", expanded=True):
        if not st.session_state.articles:
            st.write("No articles added yet.")
        for article in st.session_state.articles:
            st.write(f"- [{article['title']}]({article['url']})")

# Main UI
st.title("RAG Chatbot with Online Articles")

# Add new article section at the top
st.subheader("Add New Article")
url_input = st.text_input("Enter article URL:")
if st.button("Add Article"):
    if url_input:
        with st.spinner("Processing article..."):
            title, error = add_article(url_input)
            if error:
                st.error(f"Error adding article: {error}")
            else:
                st.success(f"Added article: {title}")
    else:
        st.warning("Please enter a URL.")

# Chat area
st.subheader("Chat with the Bot")

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
if prompt := st.chat_input("Ask a question about the articles..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            rag_chain, error = setup_rag_chain()
            if error:
                response = error
            else:
                try:
                    response = rag_chain.invoke(prompt)
                except Exception as e:
                    response = f"Error: {str(e)}"
        st.markdown(response)

    st.session_state.messages.append({"role": "assistant", "content": response})
