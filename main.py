import os
import streamlit as st
import pickle
from langchain import OpenAI
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import UnstructuredURLLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS

# Streamlit Page Config
st.set_page_config(page_title="Equity Research AI Bot 🚀", layout="wide")

# Apply Custom CSS for Styling & Watermark
st.markdown("""
    <style>
        /* Background */
        .stApp { background-color: #1e1e2f; color: white; }

        /* Title & Header */
        .title-text { text-align: center; font-size: 38px; font-weight: bold; color: #ffcc00; }
        .subtitle-text { text-align: center; font-size: 22px; color: white; margin-top: -10px; }

        /* Sidebar Background */
        [data-testid="stSidebar"] { background-color: #252540 !important; }
        .sidebar-title { font-size: 20px; font-weight: bold; color: #ffcc00; }
        .api-key-label { font-size: 18px; font-weight: bold; color: white; }
        .stTextInput>div>div>input { border-radius: 8px; padding: 10px; background-color: white !important; color: black !important; font-size: 16px; }
        .stButton>button { border-radius: 8px; background-color: #ffcc00; color: black; font-weight: bold; }
        .stButton>button:hover { background-color: #ffaa00; }

        /* Sidebar Branding */
        .sidebar-branding {
            font-size: 18px;
            font-weight: bold;
            color: #ffcc00;
            text-align: center;
            margin-bottom: 15px;
            padding: 10px;
            border-radius: 5px;
            background-color: #333355;
        }

        /* Make URL labels white & properly aligned */
        .url-container {
            display: flex;
            flex-direction: column;
            gap: 5px;
        }
        .url-label {
            font-size: 16px;
            font-weight: bold;
            color: white !important;
            margin-bottom: -5px;
        }

        /* Ask Question Section */
        .question-container { display: flex; align-items: center; gap: 15px; margin-top: 20px; }
        .ask-question { font-size: 24px !important; font-weight: bold; color: white !important; }
        .status-message { font-size: 22px; font-weight: bold; color: #ffcc00; text-align: center; margin-top: 15px; }

        /* Watermark (BY Varun Reddy Gopavaram) */
        .watermark {
            position: fixed;
            bottom: 20px;
            right: 20px;
            font-size: 24px;
            font-weight: bold;
            color: #ffcc00;
            font-style: italic;
            opacity: 0.7;
        }
    </style>
""", unsafe_allow_html=True)

# Updated Title & Subtitle
st.markdown("<h1 class='title-text'>🚀 Equity Research AI | The Ultimate Financial News Bot 📊</h1>", unsafe_allow_html=True)
st.markdown("<h2 class='subtitle-text'>Analyze Markets, Trends, & Financial Insights Instantly ⚡</h2>", unsafe_allow_html=True)

# Sidebar Branding (Project by Varun Reddy Gopavaram)
st.sidebar.markdown("<p class='sidebar-branding'>📌 Project by Varun Reddy Gopavaram</p>", unsafe_allow_html=True)

# Sidebar for API Key
st.sidebar.markdown("<p class='sidebar-title'>🔐 API Key & Settings</p>", unsafe_allow_html=True)
st.sidebar.markdown("<p class='api-key-label'>Enter your OpenAI API Key:</p>", unsafe_allow_html=True)
openai_api_key = st.sidebar.text_input("", type="password")

# Show Watermark on API Key Entry Page
st.markdown("<p class='watermark'>BY Varun Reddy Gopavaram</p>", unsafe_allow_html=True)

if not openai_api_key:
    st.sidebar.error("⚠️ Please enter your API key.")
    st.stop()

# Sidebar for URLs with proper alignment
st.sidebar.markdown("<p class='sidebar-title'>🌍 News Article URLs</p>", unsafe_allow_html=True)
urls = []
for i in range(3):
    with st.sidebar:
        st.markdown(f"<p class='url-label'>🔗 URL {i+1}</p>", unsafe_allow_html=True)
        url = st.text_input("", key=f"url_{i+1}")
        urls.append(url)

process_url_clicked = st.sidebar.button("⚡ Process URLs")

file_path = 'faiss_store_openai.pkl'

# Setup OpenAI Model
llm = OpenAI(temperature=0.7, max_tokens=500, openai_api_key=openai_api_key)

# Layout Using Columns (Align "Ask a Question" Next to Avatar)
col1, col2 = st.columns([1, 6])

with col1:
    st.image("https://cdn-icons-png.flaticon.com/512/2206/2206368.png", width=120)  # AI Avatar

with col2:
    st.markdown("<p class='ask-question'>🧐 Ask a Question:</p>", unsafe_allow_html=True)

# Process URLs with Spinner
if process_url_clicked and any(urls):
    with st.spinner("🔄 Fetching & Analyzing News..."):
        # Load documents
        loader = UnstructuredURLLoader(urls=urls)
        data = loader.load()

        # Split text into chunks
        text_splitter = RecursiveCharacterTextSplitter(separators=["\n\n", "\n", ".", ","], chunk_size=1000)
        docs = text_splitter.split_documents(data)

        # Create embeddings
        embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
        vectorstore_openai = FAISS.from_documents(docs, embeddings)

        # Save FAISS index
        with open(file_path, 'wb') as f:
            pickle.dump(vectorstore_openai, f)

    st.markdown("<p class='status-message'>✅ Processing Complete! You can now ask a question.</p>", unsafe_allow_html=True)

# User Input for Questions
query = st.text_input("")

# Generate Answer
if query:
    if os.path.exists(file_path):
        with open(file_path, 'rb') as f:
            vectorstore = pickle.load(f)

        # Use Retrieval Chain
        chain = RetrievalQAWithSourcesChain.from_chain_type(llm=llm, retriever=vectorstore.as_retriever())

        # Get Response
        response = chain({"question": query}, return_only_outputs=True)

        # Show Response with Avatar
        st.markdown("### ⚡ AI's Response:")
        st.markdown(f"""
            <div style="display: flex; align-items: center;">
                <img src="https://cdn-icons-png.flaticon.com/512/2206/2206368.png" width="50" style="margin-right: 15px;">
                <span style="font-size: 18px; color: #ffcc00;">{response['answer']}</span>
            </div>
        """, unsafe_allow_html=True)
    else:
        st.error("⚠️ No data available. Please process URLs first.")

# Watermark on All Pages
st.markdown("<p class='watermark'>BY Varun Reddy Gopavaram</p>", unsafe_allow_html=True)
