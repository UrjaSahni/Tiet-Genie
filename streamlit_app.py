import os
import tempfile
import base64
import streamlit as st
from dotenv import load_dotenv
from langchain.document_loaders import PyPDFLoader
from langchain.embeddings import SentenceTransformerEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory

from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain_together import ChatTogether

# Load environment variables for Together API key
# ---------------- SETUP ----------------
load_dotenv()

# Initialize embeddings model
EMBEDDING_MODEL = 'all-MiniLM-L6-v2'
EMBEDDINGS = SentenceTransformerEmbeddings(model_name=EMBEDDING_MODEL)

@st.cache_resource
# Create or load FAISS stores for all provided PDF paths
def init_vectorstores(pdf_paths):
    stores = {}
    for path in pdf_paths:
        loader = PyPDFLoader(path)
        docs = loader.load_and_split()
        store = FAISS.from_documents(docs, EMBEDDINGS)
        stores[path] = store
    return stores

# Initialize session state defaults
if 'pdf_paths' not in st.session_state:
    st.session_state.pdf_paths = ['rules.pdf', 'Sequence Models-I.pdf']

if 'vectorstores' not in st.session_state:
    st.session_state.vectorstores = init_vectorstores(st.session_state.pdf_paths)

if 'memory' not in st.session_state:
    st.session_state.memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)

if 'chain' not in st.session_state:
    st.session_state.chain = None

# Page configuration
st.set_page_config(page_title='PDF Chatbot', layout='wide')

# Sidebar selection of PDF
def select_pdf():
    return st.sidebar.selectbox('Select a PDF to query', st.session_state.pdf_paths)

selected_pdf = select_pdf()

# Display chat header and history
st.header(f'Chatting with: {os.path.basename(selected_pdf)}')
if 'messages' not in st.session_state:
    st.session_state.messages = []
for msg in st.session_state.messages:
    st.chat_message(msg['role']).write(msg['content'])

# Chat input above PDF uploader
query = st.chat_input('Your question:')
if query:
    st.session_state.messages.append({'role': 'user', 'content': query})

    # Setup retriever
    retriever = st.session_state.vectorstores[selected_pdf].as_retriever(search_kwargs={'k': 3})
    if st.session_state.chain is None:
        together_key = os.getenv('TOGETHER_API_KEY') or os.getenv('TOGETHER_KEY')
        # Instantiate Together chat model using the correct parameter name for API key
        llm = ChatTogether(
            model='deepseek',
            temperature=0,
            api_key=together_key
        )
        st.session_state.chain = ConversationalRetrievalChain.from_llm(
            llm,
            retriever,
            memory=st.session_state.memory,
            return_source_documents=True
        )
    else:
        st.session_state.chain.retriever = retriever

    # Execute chain
    result = st.session_state.chain({'question': query})
    answer = result['answer']
    docs = result.get('source_documents', [])

    # Format source references
    refs = []
    for d in docs:
        meta = d.metadata
        page = meta.get('page') or meta.get('page_number') or 'Unknown'
        src = os.path.basename(meta.get('source', selected_pdf))
        link = f"{src}#page={page}"
        refs.append(f"[Source: {src}, page {page}]({link})")
    ref_text = "\n".join(refs)

    full_reply = f"{answer}\n\n{ref_text}"
    st.session_state.messages.append({'role': 'assistant', 'content': full_reply})
    st.chat_message('assistant').write(full_reply)

# PDF uploader at bottom
st.markdown('---')
uploaded = st.file_uploader('Upload new PDF(s)', type=['pdf'], accept_multiple_files=True)
together_api_key = os.getenv("TOGETHER_API_KEY")

st.set_page_config(page_title="Tiet-Genie 🤖", layout="wide")

# ✅ FIXED: Background + readable overlay
def set_bg_from_local(image_path):
    with open(image_path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode()
    st.markdown(f"""
    <style>
    .main > div:has(.block-container) {{
        background: linear-gradient(rgba(255,255,255,0.85), rgba(255,255,255,0.85)),
                    url("data:image/jpg;base64,{b64}") no-repeat center center fixed;
        background-size: cover;
    }}
    .stChatMessageContent, .stMarkdown {{
        color: #111 !important;
        font-weight: 500;
    }}
    </style>
    """, unsafe_allow_html=True)

set_bg_from_local("thaparbg.jpg")

# ---------------- SIDEBAR ----------------
with st.sidebar:
    st.image("TIETlogo.png", width=120)
    st.markdown("## 🤖 Tiet-Genie")
    st.markdown("How can I assist you today? 😊")
    uploaded = st.file_uploader("📎 Attach additional PDFs", type="pdf", accept_multiple_files=True)

# --------------- PRELOADED PDFs ----------------
@st.cache_resource(show_spinner="Loading TIET manuals...")
def load_preloaded():
    docs = []
    for path in ["rules.pdf", "discipline.pdf"]:
        docs.extend(PyPDFLoader(path).load())
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_documents(docs)
    embed = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    return FAISS.from_documents(chunks, embed)

vs = load_preloaded()

# --------------- USER PDFs ----------------
if uploaded:
    for up in uploaded:
        with open(up.name, 'wb') as f:
            f.write(up.getbuffer())
        st.session_state.pdf_paths.append(up.name)
    st.session_state.vectorstores = init_vectorstores(st.session_state.pdf_paths)
    st.session_state.chain = None
    st.experimental_rerun()
    docs = []
    for f in uploaded:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(f.read())
            docs.extend(PyPDFLoader(tmp.name).load())
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_documents(docs)
    embed = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    user_vs = FAISS.from_documents(chunks, embed)
    vs.merge_from(user_vs)

# --------------- RAG SETUP ----------------
retriever = vs.as_retriever(
    search_type="mmr",
    search_kwargs={"k": 4, "fetch_k": 10, "lambda_mult": 0.5}
)

llm = ChatTogether(
    model="deepseek-ai/DeepSeek-V3",
    temperature=0.2,
    together_api_key=together_api_key
)

qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    return_source_documents=True
)

# --------------- CHAT STATE ----------------
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# --------------- DISPLAY CHAT HISTORY ----------------
for msg in st.session_state.chat_history:
    with st.chat_message(msg["role"]):
        st.write(msg["message"])

# --------------- USER INPUT ----------------
user_input = st.chat_input("Ask a question...")
if user_input:
    # Show user message
    with st.chat_message("user"):
        st.write(user_input)
    st.session_state.chat_history.append({"role": "user", "message": user_input})

    # Process with QA chain
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                result = qa_chain({"query": user_input})
                response = result["result"]
            except Exception as e:
                response = f"⚠ Error: {str(e)}"
        st.write(response)
        st.session_state.chat_history.append({"role": "assistant", "message": response})
