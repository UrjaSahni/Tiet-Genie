import os
import streamlit as st
from dotenv import load_dotenv
from langchain.document_loaders import PyPDFLoader
from langchain.embeddings import SentenceTransformerEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain_together import ChatTogether

# Load environment variables for Together API key
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
if uploaded:
    for up in uploaded:
        with open(up.name, 'wb') as f:
            f.write(up.getbuffer())
        st.session_state.pdf_paths.append(up.name)
    st.session_state.vectorstores = init_vectorstores(st.session_state.pdf_paths)
    st.session_state.chain = None
    st.experimental_rerun()
