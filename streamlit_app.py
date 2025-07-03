import streamlit as st
from dotenv import load_dotenv
from langchain.document_loaders import PyPDFLoader
from langchain.embeddings import SentenceTransformerEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.chat_models import ChatOpenAI
import os

# Load environment variables
def load_env():
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

# Initialize session state
if 'pdf_paths' not in st.session_state:
    # Keep these two PDFs loaded by default
    st.session_state.pdf_paths = ['rules.pdf', 'Sequence Models-I.pdf']

if 'vectorstores' not in st.session_state:
    st.session_state.vectorstores = init_vectorstores(st.session_state.pdf_paths)

if 'memory' not in st.session_state:
    # Conversation memory to handle long-term context
    st.session_state.memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)

if 'chain' not in st.session_state:
    st.session_state.chain = None

# Page config
st.set_page_config(page_title='PDF Chatbot', layout='wide')

# Sidebar: select which PDF to chat with
def select_pdf():
    return st.sidebar.selectbox('Select a PDF to query', st.session_state.pdf_paths)

selected_pdf = select_pdf()

# Display chat history
st.header(f'Chatting with: {os.path.basename(selected_pdf)}')
if 'messages' not in st.session_state:
    st.session_state.messages = []
for msg in st.session_state.messages:
    if msg['role'] == 'user':
        st.chat_message('user').write(msg['content'])
    else:
        st.chat_message('assistant').write(msg['content'])

# Chat input (above PDF uploader)
query = st.chat_input('Your question:')
if query:
    st.session_state.messages.append({'role': 'user', 'content': query})

    # Retrieve chain for selected PDF
    retriever = st.session_state.vectorstores[selected_pdf].as_retriever(search_kwargs={'k': 3})
    if st.session_state.chain is None:
        llm = ChatOpenAI(temperature=0)
        st.session_state.chain = ConversationalRetrievalChain.from_llm(
            llm,
            retriever,
            memory=st.session_state.memory,
            return_source_documents=True
        )
    else:
        # update retriever dynamically
        st.session_state.chain.retriever = retriever

    # Run the chain
    result = st.session_state.chain({'question': query})
    answer = result['answer']
    docs = result.get('source_documents', [])

    # Build source references
    refs = []
    for d in docs:
        meta = d.metadata
        page = meta.get('page') or meta.get('page_number') or 'Unknown'
        src = os.path.basename(meta.get('source', selected_pdf))
        # Generate a link to the PDF (download button)
        link = f"{src}#page={page}"  # Anchor link format
        refs.append(f"[Source: {src}, page {page}]({link})")
    ref_text = "\n".join(refs)

    full = f"{answer}\n\n{ref_text}"
    st.session_state.messages.append({'role': 'assistant', 'content': full})
    st.chat_message('assistant').write(full)

# PDF uploader at the bottom
delimiter = st.markdown('---')
uploaded = st.file_uploader('Upload new PDF(s)', type=['pdf'], accept_multiple_files=True)
if uploaded:
    for up in uploaded:
        # Save to disk
        with open(up.name, 'wb') as f:
            f.write(up.getbuffer())
        st.session_state.pdf_paths.append(up.name)
    # Rebuild vectorstores with new files
    st.session_state.vectorstores = init_vectorstores(st.session_state.pdf_paths)
    # Reset chain to reinitialize with updated PDFs
    st.session_state.chain = None
    st.experimental_rerun()
