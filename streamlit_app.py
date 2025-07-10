import os
import tempfile
import base64
import streamlit as st
from dotenv import load_dotenv
from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
    UnstructuredWordDocumentLoader,
    UnstructuredMarkdownLoader
)
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_together import ChatTogether
from pptx import Presentation
from fpdf import FPDF
import io

# ---------------- SETUP ----------------
load_dotenv()
together_api_key = os.getenv("TOGETHER_API_KEY")
st.set_page_config(page_title="Tiet-Genie 🤖", layout="wide")

# Dark Mode Toggle
if "dark_mode" not in st.session_state:
    st.session_state.dark_mode = False

with st.sidebar:
    st.toggle("🌙 Dark Mode", key="dark_mode")

def apply_theme():
    if st.session_state.dark_mode:
        bg_color = "#2e2e2e"
        overlay_color = "rgba(30, 30, 30, 0.95)"
        text_color = "#f1f1f1"
    else:
        bg_color = "#ffffff"
        overlay_color = "rgba(255, 255, 255, 0.82)"
        text_color = "#111111"

    st.markdown(f"""
    <style>
    .main > div:has(.block-container) {{
        background: {bg_color};
        background-size: cover;
        position: relative;
    }}
    .main > div:has(.block-container)::before {{
        content: "";
        background-color: {overlay_color};
        position: absolute;
        top: 0; left: 0;
        width: 100%; height: 100%;
        z-index: 0;
    }}
    .block-container {{
        position: relative;
        z-index: 1;
        color: {text_color};
    }}
    .stChatMessageContent, .stMarkdown {{
        color: {text_color} !important;
        font-weight: 500;
    }}
    </style>
    """, unsafe_allow_html=True)

apply_theme()

# ---------------- SIDEBAR ----------------
with st.sidebar:
    st.image("TIETlogo.png", width=120)
    st.markdown("## 🤖 Tiet-Genie")
    st.markdown("How can I assist you today? 😊")
    uploaded_files = st.file_uploader(
        "📎 Upload PDFs, DOCX, PPTX, TXT, or MD",
        type=["pdf", "docx", "pptx", "txt", "md"],
        accept_multiple_files=True
    )


# ---------------- LOAD DEFAULT PDFs ----------------
@st.cache_resource(show_spinner="Loading default PDFs...")
def load_default_vectorstore():
    default_files = ["rules.pdf", "Sequence Models-I.pdf"]
    docs = []
    for path in default_files:
        docs.extend(PyPDFLoader(path).load())

    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_documents(docs)
    embed = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    return FAISS.from_documents(chunks, embed)

vector_store = load_default_vectorstore()

# ---------------- HANDLE USER FILE UPLOADS ----------------
def load_file_to_docs(file_path, ext):
    if ext == "pdf":
        return PyPDFLoader(file_path).load()
    elif ext == "docx":
        return UnstructuredWordDocumentLoader(file_path).load()
    elif ext == "pptx":
        prs = Presentation(file_path)
        text = "\n".join(shape.text for slide in prs.slides for shape in slide.shapes if hasattr(shape, "text"))
        temp_txt = file_path + ".txt"
        with open(temp_txt, "w", encoding="utf-8") as f:
            f.write(text)
        return TextLoader(temp_txt).load()
    elif ext == "txt":
        return TextLoader(file_path, encoding="utf-8").load()
    elif ext == "md":
        return UnstructuredMarkdownLoader(file_path).load()
    return []

if uploaded_files:
    new_docs = []
    for f in uploaded_files:
        ext = f.name.split(".")[-1].lower()
        with tempfile.NamedTemporaryFile(delete=False, suffix=f".{ext}") as tmp:
            tmp.write(f.read())
            tmp_path = tmp.name
        new_docs.extend(load_file_to_docs(tmp_path, ext))

    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    new_chunks = splitter.split_documents(new_docs)
    embed = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    new_vs = FAISS.from_documents(new_chunks, embed)
    vector_store.merge_from(new_vs)

# ---------------- LLM + RETRIEVER ----------------
retriever = vector_store.as_retriever(
    search_type="mmr",
    search_kwargs={"k": 4, "fetch_k": 10, "lambda_mult": 0.5}
)

llm = ChatTogether(
    model="deepseek-ai/DeepSeek-V3",
    temperature=0.2,
    together_api_key=together_api_key
)

# ---------------- CHAT HISTORY ----------------
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "greeted" not in st.session_state:
    st.session_state.greeted = False

if not st.session_state.greeted and not st.session_state.chat_history:
    st.markdown("<h2 style='text-align:center;'>👋 Hello TIETian! How can I help you today?</h2>", unsafe_allow_html=True)

# ---------------- CHAT UI ----------------
for msg in st.session_state.chat_history:
    with st.chat_message(msg["role"]):
        st.markdown(msg["message"], unsafe_allow_html=True)

user_prompt = st.chat_input("Ask something about TIET or the lecture notes...")
if user_prompt:
    with st.chat_message("user"):
        st.write(user_prompt)
    st.session_state.chat_history.append({"role": "user", "message": user_prompt})

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                retrieved_docs = retriever.get_relevant_documents(user_prompt)

                context_text = "\n\n".join([
                    f"[Page {doc.metadata.get('page', '?')}] {doc.page_content.strip()}" for doc in retrieved_docs
                ])

                prompt_to_llm = f"""
You are an AI assistant for Thapar Institute. Use the following document snippets to answer the question. Be specific and cite relevant details.

--- DOCUMENTS ---
{context_text}

--- QUESTION ---
{user_prompt}
"""

                response_obj = llm.invoke(prompt_to_llm)
                response = response_obj.content.strip() if hasattr(response_obj, "content") else str(response_obj).strip()

                source_section = "\n\n---\n\n**📄 Source Snippets:**\n"
                for i, doc in enumerate(retrieved_docs, 1):
                    page = doc.metadata.get("page", "?")
                    snippet = doc.page_content.strip().replace("\n", " ")[:300]
                    source_section += f"- **Snippet {i} (Page {page})**: {snippet}\n"

                final_response = f"{response}\n{source_section}"
                st.markdown(final_response, unsafe_allow_html=True)
                st.session_state.chat_history.append({"role": "assistant", "message": final_response})

            except Exception as e:
                error_response = f"⚠️ Error: {str(e)}"
                st.markdown(error_response)
                st.session_state.chat_history.append({"role": "assistant", "message": error_response})

# ---------------- EXPORT CHAT HISTORY ----------------
def export_chat_history():
    chat = st.session_state.chat_history
    if not chat:
        return

    # --- PDF Export (Unicode-safe) ---
    pdf = FPDF()
    pdf.add_page()
    pdf.add_font("DejaVu", "", "DejaVuSans.ttf", uni=True)
    pdf.set_font("DejaVu", "", 12)
    pdf.set_auto_page_break(auto=True, margin=15)
    for msg in chat:
        role = "You" if msg["role"] == "user" else "Tiet-Genie"
        pdf.multi_cell(0, 10, f"{role}:\n{msg['message']}\n")
    pdf_buffer = io.BytesIO()
    pdf.output(pdf_buffer)
    pdf_buffer.seek(0)

    # --- TXT Export ---
    txt_buffer = io.StringIO()
    for msg in chat:
        role = "You" if msg["role"] == "user" else "Tiet-Genie"
        txt_buffer.write(f"{role}:\n{msg['message']}\n\n")
    txt_buffer.seek(0)

    st.sidebar.markdown("### 📤 Export Chat History")
    st.sidebar.download_button(
        "⬇️ Download as .pdf",
        data=pdf_buffer,
        file_name="chat_history.pdf",
        mime="application/pdf"
    )
    st.sidebar.download_button(
        "⬇️ Download as .txt",
        data=txt_buffer.getvalue(),
        file_name="chat_history.txt",
        mime="text/plain"
    )

export_chat_history()
