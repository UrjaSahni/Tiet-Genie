import os
import tempfile
import traceback
import streamlit as st

from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain_together import ChatTogether
from dotenv import load_dotenv
load_dotenv()


# Load API key safely
together_api_key = os.getenv("TOGETHER_API_KEY")



# Streamlit page setup
st.set_page_config(page_title="PDF Chatbot", layout="centered")
st.title("📚 Tiet-Genie")
st.markdown("Ask questions based on uploaded PDFs using HuggingFace Embeddings + Together LLM.")

# Upload PDFs
uploaded_files = st.file_uploader("Upload one or more PDFs", type=["pdf"], accept_multiple_files=True)

if uploaded_files:
    all_docs = []

    for uploaded_file in uploaded_files:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(uploaded_file.read())
            tmp_path = tmp_file.name

        loader = PyPDFLoader(tmp_path)
        documents = loader.load()
        all_docs.extend(documents)

    # Split and embed
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    split_docs = text_splitter.split_documents(all_docs)

    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = FAISS.from_documents(split_docs, embeddings)

    # Retriever and LLM
    retriever = vectorstore.as_retriever(
        search_type="mmr",  # You can change to "similarity" if needed
        search_kwargs={"k": 4, "fetch_k": 10, "lambda_mult": 0.5}
    )

    llm = ChatTogether(
    model="deepseek-ai/DeepSeek-V3",
    temperature=0.2,
    together_api_key=together_api_key  # pass key explicitly
)


    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        return_source_documents=True
    )

    # Ask a question
    query = st.text_input("❓ Ask a question from the PDFs:")
    if query:
        try:
            with st.spinner("Thinking..."):
                result = qa_chain(query)
                clean_result = result["result"].replace("**", "")
                st.success("✅ Answer:")
                st.write(clean_result)

                with st.expander("📖 Source Snippets"):
                    for i, doc in enumerate(result["source_documents"]):
                        snippet = doc.page_content.replace("**", "")
                        st.markdown(f"**Snippet {i+1}:** {snippet[:500]}...")
        except Exception as e:
            st.error(f" Error: {str(e)}")
            st.text(traceback.format_exc())

