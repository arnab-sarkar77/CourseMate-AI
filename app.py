import os
import tempfile
import time
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv
from google.genai.errors import APIError
from pypdf.errors import PdfReadError
from streamlit.errors import StreamlitSecretNotFoundError

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_google_genai.chat_models import GoogleGenerativeAIError
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate

# Load environment variables from the project directory
load_dotenv(dotenv_path=Path(__file__).resolve().parent / ".env")


def get_google_api_key() -> str | None:
    try:
        secret_key = st.secrets["GOOGLE_API_KEY"]
    except (KeyError, StreamlitSecretNotFoundError):
        secret_key = None

    if secret_key:
        return str(secret_key).strip()

    environment_key = os.getenv("GOOGLE_API_KEY")
    return environment_key.strip() if environment_key else None


google_api_key = get_google_api_key()

# Page configuration
st.set_page_config(
    page_title="CourseMate AI",
    page_icon="📚",
    layout="wide"
)

# App Title and Developer Attribution
st.title("📚 CourseMate AI")
st.caption("Developed by Arnab Sarkar")

st.markdown("---")

# Session state initialization
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None
if "messages" not in st.session_state:
    st.session_state.messages = []

if not google_api_key:
    st.error("GOOGLE_API_KEY is missing. Add it to your .env file or Streamlit Secrets and restart the app.")
    st.stop()


@st.cache_resource
def load_models(_api_key: str):
    embedding_model = GoogleGenerativeAIEmbeddings(
        model="gemini-embedding-001",
        google_api_key=_api_key
    )
    llm = ChatGoogleGenerativeAI(
        model="gemini-3.8-flash",
        temperature=0,
        google_api_key=_api_key
    )
    return embedding_model, llm


embedding_model, llm = load_models(google_api_key)


def invoke_gemini(prompt_value):
    for attempt in range(3):
        try:
            return llm.invoke(prompt_value)
        except APIError as error:
            if error.code != 503:
                raise
            if attempt == 2:
                raise
            time.sleep(2 ** attempt)


def extract_response_text(content: str | list[str | dict]) -> str:
    if isinstance(content, str):
        return content.strip()

    text_parts = []
    for block in content:
        if isinstance(block, str):
            text_parts.append(block)
        elif isinstance(block, dict) and block.get("type") == "text":
            text = block.get("text")
            if isinstance(text, str):
                text_parts.append(text)

    return "\n".join(part for part in text_parts if part.strip()).strip()


# Prompt Template
prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """You are a helpful AI assistant.

Use ONLY the provided context to answer the question.

If the answer is not present in the context,
say: "I could not find the answer in the document."
"""
        ),
        (
            "human",
            """Context:
{context}

Question:
{question}
"""
        )
    ]
)

# Sidebar UI
with st.sidebar:
    st.header("📄 Upload Document")
    uploaded_file = st.file_uploader("Upload a PDF book or course material", type=["pdf"])

    if uploaded_file is not None:
        if st.button("Process Document"):
            with st.spinner("Processing PDF and building vector database..."):
                try:
                    with tempfile.TemporaryDirectory() as temp_dir:
                        tmp_file_path = Path(temp_dir) / "uploaded.pdf"
                        tmp_file_path.write_bytes(uploaded_file.getvalue())
                        docs = PyPDFLoader(str(tmp_file_path)).load()

                    chunks = RecursiveCharacterTextSplitter(
                        chunk_size=1000,
                        chunk_overlap=200
                    ).split_documents(docs)

                    if not chunks:
                        st.error("This PDF contains no readable text to search.")
                    else:
                        vectorstore = Chroma.from_documents(
                            documents=chunks,
                            embedding=embedding_model
                        )
                        st.session_state.vectorstore = vectorstore
                        st.session_state.messages = []
                        st.success("Document processed successfully! Ask questions below.")
                except APIError as error:
                    if error.code == 401:
                        st.error(
                            "Google rejected the Gemini API key. In Streamlit Cloud, replace the root-level "
                            "GOOGLE_API_KEY secret with a valid Gemini API key (not an OAuth token), save it, "
                            "and reboot the app."
                        )
                    else:
                        st.error(f"Could not process this PDF: {error}")
                except (GoogleGenerativeAIError, OSError, PdfReadError, ValueError) as error:
                    st.error(f"Could not process this PDF: {error}")

    st.markdown("---")
    st.markdown("**Developer:** Arnab Sarkar")

# Main Interface
if st.session_state.vectorstore is None:
    st.info("👈 Please upload and process a PDF document in the sidebar to get started.")
else:
    # Display previous chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Chat Input Box
    if user_query := st.chat_input("Ask a question about your uploaded document..."):
        st.session_state.messages.append({"role": "user", "content": user_query})
        with st.chat_message("user"):
            st.markdown(user_query)

        with st.chat_message("assistant"):
            with st.spinner("Searching document..."):
                try:
                    retriever = st.session_state.vectorstore.as_retriever(
                        search_type="mmr",
                        search_kwargs={"k": 4, "fetch_k": 10, "lambda_mult": 0.5}
                    )

                    docs = retriever.invoke(user_query)
                    context = "\n\n".join([doc.page_content for doc in docs])

                    final_prompt = prompt.invoke({
                        "context": context,
                        "question": user_query
                    })

                    response = invoke_gemini(final_prompt)
                    ai_answer = extract_response_text(response.content)
                    if ai_answer:
                        st.markdown(ai_answer)
                        st.session_state.messages.append({"role": "assistant", "content": ai_answer})
                    else:
                        st.error(
                            "Gemini returned no readable text. Please try asking your question again."
                        )
                except APIError as error:
                    if error.code == 503:
                        st.error(
                            "Gemini is temporarily overloaded. The app retried; "
                            "please try again in a few minutes."
                        )
                    else:
                        st.error(f"Could not get an answer from Gemini: {error}")
                except GoogleGenerativeAIError as error:
                    st.error(f"Could not get an answer from Gemini: {error}")
                except ValueError as error:
                    st.error(f"Could not get an answer from Gemini: {error}")