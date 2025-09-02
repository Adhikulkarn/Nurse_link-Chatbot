import streamlit as st
import PyPDF2
from sentence_transformers import SentenceTransformer
import faiss
import uuid
from ibm_watsonx_ai import Credentials
from ibm_watsonx_ai.foundation_models import ModelInference
import numpy as np
import random

# ---------------- IBM Watsonx Setup ----------------
IBM_API_KEY = "3_x81uevOQViOQHejEHm1kzxrofJ8rwCris9u3dm2hg7"
IBM_PROJECT_ID = "0fbe16f0-3ef8-4ab5-8b2d-14b2fcdd198c"
WATSONX_URL = "https://us-south.ml.cloud.ibm.com"

credentials = Credentials(url=WATSONX_URL, api_key=IBM_API_KEY)
model_id = "ibm/granite-3-2b-instruct"
parameters = {
    "decoding_method": "sample",
    "max_new_tokens": 350,
    "temperature": 0.45
}
model = ModelInference(
    model_id=model_id,
    params=parameters,
    credentials=credentials,
    project_id=IBM_PROJECT_ID
)

# ---------------- PDF RAG Setup ----------------
PDF_PATH = "nurselink.pdf"

def extract_text_from_pdf(path):
    with open(path, "rb") as f:
        reader = PyPDF2.PdfReader(f)
        return "\n".join([page.extract_text() or "" for page in reader.pages])

def chunk_text(text, chunk_size=500, overlap=50):
    words = text.split()
    chunks = []
    i = 0
    while i < len(words):
        chunk = " ".join(words[i:i+chunk_size])
        chunks.append(chunk)
        i += chunk_size - overlap
    return chunks

def build_faiss_index(chunks, embed_model):
    embeddings = np.array([embed_model.encode(c) for c in chunks]).astype("float32")
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)
    return index, embeddings

def retrieve_chunks(query, embed_model, chunks, faiss_index, k=3):
    q_vec = embed_model.encode([query]).astype("float32")
    _, indices = faiss_index.search(q_vec, k)
    return [chunks[i] for i in indices[0]]

# ---------------- Greeting Setup ----------------
GREETING_KEYWORDS = ["hi", "hello", "hey", "greetings", "good morning", "good afternoon", "good evening"]
GREETING_RESPONSES = [
    "Hello! How can I assist you today?",
    "Hi there! Need help with anything?",
    "Hey! What can I do for you?",
    "Greetings! How can I help you today?"
]

# ---------------- Answer Generation ----------------
def generate_answer(query, context):
    q_lower = query.lower().strip()

    # Check for greeting
    if any(word in q_lower for word in GREETING_KEYWORDS):
        return random.choice(GREETING_RESPONSES)

    # Normal PDF-based RAG response
    prompt = f"Answer the user's question using ONLY the context and also keep your answers short do not reply in huge paragraphs:\n\nCONTEXT:\n{context}\n\nQUESTION: {query}\nAnswer:"
    try:
        response = model.generate_text(prompt=prompt)
        if isinstance(response, dict):
            return response.get('results', [{}])[0].get('generated_text', '').strip()
        elif isinstance(response, list):
            return response[0].get('generated_text', '').strip()
        return str(response)
    except Exception as e:
        return f"IBM Granite API error: {e}"

# ---------------- Streamlit UI ----------------
st.set_page_config(page_title="RAG Chatbot", layout="wide")
st.markdown("""
<style>
:root {
    --color-primary: #00a8cc;
    --color-primary-dark: #007a94;
    --color-secondary: #4ecdc4;
    --color-white: #ffffff;
    --color-dark: #2c3e50;
    --color-gray: #6c757d;
    --font-family: 'Inter', sans-serif;
    --shadow-light: 0 2px 10px rgba(0, 0, 0, 0.1);
    --shadow-medium: 0 4px 20px rgba(0, 0, 0, 0.15);
    --shadow-heavy: 0 8px 30px rgba(0, 0, 0, 0.2);
    --transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
    --gradient-teal: linear-gradient(135deg, #00a8cc, #4ecdc4);
    --gradient-card: linear-gradient(145deg, #ffffff, #f9f9f9);
    --text-gray: #6c757d;
    --border-gray: #e0e0e0;
    --light-mint: #f0fdfa;
    --primary-teal: #00a8cc;
    --white: #ffffff;
}

body {
    font-family: var(--font-family);
    background-color: var(--light-mint);
}

h1 {
    color: var(--color-primary-dark);
}

.chat-bubble {
    border-radius: 15px;
    padding: 12px 16px;
    margin: 6px;
    max-width: 70%;
    font-size: 1rem;
    transition: var(--transition);
    box-shadow: var(--shadow-light);
}

.user-bubble {
    background: var(--color-primary);
    color: var(--color-white);
    align-self: flex-end;
}

.bot-bubble {
    background: var(--gradient-card);
    color: var(--color-dark);
    align-self: flex-start;
}

.msg-row {
    display: flex;
    flex-direction: row;
}

.sidebar .stButton>button {
    background-color: var(--color-primary);
    color: var(--white);
    border-radius: 8px;
    border: none;
    padding: 6px 12px;
    margin-top: 5px;
}
</style>
""", unsafe_allow_html=True)

st.markdown("<h1 style='text-align:center'>NurseLink Help-Bot</h1>", unsafe_allow_html=True)
st.markdown("---")

# ---------------- Load PDF and build index at startup ----------------
if "rag_ready" not in st.session_state:
    st.session_state.chunks = []
    st.session_state.embed_model = SentenceTransformer("all-MiniLM-L6-v2")
    st.session_state.faiss_index = None

    st.info("Loading and processing knowledge base... This may take a few seconds.")
    pdf_text = extract_text_from_pdf(PDF_PATH)
    chunks = chunk_text(pdf_text)
    faiss_index, _ = build_faiss_index(chunks, st.session_state.embed_model)
    
    st.session_state.chunks = chunks
    st.session_state.faiss_index = faiss_index
    st.session_state.rag_ready = True
    st.success("Knowledge base loaded. You can now ask questions!")

# ---------------- Chat Functionality ----------------
if "messages" not in st.session_state:
    st.session_state.messages = []

def handle_user_input():
    query = st.session_state.msg_input.strip()
    if not query or not st.session_state.rag_ready:
        return
    relevant_chunks = retrieve_chunks(query, st.session_state.embed_model, st.session_state.chunks, st.session_state.faiss_index)
    context = "\n".join(relevant_chunks)
    answer = generate_answer(query, context)
    st.session_state.messages.append({"role": "user", "content": query})
    st.session_state.messages.append({"role": "bot", "content": answer})
    st.session_state.msg_input = ""

# ---------------- Sidebar ----------------
with st.sidebar:
    st.header("Session Info")
    st.write(f"PDF: {PDF_PATH}")
    st.write(f"Chunks: {len(st.session_state.chunks)}")
    if st.button("Reset Chat"):
        st.session_state.messages = []

# ---------------- Chat Input ----------------
col1, col2 = st.columns([6,1])
col1.text_input("Your question:", key="msg_input", on_change=handle_user_input)
col2.button("Ask", on_click=handle_user_input)

# ---------------- Display Chat ----------------
USER_ICON = "🧑"
BOT_ICON = "🤖"

for msg in st.session_state.messages:
    role = msg["role"]
    icon = USER_ICON if role == "user" else BOT_ICON
    bubble_class = "user-bubble" if role == "user" else "bot-bubble"
    row_justify = "flex-end" if role == "user" else "flex-start"
    st.markdown(
        f"<div class='msg-row' style='justify-content:{row_justify};'>"
        f"<div class='chat-bubble {bubble_class}'>{icon} {msg['content']}</div></div>",
        unsafe_allow_html=True
    )
