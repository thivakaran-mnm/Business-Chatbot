import streamlit as st
import json
import PyPDF2
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from torch.utils.data import DataLoader

# Step 1: Load Training Data from JSON File
with open(r'C:\Users\hp\Documents\Assigments\Sample Question Answers.json', 'r') as f:
    train_data_json = json.load(f)
    
with open(r'C:\Users\hp\Documents\Assigments\Corpus.pdf', 'rb') as file:
    pdf_reader = PyPDF2.PdfReader(file)
    corpus_text = ""
    for page in pdf_reader.pages:
        corpus_text += page.extract_text()

# Split the corpus into sentences
corpus_sentences = [sentence.strip() for sentence in corpus_text.split('.') if sentence.strip()]

# Initialize the model and precompute the embeddings
@st.cache_data
@st.cache_resource
def load_model_and_embeddings():
    model = SentenceTransformer('distilbert-base-nli-stsb-mean-tokens')
    corpus_embeddings = model.encode(corpus_sentences, convert_to_tensor=True).cpu().numpy()
    
    d = corpus_embeddings.shape[1]
    faiss_index = faiss.IndexFlatL2(d)
    faiss_index.add(corpus_embeddings)
    
    return model, faiss_index, corpus_embeddings

model, faiss_index, corpus_embeddings = load_model_and_embeddings()

# Function to get sentence embeddings using SentenceTransformer
def get_sentence_embeddings(sentences, model):
    embeddings = model.encode(sentences, convert_to_tensor=True)
    return embeddings.cpu().numpy()

# Function to retrieve the most relevant passages from the corpus
def retrieve_passages(question, index, sentences, model, top_k=1):
    question_embedding = get_sentence_embeddings([question], model)[0]
    D, I = index.search(np.array([question_embedding]), top_k)
    return [sentences[i] for i in I[0]]

# Function to handle questions and provide responses
def handle_question(question, conversation_history):
    conversation_history.append(question)
    relevant_passages = retrieve_passages(question, faiss_index, corpus_sentences, model)
    
    if relevant_passages:
        response = ". ".join(relevant_passages) + "."
        return response
    else:
        return "Please contact our business directly for more information."
    
# Initialize conversation history
if "conversation_history" not in st.session_state:
    st.session_state.conversation_history = []

# Streamlit App
st.title("Jessup Cellars Business Chatbot")
st.write("Ask any question related to our business, and I'll provide the best possible answers.")

# Display the conversation history
if st.session_state.conversation_history:
    #st.write("###")
    for i, message in enumerate(st.session_state.conversation_history):
        if i % 2 == 0:
            st.markdown(f"""
                <div style="text-align: right; background-color: #DCF8C6; padding: 10px; border-radius: 5px; margin: 5px 0;">
                    You: {message}
                </div>
                """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
                <div style="text-align: left; background-color: #ECECEC; padding: 10px; border-radius: 5px; margin: 5px 0;">
                    Bot: {message}
                </div>
                """, unsafe_allow_html=True)

# Input box and submit button in a fixed position
st.markdown(
    """
    <style>
    .css-1aumxhk {
        position: fixed;
        bottom: 0;
        width: 100%;
    }
    .chat-input {
        position: fixed;
        bottom: 0;
        width: 100%;
        display: flex;
        background-color: #f8f9fa;
        padding: 10px;
        border-top: 1px solid #ddd;
        box-shadow: 0 -1px 4px rgba(0,0,0,0.1);
        gap: 10px; /* Adds space between input and button */
    }
    .chat-input input {
        flex: 1;
        padding: 10px;
        border: 1px solid #ddd;
        border-radius: 4px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Sidebar for clearing history
st.sidebar.header("OPTIONS")
if st.sidebar.button("Clear History"):
    st.session_state.conversation_history = []
    st.experimental_rerun()


if 'user_question' not in st.session_state:
    st.session_state.text_input = ''
# Input box and submit button
with st.container():
    col1, col2 = st.columns([3, 1])
    with col1:
        user_question = st.text_input("Your Question:", key='question', value='')
    
    if st.button("Submit"):
        if user_question:
            response = handle_question(user_question, st.session_state.conversation_history)
            st.session_state.conversation_history.append(response)
            st.session_state.text_input= ""
            st.experimental_rerun()  # Rerun to update the conversation history
        else:
            st.write("Please enter a question.")
