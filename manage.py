import openai
import numpy as np
import streamlit as st
from pymongo import MongoClient
from sklearn.metrics.pairwise import cosine_similarity
import os
from dotenv import load_dotenv
from time import sleep

load_dotenv()

openai.api_key = os.getenv('OPENAI_API_KEY')
# MongoDB connection
client = MongoClient(os.getenv('MONGODBURL'))
db = client['PetON']
collection = db['pet-ai-questions']

def generate_embedding(text: str) -> list:
    try:
        response = openai.Embedding.create(
            model="text-embedding-ada-002",
            input=text
        )
        return response['data'][0]['embedding']
    except Exception as e:
        print(f"Error generating embedding: {e}")
        return []

def find_best_matching_answer(user_query: str):
    query_embedding = generate_embedding(user_query)
    if not query_embedding:
        return "Error generating embedding."

    answers = collection.find({"answer_embedding": {"$exists": True}})

    best_match = None
    highest_similarity = -1

    # Iterate over documents and calculate cosine similarity between query and stored embeddings
    for answer in answers:
        stored_embedding = answer['answer_embedding']
        similarity = cosine_similarity([query_embedding], [stored_embedding])[0][0]
        if similarity > highest_similarity:
            highest_similarity = similarity
            best_match = answer

    if best_match:
        return best_match['answer_openai']
    else:
        return "Sorry, no suitable answer found."

# Streamlit Interface for Chatbot
st.title("Pet AI Chatbot")

# Initialize chat history
if 'chat_history' not in st.session_state:
    st.session_state['chat_history'] = []

# Chat history container
st.markdown("<div style='height: 60vh; overflow-y: auto; display: flex; flex-direction: column;'>", unsafe_allow_html=True)
for chat in st.session_state['chat_history']:
    if chat['role'] == 'user':
        st.markdown(
            f"<div style='text-align: right; background-color: lightblue; "
            f"border-radius: 10px; padding: 10px; margin-bottom: 5px;'>"
            f"<b>User:</b> {chat['message']}</div>",
            unsafe_allow_html=True
        )
    elif chat['role'] == 'ai':
        st.markdown(
            f"<div style='text-align: left; background-color: lightgreen; "
            f"border-radius: 10px; padding: 10px; margin-bottom: 5px;'>"
            f"<b>AI:</b> {chat['message']}</div>",
            unsafe_allow_html=True
        )
st.markdown("</div>", unsafe_allow_html=True)

# User input section at the bottom
user_query = st.text_input("Ask a question:", key="input", placeholder="Type your question here...")

# Process the user input and find the answer
if user_query:
    # Add the user's question to chat history
    st.session_state['chat_history'].append({'role': 'user', 'message': user_query})
    
    # Placeholder for streaming answer
    answer_placeholder = st.empty()
    
    # Typing effect for AI response
    answer_placeholder.markdown(
        "<div style='text-align: left; background-color: lightgreen; "
        "border-radius: 10px; padding: 10px; margin-bottom: 5px;'>"
        "<b>AI:</b> Typing...</div>",
        unsafe_allow_html=True
    )

    # Get answer to query
    answer = find_best_matching_answer(user_query)
    st.session_state['chat_history'].append({'role': 'ai', 'message': answer})
    
    # Clear loading message and display the actual answer
    answer_text = ""
    for char in answer:
        answer_text += char
        answer_placeholder.markdown(
            f"<div style='text-align: left; background-color: lightgreen; "
            f"border-radius: 10px; padding: 10px; margin-bottom: 5px;'>"
            f"<b>AI:</b> {answer_text}</div>",
            unsafe_allow_html=True
        )
        sleep(0.01)  
