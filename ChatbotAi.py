import streamlit as st
import requests
import json
import re

# ----------- Page Config -----------
st.set_page_config(page_title="Chat with Mitra (AI)", page_icon="🤖", layout="wide")

# ----------- Custom CSS (same as before) -----------
st.markdown(
    """
    <style>
    body { background-color: #121212; color: #E0E0E0; font-family: 'Segoe UI', sans-serif; }
    .main { background-color: #1E1E1E; padding: 20px; border-radius: 15px; min-height: 80vh; padding-bottom: 100px; }
    .chat-message { max-width: 70%; padding: 12px 16px; border-radius: 18px; margin: 8px 0; font-size: 16px; line-height: 1.4; }
    .user-message { background: linear-gradient(135deg, #1976D2, #42A5F5); color: white; margin-left: auto; border-bottom-right-radius: 5px; text-align: right; }
    .assistant-message { background: #2C2C2C; color: #E0E0E0; margin-right: auto; border-bottom-left-radius: 5px; text-align: left; }
    .header { text-align: center; font-size: 32px; font-weight: bold; background: linear-gradient(90deg, #00C9FF, #92FE9D); -webkit-background-clip: text; -webkit-text-fill-color: transparent; margin-bottom: 20px; }
    .stChatInputContainer { position: fixed; bottom: 0; left: 0; right: 0; background-color: #1E1E1E; padding: 12px 20px; border-top: 1px solid #333; z-index: 1000; }
    </style>
    """,
    unsafe_allow_html=True
)

# ----------- Header -----------
st.markdown("<div class='header'>🤖 Chat with Mitra (AI)</div>", unsafe_allow_html=True)

# ----------- Session State -----------
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Hello! I'm Mitra, your AI assistant. How can I help you today?"}
    ]

# ----------- Function: Call Gemini 2.0 Flash API -----------

def ask_gemini(prompt, api_key):
    url = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent"
    headers = {
        "Content-Type": "application/json",
        "X-goog-api-key": api_key
    }
    payload = {
        "contents": [{"parts": [{"text": prompt}]}]
    }
    
    try:
        response = requests.post(url, headers=headers, json=payload)
        response.raise_for_status()
        data = response.json()
        
        # Safely extract text
        candidates = data.get("candidates", [])
        if not candidates:
            return "⚠️ No response from Gemini."
        
        text_parts = []
        for candidate in candidates:
            content = candidate.get("content", {})
            parts = content.get("parts", [])
            for part in parts:
                if "text" in part:
                    text_parts.append(part["text"])
        
        raw_text = "\n".join(text_parts)
        
        # Clean up Markdown, extra spaces
        plain_text = re.sub(r"\*\*(.*?)\*\*", r"\1", raw_text)  # remove bold
        plain_text = re.sub(r"\n{2,}", "\n\n", plain_text)      # normalize line breaks
        
        return plain_text.strip()
    
    except Exception as e:
        return f"⚠️ Error: {e}\nFull response: {response.text}"



# ----------- Chat History Display -----------
chat_container = st.container()
with chat_container:
    for msg in st.session_state.messages:
        if msg["role"] == "user":
            st.markdown(f"<div class='chat-message user-message'>{msg['content']}</div>", unsafe_allow_html=True)
        else:
            st.markdown(f"<div class='chat-message assistant-message'>{msg['content']}</div>", unsafe_allow_html=True)

# ----------- Chat Input -----------
api_key = st.secrets["GEMINI_API_KEY"]
if prompt := st.chat_input("Type your message here..."):
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.spinner("🤖 Mitra is thinking..."):
        response = ask_gemini(prompt, api_key)

    st.session_state.messages.append({"role": "assistant", "content": response})
    st.rerun()
