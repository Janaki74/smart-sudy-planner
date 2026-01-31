import streamlit as st
import google.generativeai as genai

# ---------------- PAGE CONFIG ----------------
st.set_page_config(page_title="AI Study Assistant", page_icon="📘")

st.title("📘 AI Study Assistant (Gemini LLM)")
st.write("Ask any study-related question")

# ---------------- GEMINI API KEY ----------------
genai.configure(api_key="AIzaSyBuWLjhmtScv5Up29IhOhk_RdoZUXeblF0")  # <-- your key

# ✅ STABLE MODEL (WORKS 100%)
model = genai.GenerativeModel("models/gemini-1.0-pro")

# ---------------- CHAT HISTORY ----------------
if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

# ---------------- USER INPUT ----------------
question = st.chat_input("Example: What is Machine Learning?")

if question:
    st.session_state.messages.append(
        {"role": "user", "content": question}
    )

    try:
        response = model.generate_content(question)
        answer = response.text
    except Exception as e:
        answer = f"❌ Error: {str(e)}"

    st.session_state.messages.append(
        {"role": "assistant", "content": answer}
    )

    st.chat_message("assistant").write(answer)
