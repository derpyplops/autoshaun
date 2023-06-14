import streamlit as st
from streamlit_chat import message

from main import answer_question


if 'message_history' not in st.session_state:
    st.session_state['message_history'] = [
        "Hello, I am your friendly financial assistant. How can I help you today?"
    ]

# input here
question = st.text_input("Your question", key="question")
if question:
    st.session_state.message_history.append(question)
    ans = answer_question(question=question)
    st.session_state.message_history.append(ans)
    st.session_state["text"] = ""

for i, message_text in enumerate(reversed(st.session_state.message_history)):
    is_user = i % 2 == 1
    message(message_text, is_user=is_user)
