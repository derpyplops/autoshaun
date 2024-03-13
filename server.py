import streamlit as st
from answer_question import answer_question
import time
import base64

from definitions import ASSETS_DIR


def chat_component(chat_text, avatar_path, is_user):
    alignment = 'flex-end' if is_user else 'flex-start'
    chat_html = f"""
    <div style='display: flex; justify-content: {alignment};'>
        <img src='data:image/png;base64,{base64.b64encode(open(avatar_path, "rb").read()).decode()}' style='height: 50px; width: 50px; border-radius: 50%; margin-right: 10px;'>
        <div style='background-color: #333333; color: #FFFFFF; border-radius: 5px; padding: 10px; margin: 5px; 
        max-width: 60%;'>
            {chat_text}
        </div>
    </div>
    """
    st.markdown(chat_html, unsafe_allow_html=True)


USER = ASSETS_DIR / 'generic_avatar.png'
SHAUN = ASSETS_DIR / 'shaun.png'


class Message:
    def __init__(self, text, is_user):
        self.text = text
        self.is_user = is_user
        self.timestamp = time.time()

    def __repr__(self):
        return f"Message(text={self.text}, is_user={self.is_user}, timestamp={self.timestamp})"


class Chat:
    def __init__(self, user_avatar_url, bot_avatar_url):
        self.user_avatar_url = user_avatar_url
        self.bot_avatar_url = bot_avatar_url
        self.messages = [
            Message("Hello, I'm Shaun, your friendly financial assistant. How can I help you today?", is_user=False)
        ]

    def add_user_message(self, message):
        self.messages.append(Message(message, is_user=True))

    def add_bot_message(self, message):
        self.messages.append(Message(message, is_user=False))


## Logic

if 'message_history' not in st.session_state:
    st.session_state['message_history'] = [
        "Hello, I'm Shaun, your friendly financial assistant. How can I help you today?"
    ]

if 'chat' not in st.session_state:
    st.session_state['chat'] = Chat(USER, SHAUN)

# Create a container for the chat history
chat_container = st.container()

# Display the chat history
with chat_container:
    for i, message in enumerate(st.session_state.chat.messages):
        chat_component(message.text, USER if message.is_user else SHAUN, message.is_user)

# Create a container for the chat input
input_container = st.container()

# Display the chat input at the bottom
with input_container:
    question = st.text_input("Your question", key="question")
    if question:
        st.session_state.chat.add_user_message(question)
        with st.spinner("Let me look that up for you..."):
            ans = answer_question(question=question)
        st.session_state.chat.add_bot_message(ans)
        st.session_state["text"] = ""

        # Scroll to the bottom of the chat history
        chat_container.empty()
        with chat_container:
            for i, message in enumerate(st.session_state.chat.messages):
                chat_component(message.text, USER if message.is_user else SHAUN, message.is_user)
