import streamlit as st
from main import answer_question
import time

def chat_component(chat_text, avatar_url, is_user):
    alignment = 'flex-end' if is_user else 'flex-start'
    chat_html = f"""
    <div style='display: flex; justify-content: {alignment};'>
        <img src='{avatar_url}' style='height: 50px; width: 50px; border-radius: 50%; margin-right: 10px;'>
        <div style='background-color: #333333; color: #FFFFFF; border-radius: 5px; padding: 10px; margin: 5px; 
        max-width: 60%;'>
            {chat_text}
        </div>
    </div>
    """
    st.markdown(chat_html, unsafe_allow_html=True)

USER = "https://i.pinimg.com/originals/75/54/a6/7554a6a61d325e31d943b2a75abeaecf.jpg"
SHAUN = "https://media.licdn.com/dms/image/C4D03AQEMosbEl8bWdA/profile-displayphoto-shrink_800_800/0/1656810215182?e=1692230400&v=beta&t=iYxkQi9av-nsIjpgamJTZrOm0Su2SqhOxuGHXtQISOs"

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
#
# input here
question = st.text_input("Your question", key="question")
if question:
    st.session_state.chat.add_user_message(question)
    with st.spinner("Let me look that up for you..."):
        ans = answer_question(question=question)
    st.session_state.chat.add_bot_message(ans)
    st.session_state["text"] = ""

for i, message in enumerate(st.session_state.chat.messages):
    chat_component(message.text, USER if message.is_user else SHAUN, message.is_user)

