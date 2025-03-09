# streamlit run chat.py

from dotenv import load_dotenv
from llm import get_ai_message
import streamlit as st

load_dotenv()

# streamlit 페이지 설정
st.set_page_config(
    page_title="소득세 챗봇",
    page_icon="🤖"
)
st.title("🤖 소득세 챗봇")
st.caption("소득세에 관련된 모든 것을 답해드립니다!")

# 사용자가 입력한 메세지들을 기억해야 한다.
if 'message_list' not in st.session_state:
    st.session_state.message_list = []

# 이전에 입력한 메세지들이 표시되어야 한다.
print(f"before == {st.session_state.message_list}")
for message in st.session_state.message_list:
    with st.chat_message(message["role"]):
        st.write(message["content"])

if user_question := st.chat_input(placeholder="소득세에 관련된 궁금한 내용들을 말씀해주세요!"):
    # pass
    with st.chat_message("user"):
        st.write(user_question)
    st.session_state.message_list.append({"role": "user", "content": user_question})

    with st.spinner("답변을 생성하는 중입니다"):
        ai_message = get_ai_message(user_question)
        with st.chat_message("ai"):
            st.write(ai_message)
        st.session_state.message_list.append({"role": "ai", "content": ai_message})
print(f"after == {st.session_state.message_list}")