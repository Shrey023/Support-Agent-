import logging
import os

import streamlit as st
from dotenv import load_dotenv
from google import genai

from agent import handle_customer_question, load_support_knowledge


st.set_page_config(
    page_title="Customer Support Agent",
    page_icon="🤖",
    layout="centered",
)

st.title("🤖 Customer Support Agent")

st.caption(
    "Ask questions about support policies or check your ticket status."
)

load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    st.error("GEMINI_API_KEY is missing. Please check your .env file.")
    st.stop()


@st.cache_resource
def initialize_agent(api_key):
    client = genai.Client(api_key=api_key)
    knowledge_base = load_support_knowledge()
    return client, knowledge_base


try:
    client, knowledge_base = initialize_agent(api_key)
except Exception:
    logging.exception("Failed to initialize support agent")
    st.error(
        "The support agent could not start. "
        "Check the terminal logs and try again."
    )
    st.stop()


if "messages" not in st.session_state:
    st.session_state.messages = []


for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])


if prompt := st.chat_input("How can I help you?"):
    st.session_state.messages.append(
        {"role": "user", "content": prompt}
    )

    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Looking into your question..."):
            try:
                answer = handle_customer_question(
                    client,
                    prompt,
                    knowledge_base,
                )
            except genai.errors.APIError as error:
                logging.exception("Gemini API request failed")
                if error.code == 503:
                    answer = (
                        "The AI service is temporarily busy. "
                        "Please try again in a few seconds."
                    )
                else:
                    answer = (
                        "The AI service encountered an error. "
                        "Please try again shortly."
                    )
            except Exception:
                logging.exception("Failed to answer customer question")
                answer = (
                    "Sorry, something went wrong while processing "
                    "your question. Please try again."
                )

        st.markdown(answer)

    st.session_state.messages.append(
        {"role": "assistant", "content": answer}
    )