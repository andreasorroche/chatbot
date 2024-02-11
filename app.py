from fastapi import HTTPException

import streamlit as st

from config import settings
from src.get_pdf_text import get_text_chunks, get_pdf_text

from src.model import conversational_retrieval_chain
from langchain.memory import ConversationSummaryBufferMemory
from langchain_openai import OpenAI
import os


def initialize_session_variables(text_chunks):
    """
    Initializes Streamlit session variables with default values.

    This function sets default values for certain keys in Streamlit's session state,
    ensuring that the initial state is defined and consistent. The keys handled include:

    - "initialized": A boolean indicating whether the session has been initialized (default: False).
    - "messages": A list to store messages in a conversation (default: []).
    - "text_chunks": A list of text chunks used for dynamic response generation (initialized with the provided
    text_chunks).
    - "responses": A list to store various responses (default: []).
    - "memory": An object representing the memory buffer for conversation summaries.

    :param text_chunks: Pre-processed list of text chunks to be used for dynamic responses.
    """
    st.session_state.setdefault("initialized", False)
    st.session_state.setdefault("messages", [])
    st.session_state.setdefault("text_chunks", text_chunks)
    st.session_state.setdefault("responses", [])
    st.session_state.setdefault(
        "memory",
        ConversationSummaryBufferMemory(
            llm=OpenAI(), max_token_limit=settings.MAX_TOKEN_LIMIT_MEMORY
        ),
    )


def initial_greeting():
    """
    Generates an initial greeting message for the user.
    :return: A personalized greeting message as a string.
    """
    greeting = (
        "¡Hola! Soy tu asistente y soy experto en Física y Astronomía. Preguntame lo que quieras."
    )
    return greeting


def generate_dynamic_chatbot_response(prompt, chunks):
    """
    Generates a response from the dynamic chatbot based on the given prompt and displays it in the chat.

    This function uses a chain model to generate a response based on the provided prompt and the text chunks
    stored in the Streamlit session state. The generated response is then appended to the session state messages
    and displayed in the chat.

    :param prompt: The user's input or prompt to which the chatbot should respond.
    :param chunks: The list of text chunks used for dynamic response generation.
    """

    conversation_chain = conversational_retrieval_chain(chunks)

    memory = st.session_state.memory

    response = conversation_chain.invoke(
        {"messages": memory.chat_memory.messages},
    )

    memory.save_context({"input": prompt}, {"output": response["answer"]})

    st.session_state.messages.append(
        {"role": "assistant", "content": response["answer"]}
    )
    st.chat_message("assistant").write(response["answer"])


def main():
    st.set_page_config(page_title="Chatbot", page_icon=":ringed_planet:")
    st.title("Chatbot  🤖:ringed_planet: :star: ")

    pdf_files = [
        filename for filename in os.listdir("/app/data") if filename.endswith(".pdf")
    ]

    if not pdf_files:
        raise HTTPException(
            status_code=404, detail="No se ha encontrado ningún PDF en el directorio."
        )
    raw_text = get_pdf_text(pdf_files)
    text_chunks = get_text_chunks(raw_text)
    initialize_session_variables(text_chunks)

    if not st.session_state["initialized"]:
        initial_message = initial_greeting()
        st.session_state["messages"] = [
            {"role": "assistant", "content": initial_message}
        ]
        st.session_state["initialized"] = True
    else:
        if "messages" not in st.session_state:
            st.session_state["messages"] = []

    for msg in st.session_state.messages:
        st.chat_message(msg["role"]).write(msg["content"])

    if prompt := st.chat_input("Escribe aquí..."):
        st.session_state.memory.chat_memory.add_user_message(prompt)
        st.session_state.messages.append({"role": "user", "content": prompt})
        st.chat_message("user").write(prompt)

        generate_dynamic_chatbot_response(prompt, text_chunks)


if __name__ == "__main__":
    main()
