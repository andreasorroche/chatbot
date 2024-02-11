from langchain_community.chat_models import ChatOpenAI
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableBranch

from config import settings

model = ChatOpenAI(model_name=settings.CHAT_MODEL_NAME, temperature=0)


def generate_embeddings(text):
    """
    Creates and returns a retriever for semantic search or document retrieval by converting input texts into
    embeddings using OpenAIEmbeddings, and storing these embeddings in a FAISS vector store.

    Args:
        text (list): The text to generate embeddings for.

    Returns:
        retriever: A retriever for semantic search or document retrieval.
    """
    vectorstore = FAISS.from_texts(text, embedding=OpenAIEmbeddings())
    retriever = vectorstore.as_retriever()
    return retriever


def q_a_prompt():
    """
    Creates and returns a chat prompt template.

    Returns:
        ChatPromptTemplate: A template for structuring chat prompts.
    """
    question_answering_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "Eres un asistente inteligente. Responde en español a las preguntas del usuario. Si no sabes la "
                "respuesta, simplemente di que no lo sabes, no intentes inventar una respuesta. Si la pregunta se puede"
                "responder con sí o no, responde únicamente 'sí' o 'no', sin añadir nada más. Utiliza un máximo "
                "de tres oraciones y mantén la respuesta lo más concisa posible. "
                "Pregunta al final de la respuesta si se tiene alguna pregunta más. "
                "Responde basándote en el contexto a continuación:\n\n{context}",
            ),
            MessagesPlaceholder(variable_name="messages"),
        ]
    )

    return question_answering_prompt


def transform_prompt():
    """
    Creates and returns a chat prompt template for query transformation.

    Returns:
        ChatPromptTemplate: A template for transforming chat prompts into search queries.
    """
    query_transform_prompt = ChatPromptTemplate.from_messages(
        [
            MessagesPlaceholder(variable_name="messages"),
            (
                "user",
                "Dada la conversación anterior, genera una consulta de búsqueda para buscar información relevante a "
                "la conversación. Solo responde con la consulta, sin agregar nada más.",
            ),
        ]
    )
    return query_transform_prompt


def retriever_chain(chunks):
    """
    Constructs and returns a retriever chain for semantic search or document retrieval.

    Args:
        chunks (list): A list of chunks of text.

    Returns:
        RunnableBranch: A retriever chain for semantic search or document retrieval.
    """
    prompt = transform_prompt()
    embeddings = generate_embeddings(chunks)

    query_transforming_retriever_chain = RunnableBranch(
        (
            lambda x: len(x.get("messages", [])) == 1,
            (lambda x: x["messages"][-1].content) | embeddings,
        ),
        prompt | model | StrOutputParser() | embeddings,
    ).with_config(run_name="chat_retriever_chain")

    return query_transforming_retriever_chain


def doc_chain():
    """
    Constructs and returns a document chain.

    Returns:
        RunnablePassthrough: A document chain for processing chat prompts.
    """
    prompt = q_a_prompt()
    document_chain = create_stuff_documents_chain(model, prompt)
    return document_chain


def conversational_retrieval_chain(chunks):
    """
    Constructs and returns a conversational retrieval chain.

    Args:
        chunks (list): A list of chunks of text.

    Returns:
        RunnablePassthrough: A conversational retrieval chain for processing chat prompts.
    """
    prompt = retriever_chain(chunks)
    doc = doc_chain()

    conversational_chain = RunnablePassthrough.assign(
        context=prompt,
    ).assign(
        answer=doc,
    )
    return conversational_chain
