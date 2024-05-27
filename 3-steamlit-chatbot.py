# https://qdrant.tech/documentation/examples/natural-language-search-oracle-cloud-infrastructure-cohere-langchain/

import streamlit as st
import qdrant_client
from langchain_openai import OpenAIEmbeddings
from langchain_qdrant import Qdrant
from langchain_openai import OpenAI
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
import argilla as rg
import os
from dotenv import load_dotenv, find_dotenv
from phoenix.trace.langchain import LangChainInstrumentor

_ = load_dotenv(find_dotenv())  # read OPENAI_API_KEY from local .env file

ARGILLA_API_URL = os.getenv("ARGILLA_API_URL")
ARGILLA_API_KEY = os.getenv("ARGILLA_API_KEY")
ARGILLA_WORKSPACE = os.getenv("ARGILLA_WORKSPACE")
ARGILLA_DATASET_NAME = os.getenv("ARGILLA_DATASET_NAME")

LangChainInstrumentor().instrument()

def add_argilla_record(prompt: str, response: str):
    """Add a prompt response pair to a dataset on the AI is succinct API."""
    rg.init(api_url=ARGILLA_API_URL, api_key=ARGILLA_API_KEY)
    dataset = rg.FeedbackDataset.from_argilla(name=ARGILLA_DATASET_NAME, workspace=ARGILLA_WORKSPACE)
    dataset.add_records(
        rg.FeedbackRecord(
            fields={
                "prompt": prompt,
                "response": response,
            },
        )
    )
    dataset.push_to_argilla(name=ARGILLA_DATASET_NAME, workspace=ARGILLA_WORKSPACE)


@st.cache_resource(show_spinner=False)
def initialize():
    llm_openai = OpenAI()
    embeddings = OpenAIEmbeddings()

    client_qdrant = qdrant_client.QdrantClient(
        "http://localhost:6333", prefer_grpc=True
    )
    vector_store_qdrant = Qdrant(
        client=client_qdrant,
        collection_name="demo_rag",
        embeddings=embeddings,
    )

    retriever = vector_store_qdrant.as_retriever()

    system_prompt = (
        "Use the given context to answer the question. "
        "If you don't know the answer, say you don't know. "
        "Use three sentence maximum and keep the answer concise. "
        "Context: {context}"
    )
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            ("human", "{input}"),
        ]
    )
    question_answer_chain = create_stuff_documents_chain(llm_openai, prompt)
    return create_retrieval_chain(retriever, question_answer_chain)


chain = initialize()

st.title("Ask the Document")
if "messages" not in st.session_state.keys():
    st.session_state.messages = [
        {"role": "assistant", "content": "Ask me a question !"}
    ]

if prompt := st.chat_input("Your question"):
    st.session_state.messages.append({"role": "user", "content": prompt})

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

if st.session_state.messages[-1]["role"] != "assistant":
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = chain.invoke(
                {"input": prompt},
            )
            st.write(response["answer"])
            print("\n----\n")
            print(response["answer"].strip())
            add_argilla_record(prompt, response["answer"])
            message = {"role": "assistant", "content": response["answer"]}
            st.session_state.messages.append(message)
