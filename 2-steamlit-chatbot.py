import streamlit as st
import qdrant_client
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Qdrant
from langchain.chains import RetrievalQA
from langchain_openai import OpenAI

from dotenv import load_dotenv, find_dotenv

_ = load_dotenv(find_dotenv())  # read OPENAI_API_KEY from local .env file

@st.cache_resource(show_spinner=False)
def initialize():
    embeddings = OpenAIEmbeddings()

    client_qdrant = qdrant_client.QdrantClient( "http://localhost:6333", prefer_grpc=True)
    vector_store_qdrant = Qdrant(
        client=client_qdrant, 
        collection_name="demo_rag", 
        embeddings=embeddings,
    )
    
    retriever = vector_store_qdrant.as_retriever()

    return RetrievalQA.from_chain_type(llm=OpenAI(), chain_type="stuff", retriever=retriever)

retrival_qa = initialize()

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
            response = retrival_qa.invoke(prompt)
            st.write(response["result"])
            print('\n----\n')
            print(response["result"].strip())
            message = {"role": "assistant", "content": response["result"]}
            st.session_state.messages.append(message) 