# RAG LANGCHAIN QDRANT STREAMLIT

_RAG (Retrieval-augmented generation)_ demo based on: 
* **LangChain** a framework for developing applications powered by language models 
* **Qdrant** a vector similarity search engine
* **Streamlit**  a faster way to build and share data apps
* **Argilla** an open-source data curation platform for LLMs

# Qdrant 

Run as Docker
```
docker pull qdrant/qdrant
docker run --name qdrant -d -p 6333:6333 -p 6334:6334 -v $(pwd)/qdrant_storage:/qdrant/storage:z qdrant/qdrant
```

Start Web UI Dashboard

```
localhost:6333/dashboard
```

# OPENAI
Create .env file and add your OPENAI_API_KEY

```
export OPENAI_API_KEY="sk-..."
```

# Python 3.11.0
Install prerequisites

```
pip3 install -r requirements.txt
```

Load txt tto 'demo_rag' collection
```
py 0-load-txt-to-collection.py
```

Ask 'demo_rag' collection a question
```
py 1-ask-collection.py
```

# Argilla

Run as docker
```
docker run -d --name argilla -p 6900:6900 argilla/argilla-quickstart:latest
```

Add Argilla constants to .env file: 

```
export ARGILLA_API_URL = "..."
export ARGILLA_API_KEY = "..."
export ARGILLA_WORKSPACE = "..."
export ARGILLA_DATASET_NAME = "..."
```

Create Argilla dataset
```
py 2-argilla-create-dataset.py
```

# Steamlit 

Start chatbot steamlit with qdrant collection

```
streamlit run 2-steamlit-chatbot.py
```

Ask any question related to 'state_of_the_union.txt' content.

Examples:

* What is the purpose of a State of the Union address and what topics does it typically cover?
* What are some of the key issues and challenges facing the United States as mentioned in the State of the Union address?
* How does the President plan to address issues such as gun violence, voting rights, and LGBTQ+ rights in the State of the Union?


# Links
https://python.langchain.com/docs/integrations/vectorstores/qdrant

https://python.langchain.com/docs/integrations/providers/streamlit 

https://docs.argilla.io/en/latest/index.html

