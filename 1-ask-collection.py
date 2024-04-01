import qdrant_client
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Qdrant

from dotenv import load_dotenv, find_dotenv

_ = load_dotenv(find_dotenv())  # read OPENAI_API_KEY from local .env file

embeddings = OpenAIEmbeddings()

client_qdrant = qdrant_client.QdrantClient( "http://localhost:6333", prefer_grpc=True)

qdrant = Qdrant(
    client=client_qdrant, 
    collection_name="demo_rag", 
    embeddings=embeddings,
)

query = "What did the president say about Ketanji Brown Jackson"
found_docs = qdrant.similarity_search(query)

print(found_docs[0].page_content)

query = "What did the president say about Ketanji Brown Jackson"
found_docs = qdrant.similarity_search_with_score(query)
document, score = found_docs[0]
print(document.page_content)
print(f"\nScore: {score}")
