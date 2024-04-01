from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import Qdrant
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import CharacterTextSplitter

from dotenv import load_dotenv, find_dotenv

_ = load_dotenv(find_dotenv())  # read OPENAI_API_KEY from local .env file

loader = TextLoader("./data/state_of_the_union.txt", encoding='utf-8')
documents = loader.load()
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
docs = text_splitter.split_documents(documents)

embeddings = OpenAIEmbeddings()

url = "http://localhost:6333"
qdrant = Qdrant.from_documents(
    documents=docs,
    embedding=embeddings,
    url=url,
    prefer_grpc=True,
    collection_name="demo_rag",
    force_recreate=True             # will delete collection if it already exists
)
